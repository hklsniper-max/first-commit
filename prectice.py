import time
import threading
import queue
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import pyttsx3

device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = (device == "cuda")

# 1. 모델 로드
yolo_model = YOLO("yolo11n.pt")

depth_model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
image_processor = AutoImageProcessor.from_pretrained(depth_model_id)
depth_model = AutoModelForDepthEstimation.from_pretrained(
    depth_model_id,
    torch_dtype=torch.float16 if use_fp16 else torch.float32
).to(device)
depth_model.eval()

# 2. 웹캠 설정 (DirectShow 백엔드로 윈도우 웹캠 지연 최소화)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 카메라 버퍼 최소화 (화면 밀림 방지)

CALIBRATION_FACTOR = 1.0
OBSTACLE_DISTANCE_M = 1.5
OBSTACLE_DANGER_M = 0.8
OBSTACLE_TOP_IGNORE_RATIO = 0.20

# 스레드 간 공유 변수 및 Lock
lock = threading.Lock()
latest_frame = None
cached_depth_map = None
running = True
speech_queue = queue.Queue()
VOICE_WARNING_COOLDOWN_SEC = 4.0
last_voice_warning_time = 0.0
last_voice_warning_key = ""


def speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.setProperty("volume", 1.0)
    except Exception as exc:
        print(f"음성 엔진을 시작할 수 없습니다: {exc}")
        return

    while running:
        message = speech_queue.get()
        if message is None:
            break
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as exc:
            print(f"음성 출력 오류: {exc}")


# --- [백그라운드 스레드] Depth 추론 전담 ---
def depth_worker():
    global latest_frame, cached_depth_map, running
    while running:
        with lock:
            if latest_frame is None:
                frame_to_process = None
            else:
                frame_to_process = latest_frame.copy()

        if frame_to_process is None:
            time.sleep(0.005)
            continue

        h, w, _ = frame_to_process.shape
        rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = image_processor(images=pil_image, return_tensors="pt").to(device)

        if use_fp16:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        depth_result = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1).float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()

        with lock:
            cached_depth_map = depth_result


# 백그라운드 스레드 시작
worker_thread = threading.Thread(target=depth_worker, daemon=True)
worker_thread.start()
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

prev_time = time.time()

# --- [메인 스레드] 실시간 영상 출력 및 YOLO 객체 탐지 ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # 최신 프레임을 백그라운드 스레드에 전달
    with lock:
        latest_frame = frame
        current_depth = cached_depth_map

    # YOLO는 멈춤 없이 매 프레임 즉시 실행
    results = yolo_model(frame, imgsz=480, conf=0.5, verbose=False)
    detections = results[0].boxes

    obstacles = []
    frame_status = "No nearby obstacle"

    if current_depth is not None:
        usable_depth = current_depth[
            int(h * OBSTACLE_TOP_IGNORE_RATIO):h,
            :
        ]
        zone_width = usable_depth.shape[1] // 3
        zone_depth_maps = [
            usable_depth[:, :zone_width],
            usable_depth[:, zone_width:zone_width * 2],
            usable_depth[:, zone_width * 2:],
        ]
        zone_names = ["Left", "Center", "Right"]
        obstacles = []

        for zone_name, zone_depth in zip(zone_names, zone_depth_maps):
            valid_depth = zone_depth[
                np.isfinite(zone_depth) & (zone_depth > 0.05)
            ]
            if valid_depth.size == 0:
                continue

            nearest_distance = float(np.percentile(valid_depth, 5))
            if nearest_distance <= OBSTACLE_DISTANCE_M:
                obstacles.append((zone_name, nearest_distance))

        frame_status = (
            "  ".join(
                f"{zone}: {distance:.1f}m"
                for zone, distance in obstacles
            )
            if obstacles
            else "No nearby obstacle"
        )

    # 객체 탐지는 Depth 준비 여부와 관계없이 매 프레임 실행하고 표시합니다.
    nearby_objects = []
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label_name = yolo_model.names[int(box.cls[0])]
        confidence = float(box.conf[0])

        box_w, box_h = x2 - x1, y2 - y1
        cx1, cx2 = max(0, int(x1 + box_w * 0.35)), min(w, int(x2 - box_w * 0.35))
        cy1, cy2 = max(0, int(y1 + box_h * 0.35)), min(h, int(y2 - box_h * 0.35))

        if current_depth is not None:
            roi_depth = current_depth[cy1:cy2, cx1:cx2]
            valid_roi_depth = roi_depth[
                np.isfinite(roi_depth) & (roi_depth > 0.05)
            ]
            distance_m = (
                float(np.median(valid_roi_depth)) * CALIBRATION_FACTOR
                if valid_roi_depth.size > 0
                else 0.0
            )
        else:
            distance_m = 0.0

        center_x = (x1 + x2) / 2
        if center_x < w * 0.33:
            direction = "Left"
        elif center_x > w * 0.66:
            direction = "Right"
        else:
            direction = "Center"

        color = (0, 0, 255) if distance_m > 0 and distance_m <= OBSTACLE_DISTANCE_M else (0, 255, 0)
        if distance_m > 0 and distance_m <= OBSTACLE_DISTANCE_M:
            nearby_objects.append((label_name, direction, distance_m))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 1)
        if distance_m > 0:
            text = f"{label_name} {confidence:.0%}: {distance_m:.1f}m [{direction}]"
        else:
            text = f"{label_name} {confidence:.0%}: depth waiting"
        cv2.putText(frame, text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if nearby_objects:
        nearest_by_key = {}
        for label_name, direction, distance_m in nearby_objects:
            key = (label_name, direction)
            if key not in nearest_by_key or distance_m < nearest_by_key[key]:
                nearest_by_key[key] = distance_m

        warning_parts = [
            f"{label_name} on the {direction.lower()}, {distance_m:.1f} meters"
            for (label_name, direction), distance_m in list(nearest_by_key.items())[:3]
        ]
        warning_message = "Warning! " + "; ".join(warning_parts)
        warning_key = "|".join(
            f"{label_name}:{direction}:{distance_m:.1f}"
            for (label_name, direction), distance_m in nearest_by_key.items()
        )
    elif obstacles:
        warning_parts = [
            f"Obstacle on the {direction.lower()}, {distance_m:.1f} meters"
            for direction, distance_m in obstacles
        ]
        warning_message = "Warning! " + "; ".join(warning_parts)
        warning_key = "|".join(
            f"{direction}:{distance_m:.1f}"
            for direction, distance_m in obstacles
        )
    else:
        warning_message = ""
        warning_key = ""

    if warning_message:
        current_time = time.time()
        if (
            current_time - last_voice_warning_time >= VOICE_WARNING_COOLDOWN_SEC
        ):
            speech_queue.put(warning_message)
            last_voice_warning_time = current_time
            last_voice_warning_key = warning_key

    curr_time = time.time()
    fps = 1 / max(1e-5, (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(frame, f"Smooth FPS: {fps:.1f} ({device.upper()})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    if current_depth is not None:
        if obstacles:
            nearest_distance = min(distance for _, distance in obstacles)
            status_color = (
                (0, 0, 255)
                if nearest_distance <= OBSTACLE_DANGER_M
                else (0, 165, 255)
            )
            cv2.rectangle(frame, (5, 42), (w - 5, h - 5), status_color, 4)
        else:
            status_color = (0, 255, 0)

        cv2.putText(
            frame,
            f"Obstacle: {frame_status}",
            (10, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        depth_min = float(np.percentile(current_depth, 2))
        depth_max = float(np.percentile(current_depth, 98))
        depth_range = max(depth_max - depth_min, 1e-6)
        depth_normalized = np.clip(
            (current_depth - depth_min) / depth_range,
            0.0,
            1.0,
        )
        depth_u8 = (depth_normalized * 255).astype(np.uint8)
        depth_view = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
        cv2.putText(depth_view, "Depth Map", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        for zone_name, zone_distance in obstacles:
            zone_index = zone_names.index(zone_name)
            zone_x = zone_index * (w // 3)
            zone_color = (
                (0, 0, 255)
                if zone_distance <= OBSTACLE_DANGER_M
                else (0, 165, 255)
            )
            cv2.rectangle(
                depth_view,
                (zone_x, int(h * OBSTACLE_TOP_IGNORE_RATIO)),
                (zone_x + w // 3, h - 1),
                zone_color,
                3,
            )
            cv2.putText(
                depth_view,
                f"{zone_name}: {zone_distance:.1f}m",
                (zone_x + 8, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
    else:
        depth_view = np.zeros_like(frame)
        cv2.putText(depth_view, "Depth Loading...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined_view = cv2.hconcat([frame, depth_view])
    cv2.imshow("YOLO Object Detection | Depth", combined_view)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

worker_thread.join(timeout=1.0)
speech_queue.put(None)
speech_thread.join(timeout=1.0)
cap.release()
cv2.destroyAllWindows()
