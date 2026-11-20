import threading
import time

import cv2
import numpy as np
import pythoncom
import win32com.client
from ultralytics import YOLO


model = YOLO("yolo11n.pt")
depth_model = YOLO("yolo26n-depth.pt")
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 웹캠 연결 상태를 확인하세요.")

MAX_OBJECT_DISTANCE_M = 2.0
VOICE_WARNING_REPEAT_SEC = 1.5  # 음성이 끝난 후 다음 반복까지 대기 시간(초)
speech_condition = threading.Condition()
current_warning = ""
running = True


def speech_worker():
    global running
    # 백그라운드 스레드에서 Windows 음성 엔진(COM)을 안정적으로 쓰기 위한 필수 초기화
    pythoncom.CoInitialize()
    try:
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Rate = 2      # 말하기 속도 (-10 ~ 10, 기본값 0, 2 정도가 적당히 빠름)
        speaker.Volume = 100  # 볼륨 (0 ~ 100)

        while True:
            with speech_condition:
                while running and not current_warning:
                    speech_condition.wait()
                if not running:
                    break
                message = current_warning

            try:
                # 동기 방식으로 음성이 끝날 때까지 재생
                speaker.Speak(message)
            except Exception as exc:
                print(f"음성 출력 오류: {exc}")

            # 음성이 끝난 후 지정된 시간(VOICE_WARNING_REPEAT_SEC)만큼 대기 후 다시 확인
            with speech_condition:
                next_repeat_time = time.monotonic() + VOICE_WARNING_REPEAT_SEC
                while running and current_warning:
                    remaining = next_repeat_time - time.monotonic()
                    if remaining <= 0:
                        break
                    speech_condition.wait(timeout=remaining)
    finally:
        pythoncom.CoUninitialize()


speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()
previous_time = time.perf_counter()

try:
    while True:
        success, frame = camera.read()
        if not success:
            raise RuntimeError("웹캠 프레임을 읽을 수 없습니다.")

        result = model(frame, imgsz=480, conf=0.5, verbose=False)[0]
        depth_result = depth_model(frame, imgsz=480, verbose=False)[0]
        if depth_result.depth is None:
            raise RuntimeError("깊이 추정 결과를 가져올 수 없습니다.")

        depth_map = depth_result.depth.data.cpu().numpy()
        height, width = frame.shape[:2]
        if depth_map.shape != (height, width):
            depth_map = cv2.resize(
                depth_map,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        boxes = result.boxes
        warning = ""
        nearby_objects = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                roi_x1 = max(0, int(x1 + (x2 - x1) * 0.35))
                roi_x2 = min(width, int(x2 - (x2 - x1) * 0.35))
                roi_y1 = max(0, int(y1 + (y2 - y1) * 0.35))
                roi_y2 = min(height, int(y2 - (y2 - y1) * 0.35))
                roi_depth = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
                valid_depth = roi_depth[
                    np.isfinite(roi_depth) & (roi_depth > 0)
                ]
                if valid_depth.size == 0:
                    continue

                distance_m = float(np.median(valid_depth))
                if distance_m <= MAX_OBJECT_DISTANCE_M:
                    nearby_objects.append(
                        (
                            (x2 - x1) * (y2 - y1),
                            box,
                            (x1, y1, x2, y2),
                            distance_m,
                        )
                    )

        if nearby_objects:
            _, largest_box, (x1, y1, x2, y2), distance_m = max(
                nearby_objects,
                key=lambda item: item[0],
            )
            label = model.names[int(largest_box.cls[0].item())]
            confidence = float(largest_box.conf[0].item())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Largest: {label} {confidence:.0%} {distance_m:.1f}m",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            warning = f"경고! {distance_m:.1f}미터 앞에 {label}이 있습니다."
        else:
            cv2.putText(
                frame,
                "No object within 2m",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        with speech_condition:
            if current_warning != warning:
                current_warning = warning
                speech_condition.notify()

        current_time = time.perf_counter()
        fps = 1 / max(current_time - previous_time, 1e-5)
        previous_time = current_time
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.imshow("Largest Object Warning", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    with speech_condition:
        running = False
        speech_condition.notify_all()
    speech_thread.join(timeout=1.0)
    camera.release()
    cv2.destroyAllWindows()
#.