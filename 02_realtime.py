import time

import cv2
from ultralytics import YOLO


model = YOLO("yolo26n.pt")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다. 웹캠 연결 상태를 확인하세요.")

previous_time = time.perf_counter()

try:
    while True:
        success, frame = camera.read()
        if not success:
            raise RuntimeError("웹캠 프레임을 읽을 수 없습니다.")

        result = model(frame, classes=[0], conf=0.8, verbose=False)[0]
        annotated_frame = result.plot()
        current_time = time.perf_counter()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("YOLO Realtime", annotated_frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
