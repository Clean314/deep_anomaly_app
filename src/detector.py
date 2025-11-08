import cv2
import numpy as np

def detect_anomaly(model, frame):
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    # 예시: 특정 조건에서 "이상행동" 감지
    # (예: 사람 감지 수가 갑자기 많아지거나, 특정 클래스가 탐지되면)
    num_people = sum(1 for b in results[0].boxes.cls if int(b) == 0)

    if num_people > 5:
        cv2.putText(
            annotated, "⚠️ Anomaly detected!", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )

    return annotated
