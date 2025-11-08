from ultralytics import YOLO
import cv2
from detector import detect_anomaly

def main():
    cap = cv2.VideoCapture(0)
    model = YOLO("models/yolov8n.pt")  # YOLOv8 모델 로드

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = detect_anomaly(model, frame)

        cv2.imshow("Anomaly Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()