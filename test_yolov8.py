# scripts/test_yolov8.py
from ultralytics import YOLO

# 1. 학습된 모델 가중치 경로 지정
best_model_path = "runs/violence_train/yolov8_violence/weights/best.pt"

# 2. 모델 불러오기
model = YOLO(best_model_path)

# 3. 테스트 세트 평가
results = model.val(
    data="data/violence.yaml",    # 같은 데이터 구성 파일
    split="test",                 # test.csv 사용
    imgsz=640,
    batch=16,
    conf=0.25,                    # confidence threshold
    iou=0.5                       # IoU threshold
)

# 4. 결과 요약 출력
print("\n테스트 완료!")
print(f"Precision: {results.results_dict['metrics/precision(B)']:.3f}")
print(f"Recall: {results.results_dict['metrics/recall(B)']:.3f}")
print(f"mAP@50: {results.results_dict['metrics/mAP50(B)']:.3f}")
print(f"mAP@50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")

print(f"결과 저장 위치: {results.save_dir}")
