# scripts/train_yolov8.py
from ultralytics import YOLO

# 1.YOLO 모델 불러오기 (기본 n-size)

model = YOLO("yolov8n.pt")

# 2. 학습 실행
results = model.train(
    data="data/violence.yaml",   # YAML 설정 파일
    epochs=100,                  # 학습 epoch 수
    imgsz=640,                   # 입력 이미지 크기
    batch=16,                    # 배치 크기
    workers=4,                   # 데이터로더 병렬 수 (CPU 코어 따라 조정)
    project="runs/violence_train",  # 결과 저장 폴더
    name="yolov8_violence",      # 세션 이름
    patience=20,                 # 조기 종료 patience (validation 개선 없을 때)
    optimizer="AdamW",           # 옵티마이저 선택
    pretrained=True              # COCO 사전학습 가중치 사용
)

# 3. 최종 결과 출력
print("\n학습 완료!")
print(f"best model weights: {results.save_dir}/weights/best.pt")
print(f"final metrics: {results.metrics}")
