# [scripts/split_dataset.py]
import pandas as pd
from sklearn.model_selection import train_test_split
import os
# -------------------------------
# 경로 설정
# -------------------------------
INPUT_CSV = "./data/manifests/violence_clips_manifest.csv"
TRAIN_CSV = "./data/manifests/train.csv"
VAL_CSV = "./data/manifests/val.csv"
TEST_CSV = "./data/manifests/test.csv"
# 분할 비율
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# -------------------------------
# 메인 분할 함수
# -------------------------------
def split_and_save():
    # 1. 입력 CSV 존재 확인
    if not os.path.exists(INPUT_CSV):
        print(f"오류: 입력 CSV 파일 없음 → {INPUT_CSV}")
        return
    # 2. CSV 로드
    print(f"CSV 로드 중: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"총 클립 수: {len(df)}개")
    print("도메인 분포 (전체):")
    print(df["domain"].value_counts().sort_index())
    # 3. 도메인 유효성 검사
    valid_domains = {"day", "night", "croki"}
    df = df[df["domain"].isin(valid_domains)]  # unknown 제외
    if df.empty:
        print("오류: 유효한 도메인 클립 없음!")
        return
    unknown_count = len(df[df["domain"] == "unknown"])
    if unknown_count > 0:
        print(f"경고: 'unknown' 도메인 {unknown_count}개 → 학습에서 제외됨")
    # 4. train / temp 분리 (stratify by domain)
    print(f"\nTrain ({TRAIN_RATIO:.0%}) vs Temp ({VAL_RATIO+TEST_RATIO:.0%}) 분리 중...")
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["domain"],
        random_state=42
    )
    # 5. temp → val / test 분리
    relative_val_ratio = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_ratio),
        stratify=temp_df["domain"],
        random_state=42
    )
    # 6. 출력 디렉토리 생성
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
    # 7. CSV 저장
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    # 8. 결과 요약
    print("\n데이터 분리 완료!")
    print(f"{'분할':<6} {'클립 수':>8} {'비율':>6}")
    print("-" * 22)
    print(f"{'Train':<6} {len(train_df):>8} {len(train_df)/len(df):.1%}")
    print(f"{'Val':<6} {len(val_df):>8} {len(val_df)/len(df):.1%}")
    print(f"{'Test':<6} {len(test_df):>8} {len(test_df)/len(df):.1%}")
    print("\n도메인별 분포 (Train):")
    print(train_df["domain"].value_counts().sort_index())
    print("\n도메인별 분포 (Val):")
    print(val_df["domain"].value_counts().sort_index())
    print("\n도메인별 분포 (Test):")
    print(test_df["domain"].value_counts().sort_index())
    print(f"\n저장 완료:")
    print(f" → {TRAIN_CSV}")
    print(f" → {VAL_CSV}")
    print(f" → {TEST_CSV}")
# -------------------------------
# 실행
# -------------------------------
if __name__ == "__main__":
    split_and_save()