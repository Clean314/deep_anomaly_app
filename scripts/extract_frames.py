# [scripts/extract_frames.py]
import os
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from difflib import SequenceMatcher
# -------------------------------
# 설정
# -------------------------------
CSV_PATHS = [
    ("./data/manifests/train.csv", "./data/frames/train/"),
    ("./data/manifests/val.csv", "./data/frames/val/"),
    ("./data/manifests/test.csv", "./data/frames/test/")
]
VIDEO_DIR = Path("./data/videos").resolve() # 절대 경로
FRAME_INTERVAL = 3
FRAME_NAME_FORMAT = "{video_stem}_{frame:06d}.jpg"
# -------------------------------
# 헬퍼 함수
# -------------------------------
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)
def find_video_file(video_name_hint):
    """
    video_name_hint: CSV에 적힌 video_path (예: "day/12-2_cam02_assault01_place09_day_spring.mp4")
    → 실제 파일 탐색 (확장자 무시, 이름 유사도 기반)
    모든 videos 하위 폴더(예: day, night, croki)의 비디오를 대상으로 검색하여 사용 보장
    """
    hint_path = Path(video_name_hint.replace("\\", "/"))
    stem = hint_path.stem  # "12-2_cam02_assault01_place09_day_spring"
    # 전체 VIDEO_DIR 아래 모든 비디오 파일 검색 (서브폴더 포함)
    video_files = list(VIDEO_DIR.rglob("*.*"))
    video_files = [f for f in video_files if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
    if not video_files:
        return None
    # 1. 정확한 매칭 (전체 경로 고려)
    exact_path = VIDEO_DIR / hint_path
    if exact_path.exists():
        return exact_path
    # 2. stem 정확한 매칭
    for f in video_files:
        if f.stem == stem:
            return f
    # 3. 부분 매칭 (계절 등 제거)
    clean_stem = stem.replace("_spring", "").replace("_summer", "").replace("_fall", "").replace("_winter", "").replace("_day", "").replace("_night", "").strip("_")
    for f in video_files:
        if clean_stem in f.stem or f.stem.startswith(clean_stem):
            return f
    # 4. 가장 유사한 매칭 (SequenceMatcher 사용, threshold 0.8)
    best_match = None
    best_ratio = 0
    for f in video_files:
        ratio = SequenceMatcher(None, stem, f.stem).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = f
    if best_ratio > 0.8:
        print(f"유사 매칭 사용: {video_name_hint} → {best_match.name} (유사도: {best_ratio:.2f})")
        return best_match
    else:
        print(f"매칭 실패: {video_name_hint} (최고 유사도: {best_ratio:.2f})")
        return None
def extract_frames_from_clip(row, output_root):
    video_hint = row["video_path"]
    label = str(row["action_label"]).lower()
    domain = str(row["domain"]).lower()
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])
    # 실제 비디오 파일 찾기
    video_path = find_video_file(video_hint)
    if not video_path:
        print(f"경고: 비디오 찾기 실패 → {video_hint}")
        return 0, video_hint
    save_dir = Path(output_root) / domain / label
    safe_mkdir(save_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"경고: 비디오 열기 실패 → {video_path}")
        return 0, video_hint
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame >= total_frames:
        end_frame = min(end_frame, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    saved_count = 0
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = FRAME_NAME_FORMAT.format(video_stem=video_path.stem, frame=current_frame)
        frame_path = save_dir / frame_filename
        if cv2.imwrite(str(frame_path), frame):
            saved_count += 1
        current_frame += FRAME_INTERVAL
        if current_frame > end_frame:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    cap.release()
    return saved_count, video_hint
# -------------------------------
# 메인 실행
# -------------------------------
def main():
    total_extracted = 0
    missing_videos = set()
    print(f"비디오 디렉토리: {VIDEO_DIR}\n")
    for csv_path, output_root in CSV_PATHS:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"경고: CSV 없음 → {csv_path}")
            continue
        safe_mkdir(output_root)
        df = pd.read_csv(csv_path)
        print(f"{csv_path.name} 처리 중 → {len(df)}개 클립")
        domain_counts = df["domain"].value_counts().sort_index()
        print(" 도메인 분포:")
        for d, c in domain_counts.items():
            print(f" - {d}: {c}개")
        extracted = 0
        pbar = tqdm(df.iterrows(), total=len(df), desc="프레임 추출")
        for _, row in pbar:
            saved, hint = extract_frames_from_clip(row, output_root)
            extracted += saved
            if saved == 0:
                missing_videos.add(hint)
        print(f" 완료: {extracted}장 저장\n")
        total_extracted += extracted
    # 최종 요약
    print(f"전체 프레임 추출 완료! 총 {total_extracted}장")
    print(f"출력 위치: {Path('./data/frames').resolve()}")
    # 누락된 비디오 진단
    if missing_videos:
        print(f"\n경고: 아래 {len(missing_videos)}개 비디오는 파일을 찾지 못했습니다:")
        for v in sorted(missing_videos)[:10]:
            print(f" → {v}")
        if len(missing_videos) > 10:
            print(f" ... 외 {len(missing_videos) - 10}개")
        print("\n해결 방법:")
        print(" 1. XML의 <filename>과 실제 파일명 일치 확인")
        print(" 2. 확장자 누락 시 .mp4 추가")
        print(" 3. 또는 아래 스크립트로 자동 매핑 생성")
    else:
        print(f"\n성공: 모든 비디오 파일이 정상 처리됨!")
# -------------------------------
# 실행
# -------------------------------
if __name__ == "__main__":
    main()