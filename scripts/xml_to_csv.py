# [scripts/xml_to_csv.py]
import os
import csv
import xml.etree.ElementTree as ET
from collections import Counter
# -------------------------------
# 경로 설정
# -------------------------------
XML_DIR = "./data/annotations_xml" # XML 폴더
OUTPUT_CSV = "./data/manifests/violence_clips_manifest.csv" # 출력 CSV
# -------------------------------
# XML 파싱: 하나의 XML → 클립 리스트
# -------------------------------
def parse_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 1. video_path 추출 (예: "croki/video1.mp4")
        filename_tag = root.find("filename")
        if filename_tag is None or not filename_tag.text:
            print(f"경고: <filename> 태그 없음 → 건너뜀: {xml_path}")
            return []
        video_path = filename_tag.text.strip().replace("\\", "/") # 통일된 슬래시
        # 2. 도메인 추출 (filename 우선)
        video_folder = os.path.dirname(video_path).split("/")[0] # "croki", "day", "night"
        valid_domains = {"day", "night", "croki"}
        domain_from_path = video_folder if video_folder in valid_domains else None
        # 3. XML <header><time> 태그 보조 확인
        header = root.find("header")
        time_tag = None
        if header is not None:
            time_elem = header.find("time")
            if time_elem is not None and time_elem.text:
                time_tag = time_elem.text.strip().lower()
        # 4. 최종 도메인 결정: path > time 태그 > unknown
        if domain_from_path:
            domain = domain_from_path
        elif time_tag in valid_domains:
            domain = time_tag
        else:
            domain = "unknown"
            print(f"도메인 불명: {video_path} → domain=unknown")
        # 5. 클립 추출
        clips = []
        for obj in root.findall("object"):
            for action in obj.findall("action"):
                action_name_tag = action.find("actionname")
                if action_name_tag is None or not action_name_tag.text:
                    continue
                label = action_name_tag.text.strip().lower()
                for frame_tag in action.findall("frame"):
                    start_tag = frame_tag.find("start")
                    end_tag = frame_tag.find("end")
                    if start_tag is None or end_tag is None:
                        continue
                    try:
                        start = int(start_tag.text)
                        end = int(end_tag.text)
                    except (ValueError, TypeError):
                        continue
                    if start > end:
                        continue # 잘못된 프레임 범위
                    clips.append([video_path, start, end, label, domain])
        return clips
    except ET.ParseError as e:
        print(f"XML 파싱 오류: {xml_path} → {e}")
        return []
    except Exception as e:
        print(f"예기치 못한 오류: {xml_path} → {e}")
        return []
# -------------------------------
# 모든 XML → 하나의 CSV
# -------------------------------
def xmls_to_csv(xml_dir, output_csv):
    if not os.path.exists(xml_dir):
        print(f"오류: XML 디렉토리 없음 → {xml_dir}")
        return
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    all_clips = []
    xml_files = [f for f in os.listdir(xml_dir) if f.lower().endswith(".xml")]
    print(f"총 {len(xml_files)}개의 XML 파일 처리 중...")
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        clips = parse_xml(xml_path)
        all_clips.extend(clips)
    # CSV 저장
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "start_frame", "end_frame", "action_label", "domain"])
        writer.writerows(all_clips)
    # 결과 요약
    print(f"\nCSV 생성 완료: {output_csv}")
    print(f"총 클립 수: {len(all_clips)}개")
    if all_clips:
        domains = [clip[4] for clip in all_clips]
        domain_counts = dict(Counter(domains))
        print("도메인별 분포:")
        for d in sorted(domain_counts.keys()):
            print(f" - {d}: {domain_counts[d]}개")
    else:
        print("경고: 클립이 하나도 추출되지 않음!")
# -------------------------------
# 실행
# -------------------------------
if __name__ == "__main__":
    xmls_to_csv(XML_DIR, OUTPUT_CSV)