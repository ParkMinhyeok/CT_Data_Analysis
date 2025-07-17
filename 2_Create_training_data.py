import pandas as pd
import json

# --- 설정 ---
# 1. Label Studio에서 Export한 JSON 파일 이름
ANNOTATION_FILE = 'labeled_data.json'
# 2. 전처리 완료된 원본 시계열 데이터 파일 이름
ORIGINAL_DATA_FILE = 'processed_temperature_only_TRAIN.csv'
# 3. 최종적으로 저장될 학습용 데이터 파일 이름
TRAINING_DATA_FILE = 'training_data.csv'
# --- 설정 끝 ---


def create_dataset(annotation_path, original_data_path, output_path):
    """
    Label Studio의 라벨링 결과와 원본 시계열 데이터를 병합하여
    최종 학습용 데이터셋(CSV)을 생성합니다.
    """
    print("1. 파일 로딩 중...")
    # 라벨링 결과(JSON) 로드
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 원본 시계열 데이터(CSV) 로드
    # '일시' 열을 datetime 객체로 변환하여 로드하는 것이 중요합니다.
    df = pd.read_csv(original_data_path, parse_dates=['일시'])
    print(f"  - 원본 데이터: {len(df)} 행")

    print("2. 라벨링 정보 처리 중...")
    # 'label' 열을 새로 만들고 기본값을 'Rest'로 설정
    # Label Studio에서 'Rest' 라벨의 값을 확인하고 맞게 수정하세요. (e.g., 'Rest', 0 등)
    df['label'] = 'Rest' 

    # 각 라벨링 결과(annotation)를 순회
    for task in annotations:
        # 'result' 키가 없는 경우는 건너뜀 (라벨링이 안된 데이터)
        if not task.get('annotations') or not task['annotations'][0].get('result'):
            continue
            
        for result in task['annotations'][0]['result']:
            value = result['value']
            start_time = pd.to_datetime(value['start'])
            end_time = pd.to_datetime(value['end'])
            label_name = value['timeserieslabels'][0]

            # 시작 시간과 종료 시간 사이에 있는 모든 데이터의 라벨을 변경
            mask = (df['일시'] >= start_time) & (df['일시'] <= end_time)
            df.loc[mask, 'label'] = label_name
            print(f"  - '{label_name}' 라벨 적용: {start_time} ~ {end_time}")

    # 3. 최종 학습 데이터셋 저장
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✨ 성공! 최종 학습 데이터가 '{output_path}' 파일로 저장되었습니다.")
        # 결과 확인
        print("\n[라벨별 데이터 수]")
        print(df['label'].value_counts())
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")


if __name__ == "__main__":
    create_dataset(ANNOTATION_FILE, ORIGINAL_DATA_FILE, TRAINING_DATA_FILE)