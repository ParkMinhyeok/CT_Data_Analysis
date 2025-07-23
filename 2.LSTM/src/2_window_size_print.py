import json
import pandas as pd

# 다운로드한 JSON 파일 경로
json_filepath = '2_labeled_dataset.json'

with open(json_filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

durations = []

for task in data:
    # 'result' 키가 없는 경우 건너뛰기
    if not task.get('annotations') or not task['annotations'][0].get('result'):
        continue
        
    for result in task['annotations'][0]['result']:
        # 'timeserieslabels' 타입의 라벨만 처리
        if result['type'] == 'timeserieslabels':
            start_time_str = result['value']['start']
            end_time_str = result['value']['end']
            
            # 문자열을 datetime 객체로 변환
            start_time = pd.to_datetime(start_time_str)
            end_time = pd.to_datetime(end_time_str)
            
            # 구간의 길이를 초 단위로 계산
            duration_seconds = (end_time - start_time).total_seconds()
            durations.append(duration_seconds)

# pandas Series로 변환하여 통계 확인
durations_series = pd.Series(durations)

print("라벨링된 구간 길이 통계 (초 단위):")
print(durations_series.describe())