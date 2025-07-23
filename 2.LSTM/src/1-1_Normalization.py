import os
import pandas as pd

# 처리할 파일들이 있는 폴더 경로
folder_path = 'CT_2025-03to06_MA'

# 폴더 내의 모든 CSV 파일 목록을 가져옵니다.
try:
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
except FileNotFoundError:
    print(f"오류: '{folder_path}' 폴더를 찾을 수 없습니다. 스크립트와 동일한 디렉토리에 있는지 확인하세요.")
    exit()

# 처리된 데이터를 담을 빈 리스트를 생성합니다.
all_data = []

# 각 CSV 파일을 순회하며 데이터를 처리합니다.
for file in file_list:
    file_path = os.path.join(folder_path, file)
    try:
        # CSV 파일을 읽어옵니다. 한글 인코딩 문제 발생 시 'cp949'를 사용합니다.
        df = pd.read_csv(file_path, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8')

    # '일시'와 '온도' 컬럼만 선택합니다.
    df_temp = df[['일시', '온도']].copy()

    # '일시' 컬럼을 datetime 형식으로 변환합니다.
    df_temp['일시'] = pd.to_datetime(df_temp['일시'])

    # '일시'를 기준으로 그룹화하여 '온도'를 정규화합니다.
    # (x - min) / (max - min) 공식을 사용합니다.
    # 만약 하루에 데이터가 하나만 있어 max와 min이 같을 경우 0으로 처리합니다.
    df_temp['정규화_온도'] = df_temp.groupby(df_temp['일시'].dt.date)['온도'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
    )

    # 처리된 데이터를 리스트에 추가합니다.
    all_data.append(df_temp)

# 리스트에 있는 모든 데이터프레임을 하나로 병합합니다.
merged_df = pd.concat(all_data, ignore_index=True)

# 병합된 데이터프레임을 새로운 CSV 파일로 저장합니다.
output_filename = 'processed_temperature_only_TRAIN.csv'
merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"'{output_filename}' 파일이 성공적으로 생성되었습니다.")