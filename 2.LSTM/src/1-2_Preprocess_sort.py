import pandas as pd

# 정렬할 파일 이름
input_filename = '1_processed_temperature_only_TRAIN.csv'

# 정렬 후 저장할 새 파일 이름
output_filename = 'sorted_temperature_data.csv'

try:
    # CSV 파일을 읽어옵니다.
    df = pd.read_csv(input_filename)

    # '일시' 컬럼을 datetime 형식으로 변환합니다.
    df['일시'] = pd.to_datetime(df['일시'])

    # '일시'를 기준으로 데이터를 오름차순으로 정렬합니다.
    df_sorted = df.sort_values(by='일시')

    # 정렬된 데이터를 새 CSV 파일로 저장합니다. (index=False는 불필요한 숫자 인덱스를 제외합니다)
    df_sorted.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"✅ 파일 정렬 완료! '{output_filename}' 파일을 사용하세요.")

except FileNotFoundError:
    print(f"🚨 오류: '{input_filename}' 파일을 찾을 수 없습니다. 스크립트와 같은 폴더에 있는지 확인하세요.")
except Exception as e:
    print(f"🚨 오류 발생: {e}")