import pandas as pd
import os
import glob

# --- 설정 부분 ---
# 1. 원본 데이터 파일들이 있는 폴더 경로 ('.'은 현재 폴더)
TRAIN_SOURCE_FOLDER = './CT_2025-03to06_MA'
TEST_SOURCE_FOLDER = './Test Data_MA'


# 2. 최종적으로 저장될 '온도' 데이터 파일 이름
TRAIN_OUTPUT_FILE = 'processed_temperature_only_TRAIN.csv'
TEST_OUTPUT_FILE = 'processed_temperature_only_TEST.csv'
# --- 설정 끝 ---


def run_temperature_preprocessing(source_path, output_filename):
    """
    지정된 폴더의 모든 CSV/Excel 파일을 읽어 '온도' 데이터만 전처리 후
    단일 CSV 파일로 저장합니다.
    """
    print(f"📂 데이터 파일을 검색할 폴더: {os.path.abspath(source_path)}")

    all_files = glob.glob(os.path.join(source_path, '*.csv')) + \
                glob.glob(os.path.join(source_path, '*.xlsx'))

    if not all_files:
        print("❌ 해당 폴더에 처리할 CSV 또는 엑셀 파일이 없습니다.")
        return

    print(f"✅ 총 {len(all_files)}개의 데이터 파일을 찾았습니다.")

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file) if file.endswith('.csv') else pd.read_excel(file)
            df_list.append(df)
            print(f"  - '{os.path.basename(file)}' 파일 읽는 중...")
        except Exception as e:
            print(f"  - ⚠️ '{file}' 파일 읽기 실패: {e}")

    if not df_list:
        print("❌ 파일을 읽어오지 못했습니다.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\n📊 전처리 전 전체 데이터 행: {len(combined_df)}개")

    # --- 핵심 전처리 단계 ---

    # 1. '일시' 열을 datetime 형식으로 변환하고, '온도' 열 존재 여부 확인
    if '일시' not in combined_df.columns or '온도' not in combined_df.columns:
        print("❌ '일시' 또는 '온도' 열을 찾을 수 없습니다. 파일 내용을 확인해주세요.")
        return
        
    combined_df['일시'] = pd.to_datetime(combined_df['일시'], errors='coerce')
    combined_df.dropna(subset=['일시', '온도'], inplace=True) # 날짜 또는 온도 오류 데이터 제거

    # 2. 필요한 '일시', '온도' 열만 선택
    temp_df = combined_df[['일시', '온도']].copy()

    # 3. '일시'를 기준으로 그룹화하고, '온도'는 평균값 계산
    print("🔄 '온도' 데이터의 중복 시간을 평균값으로 처리하는 중...")
    processed_df = temp_df.groupby('일시')['온도'].mean().reset_index()

    # 4. '일시'를 기준으로 오름차순 정렬
    processed_df.sort_values(by='일시', inplace=True)

    # 5. '온도' 값을 소수점 2자리까지만 표시하도록 반올림
    processed_df['온도'] = processed_df['온도'].round(2)

    print(f"📊 전처리 후 최종 데이터 행: {len(processed_df)}개")

    # 6. 최종 결과를 CSV 파일로 저장
    try:
        processed_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n✨ 처리가 완료되었습니다! '{output_filename}' 파일이 생성되었습니다.")
        print("이 파일은 '일시'와 '온도' 두 개의 열만 가집니다.")
    except Exception as e:
        print(f"\n❌ 파일 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    run_temperature_preprocessing(TRAIN_SOURCE_FOLDER, TRAIN_OUTPUT_FILE)
    run_temperature_preprocessing(TEST_SOURCE_FOLDER, TEST_OUTPUT_FILE)
