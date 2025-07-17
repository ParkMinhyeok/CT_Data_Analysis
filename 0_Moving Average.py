import os
import pandas as pd

def process_csv_files(source_folder, output_folder):
    """
    지정된 폴더의 모든 CSV 파일에 이동 평균을 적용하고 다른 폴더에 저장합니다.
    이 버전은 한글 깨짐 방지를 위해 'utf-8-sig' 인코딩을 사용합니다.

    :param source_folder: 원본 CSV 파일이 있는 폴더 경로
    :param output_folder: 처리된 파일을 저장할 폴더 경로
    """
    # 이동 평균 윈도우 크기 및 샘플링 비율 설정
    window_size = 30
    sample_rate = 5
    target_column = '온도'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더를 생성했습니다.")

    if not os.path.exists(source_folder):
        print(f"오류: '{source_folder}' 폴더를 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return

    files_to_process = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    if not files_to_process:
        print(f"'{source_folder}'에 처리할 CSV 파일이 없습니다.")
        return

    print(f"총 {len(files_to_process)}개의 파일을 처리합니다...")

    for filename in files_to_process:
        try:
            source_path = os.path.join(source_folder, filename)
            df = pd.read_csv(source_path)

            df_sampled = df.iloc[::sample_rate, :].reset_index(drop=True)
            df_sampled[target_column] = df_sampled[target_column].rolling(window=window_size, min_periods=1).mean()

            base_name, extension = os.path.splitext(filename)
            new_filename = f"{base_name}_MA{extension}"
            output_path = os.path.join(output_folder, new_filename)

            # --- 👇 여기를 수정했습니다! ---
            # 한글이 깨지지 않도록 encoding='utf-8-sig' 옵션을 추가하여 저장합니다.
            df_sampled.to_csv(output_path, index=False, encoding='utf-8-sig')
            # ---------------------------

            print(f"✅ '{filename}' 파일 처리 완료 -> '{new_filename}'")

        except Exception as e:
            print(f"❌ '{filename}' 파일 처리 중 오류 발생: {e}")

    print("\n모든 작업이 완료되었습니다.")


# --- 코드 실행 ---
if __name__ == '__main__':
    source_directory = 'CT_2025-03to06'  # 원본 파일이 있는 폴더
    output_directory = f"{source_directory}_MA" # 결과물을 저장할 폴더

    process_csv_files(source_directory, output_directory)


    source_directory = 'Test Data'  # 원본 파일이 있는 폴더
    output_directory = f"{source_directory}_MA" # 결과물을 저장할 폴더

    process_csv_files(source_directory, output_directory)


