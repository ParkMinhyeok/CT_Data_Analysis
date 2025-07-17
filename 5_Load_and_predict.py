import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import joblib

def format_duration(td):
    # Timedelta 객체를 'OO시간 OO분' 형태로 변환합니다.
    if pd.isnull(td):
        return "0시간 0분"
    total_minutes = int(td.total_seconds()) // 60
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}시간 {minutes}분"

# --- 폰트 및 스타일 설정 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Malgun Gothic'

# --- 파일 불러오기 ---
MODEL_FILENAME = 'xgboost_model.joblib'
ENCODER_FILENAME = 'label_encoder.joblib'
NEW_DATA_FILE = 'processed_temperature_only_TEST.csv'

try:
    loaded_model = joblib.load(MODEL_FILENAME)
    loaded_le = joblib.load(ENCODER_FILENAME)
    plot_df = pd.read_csv(NEW_DATA_FILE, parse_dates=['일시'])
except FileNotFoundError:
    print(f"필수 파일({MODEL_FILENAME}, {ENCODER_FILENAME}, {NEW_DATA_FILE})을 찾을 수 없습니다.")
    exit()


# --- 모델 예측 ---
X_new = plot_df[['온도']]
predictions_encoded = loaded_model.predict(X_new)
plot_df['predicted_label'] = loaded_le.inverse_transform(predictions_encoded)
plot_df.sort_values(by='일시', inplace=True)


# --- 4. 'Run' 가동 사이클 분석 ---
print("\n" + "="*50)
print("🔄 'Run' 가동 구간 분석")
print("="*50)

is_run_start = (plot_df['predicted_label'] == 'Run') & (plot_df['predicted_label'].shift(1) == 'Rest')
plot_df['run_cycle_block'] = is_run_start.cumsum()
run_only_df = plot_df[plot_df['predicted_label'] == 'Run'].copy()

if run_only_df.empty or run_only_df['run_cycle_block'].max() == 0:
    print("'Run'으로 예측된 가동 구간이 없습니다.")
else:
    total_net_duration_sum = pd.Timedelta(0)
    for block_num, group in run_only_df.groupby('run_cycle_block'):
        if block_num == 0 or len(group) < 2:
            continue
        
        start_time = group['일시'].min()
        end_time = group['일시'].max()
        net_duration = group['일시'].diff().sum()
        total_net_duration_sum += net_duration

        # ✨✨✨ 출력 형식 수정 ✨✨✨
        # 위에서 정의한 헬퍼 함수를 사용하여 시간 형식을 변환
        duration_formatted = format_duration(net_duration)
        print(f"  [Run 사이클 {block_num}] 시작: {start_time}, 종료: {end_time}, 가동시간: {duration_formatted}")
        # ✨✨✨✨✨✨✨✨✨✨✨✨

    print("-" * 50)
    # ✨✨✨ 총합 출력 형식 수정 ✨✨✨
    total_duration_formatted = format_duration(total_net_duration_sum)
    print(f"🕒 모든 'Run' 사이클의 실제 가동 시간 총합: {total_duration_formatted}")
    print("="*50)


# --- 결과 시각화 ---
plt.figure(figsize=(20, 8))
ax = plt.gca()

# 상태(Run/Rest)가 바뀔 때마다 그룹을 나누어 선 색상을 다르게 지정합니다.
plot_df['state_block'] = (plot_df['predicted_label'].shift() != plot_df['predicted_label']).cumsum()
for _, group in plot_df.groupby('state_block'):
    label = group['predicted_label'].iloc[0]
    color = 'crimson' if label == 'Run' else 'black'
    ax.plot(group['일시'], group['온도'], color=color, linewidth=2.5)

# 범례를 생성합니다.
rest_line = mlines.Line2D([], [], color='black', label='휴식 (Rest)')
run_line = mlines.Line2D([], [], color='crimson', label='가동 (Run)')
ax.legend(handles=[rest_line, run_line], fontsize=12, loc='upper left')

# 그래프 제목과 라벨을 설정합니다.
ax.set_title('장비 가동 상태 예측 결과', fontsize=22, pad=20)
ax.set_xlabel('시간', fontsize=14)
ax.set_ylabel('온도 (°C)', fontsize=14)

# 그래프 우측 상단에 총 가동 시간을 텍스트로 추가합니다.
ax.text(0.98, 0.95, f'총 가동 시간: {total_duration_formatted}',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        color='navy',
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

plt.tight_layout()
plt.show()