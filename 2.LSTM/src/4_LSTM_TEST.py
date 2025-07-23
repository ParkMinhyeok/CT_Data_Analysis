import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# --- 1. 설정 및 파라미터 ---
TEST_CSV_PATH = 'labeled_test_data.csv'
MODEL_PATH = 'best_model.pt'
SCALER_PATH = 'scaler.pkl'

# 모델 하이퍼파라미터 (학습 코드와 정확히 일치)
WINDOW_SIZE = 200
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 2
NUM_CLASSES = 2 # Run(1) vs Rest(0) 이진 분류
BATCH_SIZE = 64

# --- 2. 학습 코드와 동일한 모델 및 함수 정의 ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    def forward(self, x):
        x = x.unsqueeze(-1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(X, y, window_size):
    dataX, dataY = [], []
    for i in range(len(X) - window_size):
        dataX.append(X[i:(i + window_size), 0])
        dataY.append(y[i + window_size])
    return np.array(dataX), np.array(dataY)

# --- 3. 테스트 준비 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as e:
    print(f"🚨 오류: {e}")
    print("먼저 학습 스크립트를 실행하여 'best_model.pt'와 'scaler.pkl' 파일을 생성해야 합니다.")
    exit()

model.eval()

df_test = pd.read_csv(TEST_CSV_PATH)
df_test['일시'] = pd.to_datetime(df_test['일시'])
features = df_test[['정규화_온도']].values
labels = df_test['label'].values
scaled_features = scaler.transform(features)
X_test, y_test = create_sequences(scaled_features, labels, WINDOW_SIZE)
test_loader = DataLoader(dataset=list(zip(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())),
                         batch_size=BATCH_SIZE, shuffle=False)

# --- 4. 모델 예측 및 평가 ---
print("--- 모델 테스트 시작 ---")
all_preds = []
all_labels = []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 5. 결과 분석 ---
print("\n--- 분류 리포트 (Classification Report) ---")
target_names = ['Rest (0)', 'Run (1)']
print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

print("\n--- 혼동 행렬 (Confusion Matrix) ---")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# [수정] 혼동 행렬 그래프를 jpg 파일로 저장
plt.savefig('confusion_matrix.jpg', dpi=300, bbox_inches='tight')
print("✅ 'confusion_matrix.jpg' 파일 저장 완료")
plt.show()


# --- 6. 결과 시각화 ---
print("\n--- 결과 시각화 시작 ---")
df_plot = df_test.iloc[WINDOW_SIZE:].copy()
df_plot['predicted_label'] = all_preds

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(df_plot['일시'], df_plot['정규화_온도'], color='lightgray', alpha=0.7, label='Normalized Temperature', zorder=1)
for _, g in df_plot[df_plot['label'] == 1].groupby((df_plot['label'] != df_plot['label'].shift()).cumsum()):
    ax.axvspan(g['일시'].iloc[0], g['일시'].iloc[-1], color='blue', alpha=0.2, label='_nolegend_')
pred_run = df_plot[df_plot['predicted_label'] == 1]
ax.scatter(pred_run['일시'], pred_run['정규화_온도'], color='red', marker='o', s=10, label='Predicted Run', zorder=5)
handles, labels = ax.get_legend_handles_labels()
from matplotlib.patches import Patch
handles.append(Patch(facecolor='blue', alpha=0.2))
labels.append('True Run')
ax.legend(handles, labels, fontsize=12)
ax.set_title('Model Prediction vs True Labels', fontsize=18)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Normalized Temperature', fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.tight_layout()

# [수정] 예측 결과 시각화 그래프를 jpg 파일로 저장
plt.savefig('prediction_visualization.jpg', dpi=300, bbox_inches='tight')
print("✅ 'prediction_visualization.jpg' 파일 저장 완료")
plt.show()