import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 설정 및 하이퍼파라미터 그리드 ---
# 파일 경로
TRAIN_CSV_PATH = 'labeled_train_data.csv'
TEST_CSV_PATH = 'labeled_test_data.csv'
RESULTS_DIR = 'tuning_results' # 결과 저장 폴더

# 하이퍼파라미터 탐색 범위
WINDOW_SIZES = [100, 150, 200, 250, 300]
HIDDEN_SIZES = [32, 64, 128, 256]
NUM_LAYERS_LIST = [1, 2, 3, 4, 5]

# 고정 파라미터
INPUT_SIZE = 1
NUM_CLASSES = 2
LEARNING_RATE = 0.0005
NUM_EPOCHS = 200 # Early Stopping을 사용하므로 충분히 크게 설정
BATCH_SIZE = 64
PATIENCE = 10

# 결과 저장 폴더 생성
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- 2. 필요한 클래스 및 함수 정의 ---
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

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X, y, window_size):
    dataX, dataY = [], []
    for i in range(len(X) - window_size):
        dataX.append(X[i:(i + window_size), 0])
        dataY.append(y[i + window_size])
    return np.array(dataX), np.array(dataY)


# --- 3. 메인 루프 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 모든 결과를 저장할 리스트
all_results = []
trial_count = 0
total_trials = len(WINDOW_SIZES) * len(HIDDEN_SIZES) * len(NUM_LAYERS_LIST)

# 데이터 로딩
df_train_full = pd.read_csv(TRAIN_CSV_PATH)
df_test_full = pd.read_csv(TEST_CSV_PATH)

for ws in WINDOW_SIZES:
    for hs in HIDDEN_SIZES:
        for nl in NUM_LAYERS_LIST:
            trial_count += 1
            print(f"\n{'='*50}")
            print(f"TRIAL {trial_count}/{total_trials}: WINDOW_SIZE={ws}, HIDDEN_SIZE={hs}, NUM_LAYERS={nl}")
            print(f"{'='*50}")

            # --- 데이터 준비 ---
            train_features = df_train_full[['정규화_온도']].values
            train_labels = df_train_full['label'].values
            test_features = df_test_full[['정규화_온도']].values
            test_labels = df_test_full['label'].values

            scaler = MinMaxScaler(feature_range=(0, 1))
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)

            X, y = create_sequences(train_features_scaled, train_labels, ws)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_test, y_test = create_sequences(test_features_scaled, test_labels, ws)
            
            train_dataset = TimeSeriesDataset(X_train, y_train)
            val_dataset = TimeSeriesDataset(X_val, y_val)
            test_dataset = TimeSeriesDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # --- 모델 학습 ---
            model = LSTMClassifier(INPUT_SIZE, hs, nl, NUM_CLASSES).to(device)
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(NUM_EPOCHS):
                model.train()
                for sequences, labels in train_loader:
                    sequences, labels = sequences.to(device), labels.to(device)
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for sequences, labels in val_loader:
                        sequences, labels = sequences.to(device), labels.to(device)
                        outputs = model(sequences)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                if (epoch + 1) % 10 == 0:
                     print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Val Loss: {avg_val_loss:.4f}')

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'best_temp_model.pt'))
                else:
                    patience_counter += 1
                
                if patience_counter >= PATIENCE:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            # --- 모델 평가 ---
            print("Evaluating best model...")
            model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'best_temp_model.pt')))
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for sequences, labels in test_loader:
                    sequences = sequences.to(device)
                    outputs = model(sequences)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            report = classification_report(all_labels, all_preds, target_names=['Rest (0)', 'Run (1)'], output_dict=True, zero_division=0)
            
            # --- 결과 저장 ---
            result_summary = {
                'window_size': ws,
                'hidden_size': hs,
                'num_layers': nl,
                'accuracy': report['accuracy'],
                'run_precision': report['Run (1)']['precision'],
                'run_recall': report['Run (1)']['recall'],
                'run_f1-score': report['Run (1)']['f1-score']
            }
            all_results.append(result_summary)
            print(f"Trial Result: Accuracy={result_summary['accuracy']:.4f}, Run F1-Score={result_summary['run_f1-score']:.4f}")


# --- 4. 최종 결과 분석 및 시각화 ---
print(f"\n{'='*50}")
print("HYPERPARAMETER TUNING COMPLETE")
print(f"{'='*50}")

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by='run_f1-score', ascending=False)

# 최종 출력 결과 (상위 15개)
print("\n--- Top 15 Hyperparameter Combinations (Sorted by Run F1-Score) ---")
print(results_df.head(15).to_string())

# 최종 결과 파일로 저장
results_df.to_csv(os.path.join(RESULTS_DIR, 'hyperparameter_tuning_results.csv'), index=False)

# 최종 시각화 자료 (상위 15개 F1-Score)
print("\n--- Generating Visualization ---")
top_15_results = results_df.head(15)
top_15_results['params'] = top_15_results.apply(lambda row: f"W:{row['window_size']}, H:{row['hidden_size']}, L:{row['num_layers']}", axis=1)

plt.figure(figsize=(12, 8))
sns.barplot(x='run_f1-score', y='params', data=top_15_results, palette='viridis')
plt.title('Top 15 Model Performance by Hyperparameters (Run F1-Score)', fontsize=16)
plt.xlabel('F1-Score for "Run" Class', fontsize=12)
plt.ylabel('Hyperparameter Combination (Window, Hidden, Layers)', fontsize=12)
plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'top_15_performance_chart.jpg'), dpi=300)
plt.show()

print(f"\nAll results saved in '{RESULTS_DIR}' folder.")