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

# --- 1. 설정 및 하이퍼파라미터 ---
# 파일 경로
TRAIN_CSV_PATH = 'labeled_train_data.csv'
TEST_CSV_PATH = 'labeled_test_data.csv'
MODEL_SAVE_PATH = 'best_model.pt'
SCALER_SAVE_PATH = 'scaler.pkl'

# 모델 하이퍼파라미터 (이 값들을 조절하며 실험)
WINDOW_SIZE = 200
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 3
NUM_CLASSES = 2

# 학습 하이퍼파라미터
LEARNING_RATE = 0.0001
NUM_EPOCHS = 200 # Early Stopping을 사용하므로 충분히 크게 설정
BATCH_SIZE = 64
PATIENCE = 10

# --- 2. 데이터 준비 ---
def load_and_preprocess_data(csv_path, scaler=None, is_train=True):
    df = pd.read_csv(csv_path)
    features = df[['정규화_온도']].values
    labels = df['label'].values

    if is_train:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)
    
    return scaled_features, labels, scaler

def create_sequences(X, y, window_size):
    dataX, dataY = [], []
    for i in range(len(X) - window_size):
        dataX.append(X[i:(i + window_size), 0])
        dataY.append(y[i + window_size])
    return np.array(dataX), np.array(dataY)

# 학습/검증 데이터 준비
train_features_scaled, train_labels, scaler = load_and_preprocess_data(TRAIN_CSV_PATH, is_train=True)
X, y = create_sequences(train_features_scaled, train_labels, WINDOW_SIZE)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 테스트 데이터 준비
test_features_scaled, y_test_full, _ = load_and_preprocess_data(TEST_CSV_PATH, scaler=scaler, is_train=False)
X_test, y_test = create_sequences(test_features_scaled, y_test_full, WINDOW_SIZE)

# --- 3. PyTorch Dataset 및 DataLoader ---
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- 4. PyTorch 모델 정의 ---
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

# --- 5. 학습 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')
patience_counter = 0

print("\n--- 학습 시작 ---")
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
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        print(f'==> Model and scaler saved at epoch {epoch+1}')
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f'Early stopping at epoch {epoch+1}')
        break

# --- 6. 최종 평가 ---
print("\n--- 최종 테스트 평가 ---")
# 가장 좋았던 모델 불러오기
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- 분류 리포트 (Classification Report) ---")
target_names = ['Rest (0)', 'Run (1)']
print(classification_report(all_labels, all_preds, target_names=target_names))