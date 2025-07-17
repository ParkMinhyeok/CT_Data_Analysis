import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib # 모델 저장을 위해 joblib 라이브러리 임포트

# --- 1. 데이터 로딩 및 전처리 ---
try:
    df = pd.read_csv('training_data.csv')
except FileNotFoundError:
    print("❌ 'training_data.csv' 파일을 찾을 수 없습니다.")
    exit()

print("🔄 라벨 인코딩 및 데이터 준비 중...")
# LabelEncoder를 사용하여 'Rest'/'Run'을 0/1로 변환
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Feature(X)와 Label(y) 정의
X_train = df[['온도']]
y_train = df['label_encoded']


# --- 2. 모델 학습 ---
print("--- [ XGBoost ] 모델 학습 시작 ---")
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
print("--- [ XGBoost ] 모델 학습 완료 ---")


# --- 3. 학습된 모델과 라벨 인코더 저장 ---
MODEL_FILENAME = 'xgboost_model.joblib'
ENCODER_FILENAME = 'label_encoder.joblib'

print(f"\n💾 모델을 '{MODEL_FILENAME}' 파일로 저장 중...")
joblib.dump(model, MODEL_FILENAME)

print(f"💾 라벨 인코더를 '{ENCODER_FILENAME}' 파일로 저장 중...")
joblib.dump(le, ENCODER_FILENAME)

print("\n✨ 모델과 인코더 저장이 완료되었습니다.")
