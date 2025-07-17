

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 모델 라이브러리 임포트
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 데이터 로딩 및 전처리 ---
try:
    df = pd.read_csv('training_data.csv', parse_dates=['일시'])
except FileNotFoundError:
    print("❌ 'training_data.csv' 파일을 찾을 수 없습니다. 이전 단계의 스크립트를 먼저 실행해주세요.")
    exit()

# 라벨 인코딩 ('Rest' -> 0, 'Run' -> 1)
# LabelEncoder를 사용하여 클래스 이름을 기억하게 함
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = df[['일시', '온도']] # 시각화를 위해 '일시'도 포함
y = df['label_encoded']

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# --- 2. 모델 학습 및 평가 (XGBoost만 선택) ---
print("--- [ XGBoost ] 모델 학습 시작 ---")
best_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# 학습 시에는 '일시' 열을 제외하고 학습
best_model.fit(X_train[['온도']], y_train)

# 예측 시에도 '일시' 열 제외
predictions = best_model.predict(X_test[['온도']])
print("--- [ XGBoost ] 모델 학습 및 평가 완료 ---\n")

print("="*50)
print("✨ XGBoost 모델 평가 결과 ✨")
print("="*50)
# classification_report를 위해 라벨 이름을 다시 문자로 변환
y_test_labels = le.inverse_transform(y_test)
predictions_labels = le.inverse_transform(predictions)
print(classification_report(y_test_labels, predictions_labels))


# --- 3. 시각화를 위한 데이터 준비 ---
# X_test에 실제 라벨과 예측 라벨을 합치기
plot_df = X_test.copy()
plot_df['true_label'] = le.inverse_transform(y_test)
plot_df['predicted_label'] = le.inverse_transform(predictions)
plot_df.sort_values(by='일시', inplace=True) # 시간 순으로 정렬


# --- 4. 최종 결과 시각화 ---
print("📊 테스트 결과 시각화 중...")

# 2개의 그래프를 나란히 그리기
fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
fig.suptitle('XGBoost 모델 테스트 결과 시각화', fontsize=20)

# 색상 팔레트 정의
palette = {'Rest': 'royalblue', 'Run': 'crimson'}

# 첫 번째 그래프: 모델의 예측 결과
sns.scatterplot(data=plot_df, x='일시', y='온도', hue='predicted_label', palette=palette, s=15, ax=axes[0])
axes[0].set_title('모델의 예측 결과 (Model Predictions)', fontsize=15)
axes[0].set_ylabel('온도')
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(title='예측된 라벨')

# 두 번째 그래프: 실제 정답
sns.scatterplot(data=plot_df, x='일시', y='온도', hue='true_label', palette=palette, s=15, ax=axes[1])
axes[1].set_title('실제 정답 (Ground Truth)', fontsize=15)
axes[1].set_xlabel('시간')
axes[1].set_ylabel('온도')
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(title='실제 라벨')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
