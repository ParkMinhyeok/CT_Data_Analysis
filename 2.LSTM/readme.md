# LSTM 기반 장비 가동 상태 예측 모델 고도화
완성형 모델이라고 보기는 어렵고, 아직 더 많은 데이터와 학습이 필요합니다.

## 📊 프로젝트 개요

이전 연구([XGBoost 기반 모델](https://github.com/ParkMinhyeok/CT_Data_Analysis/tree/main))에서는 XGBoost를 활용하여 장비의 가동 상태를 성공적으로 분류했습니다. 본 프로젝트는 여기서 한 걸음 더 나아가, 데이터의 **시간적 연속성(Temporal Dependency)을 학습**할 수 있는 **LSTM(Long Short-Term Memory)** 모델을 도입하여 예측 성능을 고도화하는 것을 목표로 합니다. (전처리는 같은 부분은 이전 연구와 많은 부분을 공유)

XGBoost가 개별 데이터 포인트의 특징에 집중하는 반면, LSTM은 시계열 데이터의 순차적인 **패턴과 맥락**을 학습합니다. 이를 통해 온도 변화의 '흐름'을 파악하여 더 정확하고 안정적인 가동 상태 분류가 가능할 것이라는 가설을 검증하고, 성공적으로 성능을 향상시켰습니다.

## ✨ 주요 기능

-   **시계열 데이터 특화 모델링**: 데이터의 순차적 특성을 학습하기 위해 양방향(Bidirectional) LSTM 신경망을 적용했습니다.
-   **체계적인 전처리 파이프라인**: 데이터 정렬, 정규화, 라벨링 등 모델 학습에 최적화된 형태로 데이터를 가공하는 파이프라인을 구축했습니다.
-   **하이퍼파라미터 자동 최적화**: `Window Size`, `Hidden Size`, `Layer 수` 등 모델의 성능을 좌우하는 최적의 하이퍼파라미터 조합을 자동으로 탐색하고 시각화합니다.
-   **모델 성능 심층 분석**: Classification Report, 혼동 행렬(Confusion Matrix)을 통해 모델의 성능을 다각도로 평가하고, 이전 XGBoost 모델과 결과를 비교 분석합니다.

## ⚙️ 프로젝트 파이프라인

본 프로젝트는 총 6개의 단계로 구성된 파이프라인을 따릅니다.

### **1단계: 데이터 전처리 (Data Preprocessing)**

-   **실행 파일**: `1-1_Normalization.py`, `1-2_Preprocess_sort.py`
-   **설명**: 여러 파일에 나뉘어 있는 원본 데이터를 하나로 병합하고, 시간순으로 정렬한 뒤, 신경망 학습에 적합하도록 정규화(Normalization)를 진행합니다.
-   **프로세스**:
    1.  `1-2_Preprocess_sort.py`: 여러 CSV 파일을 병합하고 '일시'를 기준으로 오름차순 정렬하여 `1-2_sorted_temperature_data.csv`를 생성합니다.
    2.  `1-1_Normalization.py`: 정렬된 데이터를 학습용과 테스트용으로 분리하고, 각 데이터의 온도 값을 0과 1 사이로 정규화하여 최종 전처리 데이터(`..._normalized.csv`)를 생성합니다.

---

### **2단계: 학습 데이터 생성 (Create Training Data)**

-   **실행 파일**: `2_Create_training_data.py`
-   **설명**: 1단계에서 전처리된 데이터에 **'Run'** 또는 **'Rest'** 라벨을 부여하여 모델이 정답을 학습할 수 있는 훈련용 데이터셋을 생성합니다.
-   **프로세스**:
    -   `2_labeled_data.json` 파일의 시간 정보를 바탕으로, `1-1_..._normalized.csv` 파일의 각 행에 라벨을 할당합니다.
    -   라벨링이 완료된 최종 학습 데이터셋 `2_labeled_train_data.csv`와 `2_labeled_test_data.csv`를 생성합니다.

---

### **3단계: 최적 Window Size 탐색 (Explore Optimal Window Size)**

-   **실행 파일**: `2_window_size_print.py`
-   **설명**: LSTM 모델의 핵심 파라미터인 `Window Size`를 결정하기 위해, 라벨링된 'Run' 구간의 길이 통계를 분석합니다.
-   **프로세스**:
    -   `2_labeled_data.json` 파일의 'Run' 라벨 구간들의 길이를 분석하여 기술 통계량을 출력하고, 이를 바탕으로 `Window Size`를 결정합니다.

---

### **4단계: 하이퍼파라미터 튜닝 (Hyperparameter Tuning)**

-   **실행 파일**: `5_Hyperparameter_Tuning.py`
-   **설명**: 최적의 LSTM 모델을 찾기 위해 그리드 탐색(Grid Search)을 통해 다양한 하이퍼파라미터 조합의 성능을 체계적으로 비교 및 평가합니다.
-   **튜닝 결과 시각화**:
     - (추후 업로드)

---

### **5단계: 최종 모델 학습 및 저장 (Final Model Training & Saving)**

-   **실행 파일**: `3_LSTM_TRAIN.py`
-   **설명**: 튜닝 단계에서 찾은 최적의 하이퍼파라미터를 사용하여 최종 LSTM 모델을 학습시키고, 재사용을 위해 파일로 저장합니다.
-   **프로세스**:
    -   `2_labeled_train_data.csv`를 사용하여 최적의 파라미터로 양방향 LSTM 모델을 학습합니다.
    -   과적합 방지를 위해 `Dropout`과 `Early Stopping`을, 클래스 불균형 완화를 위해 `Class Weight`를 적용합니다.
    -   학습이 완료된 모델(`best_model.pt`)과 스케일러(`scaler.pkl`)를 저장합니다.

---

### **6단계: 모델 성능 평가 및 시각화 (Model Performance Evaluation & Visualization)**

-   **실행 파일**: `4_LSTM_TEST.py`
-   **설명**: 학습된 `best_model.pt`를 불러와, 별도의 테스트 데이터(`2_labeled_test_data.csv`)에 대한 예측 성능을 최종 평가하고 결과를 시각화합니다.
-   **성능 비교 (XGBoost vs LSTM)**:

| 모델 | 클래스 | 정밀도(Precision) | 재현율(Recall) | F1-점수(F1-Score) |
| :--- | :--- | :---: | :---: | :---: |
| **LSTM (최종)** | **Run (1)** | **0.80** | **0.87** | **0.83** |
| | Rest (0) | 0.94 | 0.91 | 0.93 |
| XGBoost (이전) | Run (1) | 0.94 | 0.81 | 0.71 |
| | Rest (0) | 0.88 | 0.98 | 0.92 |

-   **혼동 행렬 및 예측 결과 시각화**:
![confusion_matrix](https://github.com/user-attachments/assets/0e7b844f-4578-411e-a0de-6dacd55071f0)
![prediction_visualization](https://github.com/user-attachments/assets/db0e7902-2ff8-4b9a-b070-4e23aa9ba689)

## 🚀 실행 방법

### **1. 요구사항 설치**
```bash
pip install pandas torch scikit-learn joblib matplotlib seaborn
```

### **2. 폴더 구조**
```bash
.
├── data/
│   ├── raw/                  # 📁 원본 데이터 (CSV 파일들)
│   ├── processed/            # 📂 (자동 생성) 전처리된 데이터
│   │   ├── 1-2_sorted_temperature_data.csv
│   │   ├── 1-1_processed_temperature_only_TRAIN_normalized.csv
│   │   ├── 1-1_processed_temperature_only_TEST_normalized.csv
│   │   ├── 2_labeled_train_data.csv
│   │   └── 2_labeled_test_data.csv
│   └── external/             # 📂 외부 데이터 (라벨링 파일)
│       └── 2_labeled_data.json
│
├── outputs/
│   ├── figures/              # 📈 (자동 생성) 시각화 결과 이미지
│   └── models/               # 📦 (자동 생성) 최종 모델 파일
│
├── src/                      # 📜 모든 파이썬 소스 코드
    ├── 1-1_Normalization.py
    ├── 1-2_Preprocess_sort.py
    ├── 2_Create_training_data.py
    ├── 2_window_size_print.py
    ├── 3_LSTM_TRAIN.py
    ├── 4_LSTM_TEST.py
    └── 5_Hyperparameter_Tuning.py
```

### **3. 실행 순서**
```bash
# 1. 데이터 정렬
python src/1-2_Preprocess_sort.py

# 2. 데이터 정규화
python src/1-1_Normalization.py

# 3. 학습 데이터 생성
python src/2_Create_training_data.py

# 4. (선택) 최적 Window Size 탐색
python src/2_window_size_print.py

# 5. 하이퍼파라미터 튜닝 (시간이 오래 소요될 수 있음)
python src/5_Hyperparameter_Tuning.py

# 6. 최종 모델 학습
python src/3_LSTM_TRAIN.py

# 7. 모델 성능 평가
python src/4_LSTM_TEST.py
```
