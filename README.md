# 장비 온도 데이터를 활용한 가동 상태 예측 및 시간 분석

## 📊 프로젝트 개요

이 프로젝트는 장비의 **온도 센서 데이터**만을 활용하여 장비의 **가동(Run)** 및 **휴식(Rest)** 상태를 자동으로 예측하는 머신러닝 모델을 구축합니다. 최종적으로, 모델의 예측 결과를 바탕으로 장비의 일일 총 가동 시간을 계산하고 시각화하는 것을 목표로 합니다.

## ✨ 주요 기능

-   **데이터 전처리 자동화**: 노이즈가 포함된 시계열 데이터를 이동 평균(Moving Average)으로 스무딩하고, 모델 학습에 적합한 형태로 가공합니다.
-   **머신러닝 기반 상태 예측**: XGBoost 분류 모델을 사용하여 온도 변화 패턴을 학습하고, 장비의 가동/휴식 상태를 높은 정확도로 예측합니다.
-   **가동 시간 자동 계산**: 예측된 'Run' 상태 구간을 분석하여 각 사이클의 순수 가동 시간과 총 가동 시간을 자동으로 계산합니다.
-   **결과 시각화**: 장비의 상태 변화와 총 가동 시간을 직관적으로 파악할 수 있는 그래프를 생성합니다.

## ⚙️ 프로젝트 파이프라인

이 프로젝트는 총 5개의 단계로 구성된 파이프라인을 따릅니다.

### **1단계: 데이터 스무딩 (Data Smoothing)**

-   **실행 파일**: `0_Moving Average.py`
-   **설명**: 원본 시계열 데이터에 포함된 노이즈를 줄이고 데이터의 전반적인 경향성을 명확히 하기 위해 **이동 평균(Moving Average)** 기법을 적용합니다. 이 단계는 모델이 온도 변화 패턴을 더 쉽게 학습하도록 돕습니다.
-   **프로세스**:
    -   지정된 폴더(`CT_2025-03to06`, `Test Data`)의 모든 CSV 파일을 읽습니다.
    -   온도 데이터에 이동 평균을 적용하고 일정 비율로 샘플링하여 데이터 양을 최적화합니다.
    -   처리된 결과는 `_MA` 접미사가 붙은 폴더에 새로운 CSV 파일로 저장됩니다.

<img width="1389" height="590" alt="1" src="https://github.com/user-attachments/assets/a448aa04-5176-4c1c-b8dd-991c34dd3764" />
<img width="1389" height="590" alt="2" src="https://github.com/user-attachments/assets/39d658be-7393-4608-9925-76f2cb711c91" />

---

### **2단계: 데이터 전처리 (Preprocessing)**

-   **실행 파일**: `1_Preprocess.py`
-   **설명**: 여러 파일로 나뉜 스무딩 데이터를 하나로 병합하고, 모델 학습에 사용할 최종 데이터셋 형태로 가공합니다.
-   **프로세스**:
    -   학습용과 테스트용 데이터를 각각 병합합니다.
    -   **'일시'**와 **'온도'** 열만 추출하고, 중복된 시간대의 데이터는 온도의 평균값으로 처리하여 정제합니다.
    -   최종적으로 `processed_temperature_only_TRAIN.csv` (학습용)와 `processed_temperature_only_TEST.csv` (예측용) 파일을 생성합니다.

<img width="556" height="435" alt="1_peak_temp" src="https://github.com/user-attachments/assets/7175c50d-9312-4162-9da6-f17bec453688" />

---

### **3단계: 학습 데이터 생성 (Labeling)**

-   **실행 파일**: `2_Create_training_data.py`
-   **설명**: 전처리된 온도 데이터에 **'Run'** 또는 **'Rest'** 라벨을 부여하여 모델이 정답을 학습할 수 있는 훈련용 데이터셋을 생성합니다. 이 프로젝트에서는 Label Studio와 같은 외부 툴을 통해 생성된 `labeled_data.json` 파일을 사용했습니다.
-   **프로세스**:
    -   `labeled_data.json` 파일에 기록된 시간 정보를 바탕으로, `processed_temperature_only_TRAIN.csv` 파일의 각 행에 'Run' 또는 'Rest' 라벨을 할당합니다.
    -   라벨링이 완료된 최종 학습 데이터셋 `training_data.csv`를 생성합니다.

---

### **4단계: 모델 학습 및 성능 검증**

-   **실행 파일**: `3_BenchMark.py`, `4_Train_and_save_model.py`
-   **설명**: `training_data.csv`를 이용해 **XGBoost 분류 모델**을 학습하고 성능을 평가합니다. 이후, 새로운 데이터 예측에 사용할 수 있도록 최종 모델을 파일로 저장합니다.
-   **프로세스**:
    1.  **성능 검증 (`3_BenchMark.py`)**: 학습 데이터를 훈련/테스트 세트로 분리하여 모델을 학습시킨 후, 예측 정확도를 평가합니다. 예측 결과와 실제 정답을 시각적으로 비교하여 모델의 성능을 직관적으로 확인합니다.
    2.  **최종 모델 학습 및 저장 (`4_Train_and_save_model.py`)**: 검증이 완료되면 전체 학습 데이터를 사용하여 모델을 최종적으로 학습시킵니다. 학습된 모델은 `xgboost_model.joblib`, 라벨 정보는 `label_encoder.joblib` 파일로 저장하여 재사용이 가능하도록 만듭니다.
 
<img width="1770" height="495" alt="3" src="https://github.com/user-attachments/assets/59c8be93-827d-4a7c-90ed-861b3831c14a" />
<img width="1920" height="975" alt="2_Figure_1_MA" src="https://github.com/user-attachments/assets/78d0cbbc-9ae8-4e1b-86d9-268ed69d588f" />

---

### **5단계: 새로운 데이터 예측 및 분석**

-   **실행 파일**: `5_Load_and_predict.py`
-   **설명**: 4단계에서 저장된 모델을 불러와, 라벨이 없는 새로운 데이터의 가동 상태를 예측하고 'Run' 사이클의 총 가동 시간을 계산하여 최종 결과를 도출합니다.
-   **프로세스**:
    -   `xgboost_model.joblib` 파일을 로드하여 새로운 온도 데이터(`processed_temperature_only_TEST.csv`)의 상태를 예측합니다.
    -   'Run'으로 예측된 모든 구간을 식별하고, 각 구간의 시작-종료 시간과 순수 가동 시간을 계산합니다.
    -   모든 'Run' 사이클의 가동 시간을 합산하여 **총 가동 시간**을 구합니다.
    -   최종 예측 결과를 상태에 따라 다른 색상(Run: Crimson, Rest: Black)으로 표현하고, 총 가동 시간을 그래프에 텍스트로 추가하여 시각화합니다.

<img width="2000" height="800" alt="4_장비 가동 상태 예측 최종 결과" src="https://github.com/user-attachments/assets/5b1f134e-a050-4f81-8b54-eb9f0982b3da" />

## 🚀 실행 방법

### **1. 요구사항 설치**

```bash
pip install pandas matplotlib scikit-learn xgboost joblib openpyxl
