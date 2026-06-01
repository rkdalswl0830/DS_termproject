#  Mental Health in Tech Survey Analysis

> **Data Science Term Project — Team 10**
> 기술 산업 종사자의 정신 건강 치료 경험 예측 및 업무환경 영향 분석

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 프로젝트 개요

본 프로젝트는 **Mental Health in Tech Survey** 데이터셋을 활용하여 IT 업계 종사자의 **정신 건강 치료 경험 여부(treatment)** 를 예측하고, 업무 환경 요인이 치료 결정에 미치는 영향을 분석한다.

### 핵심 메시지
> **"정신 건강 치료는 단순한 개인 선택이 아니라, 직장 환경과 조직 지원 체계의 결과이다."**

### 데이터셋

- **출처**: [Kaggle — Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **규모**: 1,259명 응답자 × 27개 변수
- **Target**: `treatment` (Yes / No) — Binary Classification
- **특징**: 수치형 + 범주형 혼합, 결측치·이상치 다수 포함

---

##  분석 프로세스

전체 분석은 다음과 같은 End-to-End 파이프라인으로 진행되었다.

```
EDA → Data Preprocessing → Classification → Clustering → Interpretation
```

### 1️. EDA (Exploratory Data Analysis)
- 변수별 분포 및 결측치 분석
- Gender(49개 표기) 및 Country(48개국) 비정형 데이터 식별
- Age 이상치(-1726, 99999999999 등) 검출
- Target 분포 확인: Yes 50.6% / No 49.4% (Balanced)

### 2️. Data Preprocessing
- **불필요 컬럼 제거**: `Timestamp`, `comments` (87% 결측), `state` (41% 결측)
- **Age 이상치 처리**: 18~80 범위 외 값을 NaN으로 변환 후 Median(31세) 대체
- **Gender 정규화**: Male / Female / Other 3개 카테고리로 통합
- **Country 그룹화**: 48개국 → 6개 대륙 (North America, Europe, Asia 등)
- **결측치 처리**: `self_employed` → mode, `work_interfere` → mode
- **Encoding**: One-Hot Encoding (pd.get_dummies)
- **Scaling**: Min-Max Scaling (Age 변수)
- **결과**: 27개 변수 → 72개 변수 (1,259 × 72)

### 3️. Classification Modeling

7개 모델을 비교하였으며 GridSearchCV + Stratified 5-Fold Cross Validation으로 하이퍼파라미터 최적화 수행.

| 모델 | Accuracy | Precision | Recall | **F1** | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.742 | 0.765 | 0.711 | **0.737** | **0.824** |
| **Gradient Boosting** | 0.742 | 0.765 | 0.711 | **0.737** | 0.824 |
| AdaBoost | 0.746 | 0.781 | 0.695 | 0.736 | 0.823 |
| Bagging | 0.730 | 0.746 | 0.711 | 0.728 | 0.784 |
| Random Forest | 0.730 | 0.750 | 0.703 | 0.726 | 0.812 |
| Decision Tree | 0.718 | 0.757 | 0.656 | 0.703 | 0.773 |
| KNN | 0.671 | 0.759 | 0.516 | 0.614 | 0.746 |

**Top Model**: Logistic Regression & Gradient Boosting (F1 = 0.737)

### 4️. Feature Importance (Permutation Importance)

| 순위 | 변수 | 평균 중요도 |
|---:|---|---:|
| 1 | `work_interfere_Never` | 0.055 |
| 2 | `care_options_Yes` | 0.023 |
| 3 | `work_interfere_Often` | 0.015 |
| 4 | `family_history_Yes` | 0.012 |
| 5 | `family_history_No` | 0.008 |

→ **업무 영향도, 회사 지원 제도, 가족력**이 치료 경험 예측의 핵심 변수.

### 5️. Clustering Analysis

업무환경 및 조직문화 관련 12개 변수를 기반으로 K-Means 수행 (k=2 from Elbow Method).

| Cluster | 인원 | 특성 | Treatment 비율 |
|---|---:|---|---:|
| **Cluster 0** | 475명 | 정신건강 지원 환경 **우수** | **68.4%** |
| **Cluster 1** | 784명 | 정신건강 지원 환경 **부족** | **39.8%** |

- Silhouette Score: 0.381
- PCA 설명력: 30.24%

→ **회사의 정신건강 지원 환경에 따라 치료 경험 비율이 28.6%p 차이** 발생.

---

## 주요 인사이트

1. **조직 환경의 영향력**: 정신건강 지원 제도, 진료 옵션, 익명성 보장이 잘 갖춰진 환경에서 치료 경험 비율이 현저히 높음.
2. **인지의 중요성**: 제도 존재 여부보다 직원의 인지 수준이 치료 결정에 더 큰 영향을 미침 ("Don't know" 응답자의 치료율이 가장 낮음).
3. **개인 요인**: 가족력(family_history)과 업무 영향도(work_interfere)가 가장 강한 개인 변수.
4. **모델 안정성**: Logistic Regression과 Gradient Boosting이 가장 안정적인 예측 성능 제공.

---

## 폴더 구조

```
DS_termproject/
├── data/                          # 데이터셋
│   ├── survey.csv                 # 원본 데이터
│   └── cleaned_survey.csv         # 전처리 완료 데이터
├── notebooks/                     # Jupyter 노트북
│   ├── 01_eda_강민지.ipynb
|   ├── 02_preprocessing_임우진.ipynb
|   ├── 03_training_김서준.ipynb
|   ├── 04_clustering_장혁진.ipynb
|   ├──clustering_분석_장혁진.md
|   └──eda_해석.md
├── src/                           # 통합 함수 모듈
│   └── auto_ml_pipeline.py        # Open Source SW Contribution
├── outputs/                       # 시각화 결과 (PNG)
├── docs/                          # 문서
│   └── Report-team10.docx         # 최종 보고서
├── README.md                      # 본 파일
├── LICENSE                        # MIT License
├── requirements.txt               # 의존성 패키지
└── .gitignore
```

---

## 사용 방법

### 1. Repository Clone

```bash
git clone https://github.com/rkdalswl0830/DS_termproject.git
cd DS_termproject
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. Notebook 실행

Jupyter Lab 또는 VS Code에서 `notebooks/` 폴더의 노트북을 순서대로 실행:

```bash
jupyter lab
```

순서: `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_classification.ipynb` → `04_clustering.ipynb`

### 4. Open Source 함수 사용

```python
from src.auto_ml_pipeline import run_training_pipeline

artifacts = run_training_pipeline(
    csv_path="data/survey.csv",
    target_col="treatment",
    k_values=(3, 5, 10),
    primary_metric="f1",
    test_size=0.2,
    random_state=42,
)

# Top 5 best combinations
print(artifacts["holdout_df"].head(5))
```
---

## 사용 라이브러리

| 라이브러리 | 용도 |
|---|---|
| `pandas` | 데이터 처리 |
| `numpy` | 수치 연산 |
| `matplotlib`, `seaborn` | 시각화 |
| `scikit-learn` | 머신러닝 모델 및 평가 |
| `imbalanced-learn` (optional) | 클래스 균형 처리 |

---

## 한계점 및 향후 과제

1. **설문 데이터의 주관성**: 응답자의 주관적 판단이 포함될 수 있음.
2. **PCA 설명력 제한**: One-Hot Encoding된 범주형 변수가 많아 차원 축소 효과가 제한적.
3. **미국 중심 편향**: 전체 응답자의 약 60%가 미국 → 일반화 한계.
4. **인과 관계 한계**: 상관 분석이며 인과 관계를 직접 증명하지는 않음.

**향후 개선**: DBSCAN/Hierarchical Clustering 비교, 국가·문화권별 세분화 분석, SHAP 등 모델 해석 기법 추가.

---

## 팀 구성 (Team 10)

| 이름 | 역할 |
|---|---|
| 강민지 | EDA & 시각화 |
| 임우진 | Data Preprocessing |
| 김서준 | Classification Modeling |
| 장혁진 | Clustering Analysis |
| 이민서 | Open Source Function & 문서화 |

---

## 라이선스

본 프로젝트는 [MIT License](LICENSE)를 따른다.

## 참고 자료

- Dataset: [Kaggle — Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- 최종 보고서: [`docs/Report-team10.docx`](docs/Report-team10.docx)
