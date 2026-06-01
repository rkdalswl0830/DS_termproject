# Mental Health in Tech Survey 기반 Clustering 분석

# 1. 프로젝트 목표
Mental Health in Tech Survey 데이터셋을 활용하여 업무환경에 따른 서로 다른 직원 그룹의 존재 여부 확인 및 이에 따른 정신건강 치료 여부 분석을 목표로 한다. 

특히

- 정신 건강 지원 환경
- 조직 차원의 지원 체계
- 업무 간섭

등의 feature를 기반으로 clustering을 수행하였다. 

또한 clustering 결과를 treatment 경험 여부와 연결하여, 업무환경과 정신건강 치료 여부 간의 패턴 및 연관성을 분석하고자 하였다. 
<br><br>
# 2. Clustering Feature Selection
초기 전체 데이터셋을 활용한 clustering에서는 70개 이상의 feature를 모두 사용하였다.

하지만

- PCA explained variance 감소
- cluster 해석 어려움

의 문제가 발생하였다.

따라서 업무환경 및 조직문화와 직접적으로 관련된 핵심 feature만 선택하는 방향으로 개선하였다.

## 최종 선택 Feature
| Feature            | 의미                    |
| ------------------ | --------------------- |
| Age                | 직원 나이                 |
| family_history_Yes | 가족 정신 건강 이력           |
| work_interfere_-   | 정신 건강 문제가 업무에 미치는 영향  |
| benefits_Yes       | 정신 건강 관련 복지 제공 여부     |
| care_options_Yes   | 정신 건강 지원 선택지 제공 여부    |
| anonymity_Yes      | 익명성 보장 여부             |
| leave_-            | 정신 건강 관련 휴가 사용 용이성    |
| supervisor_Yes     | 상사와 정신 건강 문제 논의 가능 여부 |
| coworkers_Yes      | 동료와 정신 건강 문제 논의 가능 여부 |

<br>

# 3. PCA
## 목적
고차원 데이터를 2차원으로 축소하여

- 데이터 시각화
- cluster 분포 확인

을 위해 사용하였다.

## PCA 결과
```text
Exaplanied Variance = 0.3024320475535962
```

## 해석
PCA의 총 설명력은 약 30%로 나타났다.
이는

- categorical feature
- one-hot encoding
- 사회 데이터 특성

때문에 완전히 높은 설명력을 가지기는 어려웠다.

다만 feature selection 이후 explained variance가 0.148 -> 0.302 로 증가하여 clustering 품질이 개선되었음을 확인할 수 있었다.
<br><br>
# 4. Elbow Method
## 목적
KMeans의 적절한 cluster 개수(k)를 선택하기 위해 inertia 감소 추이를 확인하였다.

## Inertia란?
각 데이터가 자기 cluster 중심으로부터 얼마나 떨어져 있는가를 의미한다.
해당 값이 낮을수록

- cluster 내부 응집도
- clustering 품질

이 좋아진다. 

## Elbow Method 결과 해석
Elbow graph에서 k=2 이후 inertia 감소폭이 완만해지는 경향을 확인할 수 있다. 
따라서 n_clusters=2를 최종 선택하였다.
<br><br>
# 5. KMeans Clustering
## 사용 모델
```python
KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)
```

## Clustering 결과
```text
CLUSTER COUNTS

Cluster 1 : 784
Cluster 0 : 475
```

## Silhouette Score

```text
0.3810129701981751
```

## Silhouette score는 cluster간 분리도와 응집도를 동시에 평가하는 지표이다.
현재 결과는 0.38 수준으로 다소 낮은 편이지만, 노이즈가 많고 복잡한 사회 데이터 기반 clustering에서는 비교적 양호한 수준으로 볼 수 있다. 
즉, 이를 통해 업무화녕 특성 기반의 유사한 직원 그룹이 어느 정도 형성되었음을 확인할 수 있었다. 
<br><br>
# 6. Cluster Interpretation
## Cluster 0 - 정신건강 지원 환경 우수 그룹
### 특징

- benefits 높음
- care_options 매우 높음
- anonymity 매우 높음
- supervisor support 높음
- coworkers support 높음
- treatment 비율 높음

### 해석
이 그룹은 조직 차원의 정신 건강 지원 제도가 비교적 잘 구축되어 있으며, 상사 및 동료와의 지원 환경도 양호한 특성을 보였다.

특히

* anonymity
* care_options
* benefits

값이 높게 나타나, 정신 건강과 관련된 지원 체계 및 심리적 안전성이 상대적으로 높은 업무환경 그룹으로 해석할 수 있다.

또한 treatment 경험 비율 역시 상대적으로 높게 나타나, 근무 환경과 치료 경험 사이의 연관 가능성을 확인할 수 있었다.

## Cluster 1 - 정신건강 지원 환경 부족 그룹
### 특징

* work_interfere_Sometimes 높음
* benefits 매우 낮음
* care_options 매우 낮음
* anonymity 매우 낮음
* supervisor support 낮음
* coworkers support 낮음
* treatment 비율 낮음

### 해석
이 그룹은 정신 건강 문제가 업무에 일정 수준 영향을 주고 있음에도 불구하고, 조직 차원의 지원 환경은 부족한 특성을 보였다.

특히

* benefits
* care_options
* anonymity

와 같은 정신 건강 지원 관련 feature가 전반적으로 낮게 나타났으며, 상사 및 동료와의 지원 환경도 역시 상대적으로 낮은 경향을 보였다.

또한 treatment 경험 비율 역시 상대적으로 낮게 나타나, 근무 환경과 치료 경험 사이의 연관 가능성을 확인할 수 있었다. 
<br><br>
# 7. Treatement Radio Analysis
Cluster 별 treatment 비율을 비교한 결과
| Cluster   | Treatment Yes 비율 |
| --------- | ---------------- |
| Cluster 0 | 약 68%            |
| Cluster 1 | 약 40%            |

차이가 나타났다.

## 해석
근무 환경의 정신 건강 지원이 활발한 cluster에서 상대적으로 높은 treatment 경험 비율이 관찰되었다.
<br><br>
# 8. 프로젝트 결론
본 clustering 분석에서는 Mental Health in Tech Survey 데이터셋을 기반으로,
업무환경 및 조직문화 특성에 따라 서로 다른 직원 그룹이 존재하는지 분석하였다.

분석 결과:

* 정신 건강 지원 체계
* 심리적 안전성
* 조직 차원의 지원 환경

등의 feature에서 cluster 간 차이가 나타났다.

특히:

```text
benefits
care_options
anonymity
supervisor support
```

와 같은 feature들이 cluster 형성에 중요한 역할을 하였다.

또한 cluster별 treatment 경험 비율 역시 차이를 보이며,
업무환경 특성과 정신 건강 관련 행동 사이의 연관 가능성을 확인할 수 있었다.

다만, 해당 결과는 근무환경 - 정신건강 치료 간의 인과관계를 의미하지 않으며, 근무환경 특성과 치료 경험 사이의 패턴 및 연관성을 보여주는 탐구적인 분석의 결과이다. 
<br><br>
# 9. 한계점 및 개선 방안
## 한계점
* categorical feature 비중 높음
* survey 데이터 특성상 subjective bias 존재 가능
* treatment 여부에 영향을 주는 외부 요인 반영 어려움
* PCA explained variance 제한적

## 향후 개선 방향
* 추가 feature engineering
* 다른 clustering 알고리즘(DBSCAN, Hierarchical Clustering) 비교
* feature importance 기반 재선정
* 문화권별 분석 세분화
* longitudinal data 기반 분석
<br><br>
# 10. 사용 기술
| 기술               | 목적              |
| ---------------- | --------------- |
| Pandas           | 데이터 처리          |
| NumPy            | 수치 연산           |
| Matplotlib       | 시각화             |
| StandardScaler   | Feature Scaling |
| PCA              | 차원 축소           |
| KMeans           | Clustering      |
| Silhouette Score | Cluster 품질 평가   |
<br><br>
# 11. 최종 정리
본 clustering 분석은 단순히 treatment 여부를 예측하는 것이 아니라, 업무환경 및 조직문화 특성에 따라 서로 다른 직원 경험 패턴이 존재하는지를 탐색적으로 분석하는 데 목적이 있었다.

Clustering 결과로 조직의 지원 환경에 따라 서로 다른 직원 그룹이 형성되는 경향을 확인할 수 있었으며, 이는 근무 환경이 직원의 정신 건강 치료 여부에 중요한 요소가 될 가능성을 보여준다.
