
---

```markdown
# DenseNet161 AI모델 재현 및 타 AI모델 성능 비교 과제

## 1. 코드 설명
본 과제에서는 2채널 손바닥 표면 근전도(palm-sEMG) 데이터를 활용하여 사용자 식별(Identification)을 수행하는 AI 모델 파이프라인을 구현하고, Baseline 2D CNN, ResNet18, DenseNet161 모델 간 성능 비교 실험을 진행함.

- **사용한 데이터 및 전처리 방법**: `palm-sEMG-doorknob-filtered` 데이터셋(5개 클래스 A~E, 사용자/시행별 50개 파일, 총 250개 파일)을 사용함. 60Hz Notch Filter 및 20~495Hz Bandpass Filter로 노이즈를 제거한 후, 300ms 윈도우 크기(50% Overlap)의 슬라이딩 윈도우를 적용하였으며 Morlet Wavelet 기반 CWT 변환을 통해 (3, 32, 300) 크기의 시간-주파수 스펙트로그램 이미지를 생성함.
- **사용한 AI/ML 모델**: Baseline 2D CNN, ResNet18, DenseNet161 총 3개 모델을 구현하여 사용함.
- **학습 및 테스트 방법**: 데이터 누수 방지를 위해 Trial 단위 Stratified 5-Fold Cross Validation을 수행하였으며, 무작위성을 배제하고 결과의 통계적 신뢰성을 확보하기 위해 전체 과정을 3회 반복 실행하여 평균 수치를 산출함. (Batch size=16, Epochs=30, Adam optimizer, lr=0.001)

### 실행 방법
```bash
python main.py

```

### 코드 설명

각 코드 파일의 역할은 다음과 같습니다.

| 파일 | 설명 |
| --- | --- |
| `main.py` | 전체 실험 실행 (데이터 로딩, 3개 모델 5-Fold CV 진행 및 요약 출력) |
| `preprocessor.py` | 60Hz 노치/BPF 필터링, 슬라이딩 윈도우 및 Morlet CWT 변환 처리 |
| `dataset.py` | CSV 헤더 처리, 수치 데이터 파싱 및 PyTorch Dataset 구축 |
| `models.py` | Baseline 2D CNN, ResNet18, DenseNet161 모델 구조 정의 |
| `train.py` | Stratified 5-Fold Cross Validation 학습 및 평가 루프 |
| `utils.py` | 혼동 행렬(Confusion Matrix) 이미지 생성 및 저장 모듈 |

## 2. 모델 성능 비교

3회 반복 실행(Stratified 5-Fold CV)을 통해 산출된 모델별 평균 성능 수치입니다.

| Model | Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- | --- |
| Baseline 2D CNN | 49.95% | 51.81% | 49.95% | 49.29% |
| ResNet18 | 78.70% | 80.52% | 78.70% | 78.59% |
| **DenseNet161** | **79.48%** | **81.24%** | **79.48%** | **79.25%** |

### 성능 분석

DenseNet161 모델이 가장 높은 평균 F1-score(79.25%)를 나타냈으며, 대부분의 클래스에서 가장 안정적인 분류 성능을 보임.
Baseline 2D CNN(49.29%) 대비 깊은 인공신경망 적용 시 약 +29.96%p의 성능 향상이 확인되었으며, 이는 DenseNet의 Feature Reuse(특징 재사용) 구조가 sEMG CWT 스펙트로그램 상의 세밀한 주파수-시간 패턴을 효과적으로 학습했기 때문이다.

## 3. Confusion Matrix 분석

### Model 1: Baseline 2D CNN

#### 분석

* **가장 잘 분류된 클래스**: 클래스 A
* **가장 많이 오분류된 클래스**: 클래스 C, D, E
* **주요 오분류 유형**: 클래스 D $\rightarrow$ 클래스 E, 클래스 C $\rightarrow$ 클래스 D
* **오분류가 발생한 이유에 대한 분석**: 얕은 2개 레이어 수준의 Conv 구조로는 다채널 CWT 스펙트로그램 내 미세한 주파수 변화 및 채널 간 복합 특성을 다차원적으로 표현하기에 모델 용량이 부족하여 약 50%의 무작위 예측 성능에 머무름.

### Model 2: ResNet18

#### 분석

* **가장 잘 분류된 클래스**: 클래스 A, B, E
* **가장 많이 오분류된 클래스**: 클래스 C
* **주요 오분류 유형**: 클래스 C $\rightarrow$ 클래스 D
* **오분류가 발생한 이유에 대한 분석**: Residual Connection을 통해 특성 추출 성능이 크게 개선되었으나, 문고리를 잡는 초기 악력(Grasping) 동작 시 C와 D 사용자 간 sEMG 스펙트럼 유사도가 높아 순간적인 오분류가 발생함.

### Model 3: DenseNet161

#### 분석

* **가장 잘 분류된 클래스**: 클래스 A, B, E
* **가장 많이 오분류된 클래스**: 클래스 C
* **주요 오분류 유형**: 클래스 C $\rightarrow$ 클래스 D
* **오분류가 발생한 이유에 대한 분석**: 모든 레이어의 Feature Map을 누적 연결하는 구조 덕분에 저차원 신호 형태와 고차원 주파수 패턴을 동시에 보존하여 3개 모델 중 가장 높은 정확도와 균형 잡힌 분류를 달성함.

## 4. 최종 결과

* **가장 성능이 좋은 모델**: DenseNet161 (Accuracy: 79.48%, F1-score: 79.25%)
* **가장 성능이 낮은 모델**: Baseline 2D CNN (Accuracy: 49.95%, F1-score: 49.29%)
* **주요 오분류 클래스**: 클래스 C 및 클래스 D
* **전체적인 실험 결과 및 느낀 점**:
1. 1차원 표면 근전도(sEMG) 신호를 Morlet CWT 변환하여 2차원 시간-주파수 스펙트로그램 이미지로 변환함으로써 딥러닝 기반 이미지 분류 아키텍처를 효과적으로 적용할 수 있음을 확인.
2. 단순 Conv 구조 대비 ResNet 및 DenseNet과 같이 지름길 연결(Shortcut/Dense Connection)이 포함된 최신 아키텍처가 시계열 이미지 특징을 추출하는 데 뛰어난 성능을 발휘함을 검증.
3. 총 3회의 Stratified 5-Fold 교차 검증 반복 실험을 수행하여 모델 학습의 안정성과 통계적 신뢰성을 확보할 수 있었다.



```

```
