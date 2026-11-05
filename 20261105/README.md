# Day 14 — Vision·멀티모달 AI 응용

> **학습 자료:** `Day14_20261105.pdf` (99페이지)  
> **수업 주제:** 데이터셋 설계 → 전이학습·Fine-tuning → 클래스 불균형 평가 → 오류 분석·Grad-CAM  
> **목표:** 이미지 분류 실험을 설계하고, 모델의 성능과 실패 원인을 근거와 함께 설명하기

## 목차

1. [전체 흐름](#1-전체-흐름)
2. [1교시 — 데이터셋 설계·분할·누수](#2-1교시--데이터셋-설계분할누수)
3. [2교시 — 사전학습 Backbone·Head](#3-2교시--사전학습-backbonehead)
4. [3교시 — Freeze·Fine-tuning](#4-3교시--freezefine-tuning)
5. [4교시 — 클래스 불균형·Loss·Sampler](#5-4교시--클래스-불균형losssampler)
6. [5교시 — 평가·Confusion Matrix·오류 분석](#6-5교시--평가confusion-matrix오류-분석)
7. [6교시 — Grad-CAM·편향·설명 한계](#7-6교시--grad-cam편향설명-한계)
8. [7교시 — 모델 비교·프로젝트 통합](#8-7교시--모델-비교프로젝트-통합)
9. [8교시 — 실습평가·검증·회고](#9-8교시--실습평가검증회고)
10. [핵심 용어 암기표](#10-핵심-용어-암기표)
11. [자가 점검](#11-자가-점검)

---

## 1. 전체 흐름

```text
이미지·라벨·출처 확인
        ↓
train / valid / test 분할 및 누수 검사
        ↓
사전학습 ResNet18 Backbone + 새 분류 Head
        ↓
Head-only → Partial → Full Fine-tuning 비교
        ↓
클래스 불균형 대응 (Class Weight / Sampler)
        ↓
Accuracy · Macro F1 · 클래스별 Recall · Confusion Matrix
        ↓
오분류 이미지와 Grad-CAM 분석
        ↓
실험 설정·결과·오류 근거 저장 및 재실행
```

수업의 주요 합성 데이터 예시는 **circle / square / triangle**의 3클래스 분류다. 슬라이드의 합성 실험 수치는 실제 자연 이미지나 ResNet18의 일반적인 성능을 뜻하지 않는다.

## 2. 1교시 — 데이터셋 설계·분할·누수

**범위:** PDF 17~26페이지

### 용어

| 용어 | 의미 |
|---|---|
| Dataset | 학습·평가에 쓰이는 데이터 모음 |
| Sample | 개별 이미지와 그에 연결된 정보 |
| Label / Class | 이미지의 정답 / 구분할 범주 |
| Class ID | 클래스를 나타내는 정수 |
| `class_to_idx` | 클래스 폴더 이름 → 정수 라벨 매핑 |
| Train | 파라미터를 학습하는 데이터 |
| Valid | 학습 설정과 모델을 선택하는 데이터 |
| Test | 선택 완료 후 최종 성능을 확인하는 데이터 |
| Data Contract | 이미지 경로·라벨·그룹·입력 형식 등의 규칙 |
| Group ID | 동일 원본·사람·제품·영상 등을 묶는 식별자 |
| Data Leakage | 학습 정보가 평가 데이터에 유입되는 현상 |
| Duplicate / Near Duplicate | 동일 파일 / 거의 같은 장면 |
| Annotation / Annotator | 정답 부여 작업 / 정답을 부여하는 사람 |
| Data Distribution | 클래스·배경·카메라·조명 등의 데이터 분포 |
| RGB / BGR | 이미지 색상 채널 순서 |
| Resize / Interpolation | 크기 변경 / 픽셀값 보간 |
| Normalize | 입력값을 지정된 기준으로 정규화 |
| Augmentation | 라벨 의미를 유지하며 학습 이미지 다양화 |

### 폴더 구조와 매핑

```text
data/
├── train/
│   ├── circle/
│   ├── square/
│   └── triangle/
├── valid/
│   ├── circle/
│   ├── square/
│   └── triangle/
└── test/
    ├── circle/
    ├── square/
    └── triangle/
```

```python
from collections import Counter
from torchvision.datasets import ImageFolder

train = ImageFolder("data/train")
valid = ImageFolder("data/valid")
test = ImageFolder("data/test")

assert train.class_to_idx == valid.class_to_idx
assert train.class_to_idx == test.class_to_idx

print(train.class_to_idx)
for ds in (train, valid, test):
    print(ds.root, Counter(ds.targets))
```

`ImageFolder(root, transform=None, target_transform=None, loader=..., is_valid_file=None, allow_empty=False)`의 핵심은 클래스별 하위 폴더를 읽어 `classes`, `class_to_idx`, `samples`, `targets`를 제공한다는 점이다. **클래스 폴더명 정렬 순서가 숫자 라벨을 결정**하므로 모든 분할에서 동일해야 한다.

### 데이터 누수 방지

| 상황 | 권장 분할 단위 |
|---|---|
| 서로 독립적인 상품 사진 | 이미지 |
| 동일 사람·환자의 사진 | 사람·환자 ID |
| 동영상 Frame | 영상·장면 ID |
| 원본과 증강 이미지 | 원본 ID |

- 같은 원본이나 거의 같은 장면을 train과 test에 나누지 않는다.
- 라벨 기준의 모호함, 중복, 배경·촬영 조건 편향을 확인한다.
- 학습에는 허용된 무작위 증강을 사용하되, valid/test 입력은 일관되게 평가한다.
- RGB·크기·Normalize는 선택한 사전학습 Weight의 입력 조건과 맞춘다.

**핵심:** 좋은 분류 모델은 데이터셋 품질과 올바른 분할에서 시작한다.

## 3. 2교시 — 사전학습 Backbone·Head

**범위:** PDF 27~36페이지

### 용어

| 용어 | 의미 |
|---|---|
| Transfer Learning | 다른 데이터에서 학습한 표현을 새 문제에 재사용 |
| Pretrained Model / Weight | 사전학습 모델 / 학습된 파라미터 |
| Backbone | 이미지의 특징을 추출하는 부분 |
| Head | 특징을 현재 클래스의 점수로 바꾸는 분류 부분 |
| Feature / Feature Vector | 추출한 특징 / 특징을 모은 벡터 |
| ResNet18 | 잔차 연결을 사용하는 이미지 분류 모델 |
| Linear / FC | 가중합과 Bias로 출력값을 계산하는 층 |
| Logit | Softmax 전 클래스별 원시 점수 |
| Weight / Bias | 입력 특징의 영향력 / 가중합에 더하는 값 |
| Parameter | 학습 가능한 모델 값 |
| Head-only | Backbone을 고정하고 Head만 학습 |
| Source / Target Domain | 사전학습 데이터 영역 / 현재 문제 영역 |
| VRAM | GPU 메모리 |
| BatchNorm | Batch 통계와 Running 통계를 사용하는 정규화 층 |
| Dropout | 학습 중 일부 출력을 무작위로 제외하는 기법 |

### 모델 구조

```text
RGB 이미지 Batch
      ↓
ResNet18 Backbone (특징 추출·풀링)
      ↓
512차원 특징 벡터
      ↓
새 Head: nn.Linear(512, 3)
      ↓
3개 클래스 Logit
```

`Linear(512, 3)`의 파라미터는 Weight `3 × 512 = 1,536`개와 Bias `3`개로 **총 1,539개**다. 선형층은 `y = Wx + b`로 이해할 수 있다.

```python
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

num_classes = 3
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

for p in model.parameters():
    p.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
preprocess = weights.transforms()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable:", trainable)
```

**주요 API:** `resnet18(weights=...)`, `ResNet18_Weights.DEFAULT`, `weights.transforms()`, `model.fc.in_features`, `nn.Linear`, `parameters()`, `numel()`, `requires_grad`.

### 주의: Freeze와 모드는 별개

- `requires_grad=False`: 해당 파라미터의 Gradient 계산을 비활성화한다.
- `model.train()`: BatchNorm·Dropout 등이 학습 모드로 동작한다.
- `model.eval()`: BatchNorm·Dropout 등이 평가 모드로 동작한다.
- Backbone을 고정해도 BatchNorm의 Running 통계가 자동으로 고정되는 것은 아니다.

**핵심:** Backbone은 특징을 추출하고 Head는 현재 문제의 클래스 점수를 계산한다.

## 4. 3교시 — Freeze·Fine-tuning

**범위:** PDF 37~46페이지

### 용어

| 용어 | 의미 |
|---|---|
| Freeze / Unfreeze | 파라미터 고정 / 고정 해제 |
| Fine-tuning | 사전학습 파라미터를 새 데이터에 맞게 추가 학습 |
| Partial / Full Fine-tuning | 일부 Backbone+Head / 전체 Backbone+Head 학습 |
| Scratch | 사전학습 Weight 없이 처음부터 학습 |
| Forward / Backpropagation | 순전파 / Gradient를 계산하는 역전파 |
| Gradient | 손실의 파라미터에 대한 변화율 |
| Optimizer | Gradient를 이용해 파라미터 갱신 |
| Learning Rate (LR) | 갱신 폭을 조정하는 학습률 |
| Epoch | 학습 데이터를 한 차례 사용하는 학습 단위 |
| AdamW | Weight Decay를 분리 적용하는 Adam 계열 Optimizer |
| Weight Decay | 가중치 크기에 대한 규제 설정 |
| Parameter Group | 파라미터 묶음별 학습률·설정 지정 |
| Overfitting | 학습 데이터에 과도하게 맞춰지는 현상 |
| Baseline | 비교 기준 모델 |
| Checkpoint | 저장된 모델 상태 |
| Best Epoch | 선택한 검증 지표가 가장 좋았던 Epoch |
| Scheduler / Gradient Norm | 학습 설정 조절 도구 / Gradient 크기 요약 |

### 전략 비교

| 전략 | 앞쪽 Backbone | `layer4` | 새 Head | 특징 |
|---|---|---|---|---|
| Head-only | 고정 | 고정 | 학습 | 빠르고 비용이 적은 기준선 |
| Partial | 고정 | 학습 | 학습 | 마지막 특징부터 조정 |
| Full | 학습 | 학습 | 학습 | 비용·과적합 위험 증가 |

PDF의 **학습률 예시**:

| 단계 | 학습 범위 | 학습률 예 |
|---|---|---|
| A | Head | `1e-3` |
| B | layer4 + Head | `1e-4` / `1e-3` |
| C | 전체 Backbone + Head | `1e-5` / `5e-4` |

```python
import torch

for p in model.layer4.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
], weight_decay=1e-4)
```

**관련 API:** `named_parameters()`, `requires_grad_(True)`, `AdamW`, `params`, `lr`, `weight_decay`, `betas`, `eps`.

**실수 방지:** Unfreeze한 파라미터가 Optimizer에 포함되어 있는지 확인한다. 학습 전략을 비교할 때는 분할·평가 기준·학습량·시작 Checkpoint를 기록한다.

### PDF의 설명용 결과 (42페이지)

| 전략 | Valid Accuracy | Macro F1 |
|---|---:|---:|
| Scratch | 0.911 | 0.911 |
| Head-only | 0.389 | 0.269 |
| Partial FT | 0.678 | 0.674 |

> **주의:** 위 표는 설명용 소형 모델 예시이며 ResNet18 결과가 아니다. 전이학습이 항상 Scratch보다 좋은 것은 아니다.

## 5. 4교시 — 클래스 불균형·Loss·Sampler

**범위:** PDF 47~58페이지

### 용어

| 용어 | 의미 |
|---|---|
| Class Imbalance | 클래스별 데이터 개수가 불균등한 상태 |
| Majority / Minority Class | 다수 / 소수 클래스 |
| Loss / Loss Function | 예측과 정답 차이 / 이를 계산하는 함수 |
| Cross Entropy Loss | 대표적인 분류 손실 함수 |
| Class Weight / Weighted Loss | 클래스별 손실 기여도 조정 / 가중 손실 |
| Sampling / Weighted Sampling | 샘플 추출 / 추출 가중치 적용 |
| Oversampling | 특정 샘플의 학습 등장 빈도 증가 |
| Replacement | 같은 샘플을 반복 추출할 수 있는 복원 추출 |
| DataLoader / Batch Size | Batch 공급 도구 / Batch의 샘플 수 |
| Threshold / Post-processing | 판단 임계값 / 출력 후처리 |
| Label Smoothing | 정답 라벨의 확률 목표를 완화 |
| Reduction | 손실값의 평균·합계·개별 반환 방식 |

### 불균형 예시

| 클래스 | Train 이미지 수 |
|---|---:|
| circle | 120 |
| square | 40 |
| triangle | 20 |

모든 이미지를 circle로만 예측해도 `Accuracy = 120 / 180 ≈ 66.7%`가 된다. 하지만 square와 triangle의 Recall은 0%다.

### 대응 전략

| 방법 | 바꾸는 대상 | 주의점 |
|---|---|---|
| Class Weight | Loss 기여도 | 지나친 가중치로 학습 불안정 가능 |
| Weighted Sampler | Batch 표본 빈도 | 중복 Sampling과 추출 확률 기록 |
| 데이터 수집·증강 | 학습 자료의 범위 | 라벨 의미·편향 검증 |
| Threshold·후처리 | 운영 판단 규칙 | 다중 클래스·확률 보정 고려 |

```python
# 클래스별 Loss Weight: 수업 자료 예시
counts = torch.tensor([120, 40, 20], dtype=torch.float)
class_weight = counts.sum() / counts
class_weight = class_weight / class_weight.mean()
loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))
# 대략 tensor([0.30, 0.90, 1.80])
```

```python
from torch.utils.data import DataLoader, WeightedRandomSampler

targets = torch.tensor(train_dataset.targets)
counts = torch.bincount(targets)
assert torch.all(counts > 0)
class_weight = 1.0 / counts.float()
sample_weight = class_weight[targets]

sampler = WeightedRandomSampler(
    sample_weight,
    num_samples=len(sample_weight),
    replacement=True,
)
loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

`CrossEntropyLoss(weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0)`에서 `weight`는 **클래스별** 가중치다. `WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)`에서 `weights`는 **샘플별** 가중치다. `sampler`를 사용할 때 `DataLoader(shuffle=True)`를 함께 지정하지 않는다.

**핵심:** Class Weight는 손실 기여도를, Weighted Sampler는 샘플 추출 빈도를 바꾼다. valid/test의 분포는 평가 목적에 맞게 유지한다.

## 6. 5교시 — 평가·Confusion Matrix·오류 분석

**범위:** PDF 59~68페이지

### 용어

| 용어 | 의미 |
|---|---|
| Evaluation / Inference | 성능 평가 / 학습된 모델로 예측 |
| Prediction / Ground Truth | 예측 결과 / 실제 정답 |
| Accuracy | 전체 정답 비율 |
| Precision | 특정 클래스로 예측한 것 중 정답 비율 |
| Recall | 실제 특정 클래스 중 찾아낸 비율 |
| F1-score | Precision과 Recall의 조화평균 |
| Support | 해당 클래스의 실제 표본 수 |
| Confusion Matrix | 실제 클래스와 예측 클래스의 교차 빈도표 |
| TP / FP / FN / TN | 참양성 / 거짓양성 / 거짓음성 / 참음성 |
| Macro Average | 클래스별 지표의 단순 평균 |
| Weighted Average | Support를 반영한 가중평균 |
| Classification Report | 클래스별 Precision·Recall·F1·Support 표 |
| Misclassification | 오분류 |
| Error Taxonomy / Gallery | 오류 유형 분류 / 오류 이미지 모음 |
| Error Slice | 특정 조건의 데이터 부분집합 |
| Bootstrap / Confidence Interval | 재표본추출 / 추정 불확실성 구간 |

### 핵심 지표

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
Accuracy  = 전체 정답 수 / 전체 샘플 수
Macro F1  = 클래스별 F1의 단순 평균
```

TP·FP·FN·TN은 다중 클래스 분류에서 **특정 클래스를 기준으로** 해석한다.

### Confusion Matrix 예시 (PDF 60페이지)

**행 = 실제, 열 = 예측**

| 실제 \ 예측 | circle | square | triangle |
|---|---:|---:|---:|
| circle | 30 | 0 | 0 |
| square | 8 | 22 | 0 |
| triangle | 5 | 4 | 21 |

circle Recall은 `30/30`, triangle Recall은 `21/30`이다.

```python
from sklearn.metrics import confusion_matrix, classification_report

labels = list(range(len(class_names)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
print(cm)
print(report["macro avg"]["f1-score"])
```

`classification_report`의 주요 인자: `y_true`, `y_pred`, `labels`, `target_names`, `sample_weight`, `digits`, `output_dict`, `zero_division`.

### 오류 원인 후보

1. **데이터 부족:** 소수 클래스의 다양한 형태·배경·조명이 부족함
2. **도메인 차이:** 사전학습 표현과 현재 촬영 환경이 다름
3. **라벨 문제:** 모호한 기준·오라벨·중복
4. **모델 문제:** 입력 크기·증강·동결 범위·학습률 문제

**핵심:** Accuracy만으로 판단하지 않는다. Macro F1·클래스별 Recall·Support·Confusion Matrix·오류 이미지를 함께 확인한다.

## 7. 6교시 — Grad-CAM·편향·설명 한계

**범위:** PDF 69~78페이지

### 용어

| 용어 | 의미 |
|---|---|
| Grad-CAM | 클래스 점수 Gradient와 Feature Map으로 관련 영역을 시각화 |
| Explainability / Interpretability | 모델 결과의 설명 / 해석 가능성 |
| Activation / Feature Map | 층의 활성화 값 / 공간적 특징 배열 |
| Convolution | 국소 필터로 특징을 추출하는 연산 |
| Target Class / Target Layer | 분석할 클래스 / 분석에 사용할 층 |
| Heatmap / Overlay | 값의 크기 시각화 / 원본과 겹쳐 표시 |
| Channel Weight | 채널별 Grad-CAM 중요도 |
| ReLU | 음수를 0으로 만드는 함수 |
| Hook / Forward Hook | 계산 중 값 수집 기능 / 순전파 값 수집 |
| Spatial Resolution | 특징맵의 공간적 세밀함 |
| Dataset Bias | 데이터의 편향된 단서에 의존하는 현상 |
| Occlusion / Counterfactual | 일부 영역 가림 / 조건을 바꾼 비교 |
| Causal Explanation | 결과의 인과적 원인에 대한 설명 |

### 계산 순서

```text
입력 이미지 Forward
       ↓
관심 클래스 Logit 선택
       ↓
Feature Map에 대한 Gradient 계산
       ↓
Gradient의 공간 평균 → 채널별 Weight
       ↓
Feature Map 가중합 + ReLU
       ↓
Heatmap 확대 및 원본 Overlay
```

**관련 API:** `register_forward_hook()`, `register_hook()`, `detach()`, `requires_grad_(True)`, `torch.enable_grad()`, `model.zero_grad()`, `backward()`, `mean(dim=(1,2))`, `torch.relu()`, `interpolate()`, `handle.remove()`.

- 깊은 Convolution 층: 클래스 의미가 강하지만 공간 해상도는 낮다.
- 얕은 Convolution 층: 가장자리·질감 정보가 세밀하지만 클래스 의미는 상대적으로 약하다.
- Grad-CAM은 분석용 Gradient를 계산하지만, **Optimizer로 모델을 갱신하는 단계가 아니다**.
- Heatmap은 관련 영역에 대한 단서이지 인과적 설명의 증거가 아니다. 가림 실험 등으로 원인 가설을 확인한다.

**핵심:** Grad-CAM으로 모델이 어떤 영역과 관련된 신호를 사용했는지 살펴보되, Heatmap만으로 원인을 확정하지 않는다.

## 8. 7교시 — 모델 비교·프로젝트 통합

**범위:** PDF 79~89페이지

### 용어

| 용어 | 의미 |
|---|---|
| Experiment / Experiment Contract | 실험 / 비교 시 고정·기록할 조건 |
| Fair Comparison | 동일한 기준의 비교 |
| Reproducibility | 설정·환경을 기록해 재실행 가능한 성질 |
| Random Seed | 난수 생성의 초기값 |
| Config / Run ID | 실험 설정 / 실행 식별자 |
| Learning Curve | Epoch별 성능 변화 그래프 |
| Checkpoint / `state_dict` | 저장된 학습 상태 / 모델 상태 딕셔너리 |
| Artifact / Manifest | 생성된 결과물 / 결과물 목록·정보 |
| Model Card | 모델의 성능·한계·사용 조건 설명 |
| Pipeline / Workflow | 처리 단계 연결 / 작업 흐름 |
| Regression Test | 변경 후 기존 동작 유지 검사 |
| Input Contract | 입력 크기·채널·정규화 규칙 |

### 공정한 비교를 위한 기록

| 고정하거나 기록할 조건 | 목적 |
|---|---|
| train/valid/test, 클래스 매핑 | 동일 데이터·정답 기준 |
| 평가 전처리, 주요 지표 | 동일 평가 기준 |
| Weight, 시작 Checkpoint | 시작 상태 확인 |
| Freeze 범위, LR, Epoch | 학습 전략·학습량 기록 |
| BatchNorm, Loss, Sampler, Seed | 실험 차이와 재현 조건 기록 |
| 학습 시간, VRAM | 비용·효율 비교 |

PDF의 수업용 합성 실험에서 전략별 Macro F1 예시는 Scratch `0.911`, Head-only `0.269`, Partial FT `0.674`, Class Weight `0.763`, Sampler `0.810`이다. **설명용 예시이며 일반적인 성능 순위를 뜻하지 않는다.**

### 프로젝트 구조 (수업 예시)

```text
day14_transfer/
├── main.py
├── data.py
├── models.py
├── train.py
├── evaluate.py
├── explain.py
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── outputs/
│   ├── metrics.csv
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── errors/
│   └── gradcam/
└── reports/
    └── model_card.md
```

### 검증 시나리오

- 정상 3클래스 데이터: 라벨·Batch·학습 정상 동작
- Minority 클래스 1장: 지표 계산과 경고 처리
- Train/Test 중복: 누수 감사에서 발견
- 사전학습 Weight 없음: 임의의 Weight로 몰래 대체하지 않음
- 저장 모델 복원: 같은 입력에서 Logit이 허용오차 내 일치

**핵심:** 점수뿐 아니라 분할·설정·학습량·오류 근거를 함께 저장해야 결과를 설명하고 재현할 수 있다.

## 9. 8교시 — 실습평가·검증·회고

**범위:** PDF 90~99페이지

### 용어

| 용어 | 의미 |
|---|---|
| Deliverable | 최종 제출물 |
| Validation / Model Restoration | 요구사항 검증 / 저장 모델 복원 |
| Checkpoint Resume | 저장 상태에서 학습 재개 |
| Environment / Device | 실행 환경 / CPU·GPU 등 장치 |
| CUDA | NVIDIA GPU 병렬 연산 플랫폼 |
| AMP | 자동 혼합 정밀도 |
| Distributed Training | 여러 장치에 분산하는 학습 |
| Data Versioning | 데이터 버전 관리 |
| Nondeterminism | 실행마다 결과가 달라질 수 있는 성질 |
| Git Commit / README | 변경 이력 단위 / 프로젝트 안내 문서 |
| Rubric / Demonstration | 평가 기준 / 결과 시연 |
| Deployment | 실제 사용 환경에 모델 적용 |

### 최종 제출물

- 실행 코드와 환경·설정 설명
- `metrics.csv`: 주요 지표와 실험 설정
- `predictions.csv`: 이미지별 실제 라벨·예측 라벨·확률
- `confusion_matrix.png`: 오분류 현황
- `errors/`: 오류 이미지
- `gradcam/`: 클래스 관련 영역 Heatmap
- `model_card.md`: 모델 성능·한계·개선 방향
- 재실행 안내 및 변경 이력

### 최종 점검

- [ ] train/valid/test의 클래스 매핑이 일치한다.
- [ ] 동일 원본·그룹의 분할 누수를 검사했다.
- [ ] 사전학습 Weight와 전처리 규칙을 기록했다.
- [ ] Head-only 모델의 학습 가능 파라미터를 확인했다.
- [ ] Partial/Full 학습 시 Optimizer 파라미터를 확인했다.
- [ ] Accuracy, Macro F1, 클래스별 Recall을 확인했다.
- [ ] Confusion Matrix와 대표 오분류 이미지를 저장했다.
- [ ] Grad-CAM의 Target Class·Layer를 기록했다.
- [ ] 실험 조건과 결과를 저장했다.
- [ ] README를 보고 재실행할 수 있다.

## 10. 핵심 용어 암기표

| 우선순위 | 용어 | 한 줄 암기 |
|---|---|---|
| ★★★ | Dataset | 학습·평가 데이터 집합 |
| ★★★ | Data Leakage | 학습 정보가 평가 데이터에 유입 |
| ★★★ | `class_to_idx` | 클래스 이름과 숫자 라벨 연결 |
| ★★★ | Transfer Learning | 사전학습 표현 재사용 |
| ★★★ | Backbone | 이미지 특징 추출 |
| ★★★ | Head | 특징을 클래스 점수로 변환 |
| ★★★ | Freeze | 파라미터 고정 |
| ★★★ | Fine-tuning | 사전학습 모델 추가 학습 |
| ★★★ | `requires_grad` | Gradient 계산 여부 |
| ★★★ | Optimizer | 파라미터 갱신 |
| ★★★ | Learning Rate | 갱신 폭 |
| ★★★ | Loss | 예측과 정답 차이 |
| ★★★ | Class Imbalance | 클래스 개수 불균형 |
| ★★★ | Class Weight | Loss 기여도 조정 |
| ★★★ | Weighted Sampler | 추출 빈도 조정 |
| ★★★ | Accuracy | 전체 정답 비율 |
| ★★★ | Precision | 예측한 것 중 정답 비율 |
| ★★★ | Recall | 실제 대상 중 찾아낸 비율 |
| ★★★ | F1-score | Precision·Recall 조화평균 |
| ★★★ | Macro F1 | 클래스별 F1 단순 평균 |
| ★★★ | Confusion Matrix | 실제·예측 교차표 |
| ★★★ | Grad-CAM | 클래스 관련 영역 시각화 |
| ★★☆ | BatchNorm | Batch·Running 통계 기반 정규화 |
| ★★☆ | Checkpoint | 저장된 모델 상태 |
| ★★☆ | Reproducibility | 실험 재현 가능성 |

> 우선순위 표시는 학습 편의를 위해 정리한 것이며 PDF의 공식 평가 중요도는 아니다.

---

### 학습 메모

- **1~3교시:** 데이터 준비 → 특징 추출 → 학습 범위 이해
- **4~5교시:** 불균형 대응 → 성능 지표와 오류 해석
- **6~8교시:** Grad-CAM → 모델 비교 → 재현 가능한 결과물

**전체 핵심:** `데이터 품질 → 특징 추출 → 학습 범위 → 불균형 대응 → 성능 평가 → 오류 분석 → 재실험`

**출처:** 제공된 `Day14_20261105.pdf`의 교시별 개념·실습·설명용 수치를 정리. 이 README는 학습용 요약이며 전체 실습 코드의 완성본은 아니다.
