# Day 11. 딥러닝 핵심 아키텍처 및 학습

> **학습일:** 2026-11-02
>
> **주제:** 이미지 Tensor → CNN·Pooling → 이미지 분류 → 전이학습 → CPU/GPU 비교
>
> **기준 자료:** `20261102.pdf`

---

## 1. 오늘 수업의 전체 흐름

이미지 분류 모델을 만드는 과정은 다음과 같다.

1. 이미지 파일을 읽고 모델 입력 형태인 Tensor로 변환한다.
2. CNN의 합성곱(Convolution) 층으로 이미지의 국소 패턴을 추출한다.
3. Pooling 또는 Stride로 공간 크기를 줄인다.
4. 분류 Head가 추출된 특징을 클래스별 Logit으로 바꾼다.
5. Train 데이터로 학습하고 Valid 데이터로 모델을 선택한다.
6. 필요하면 사전학습 모델(ResNet18)을 전이학습한다.
7. CPU/GPU의 학습 시간·처리량·메모리를 같은 조건에서 비교한다.

**핵심:** CNN은 이미지의 공간적 관계를 활용하고, 모델 개발에서는 데이터 준비·학습·검증·성능 측정까지 함께 관리한다.

---

## 2. 이미지 Tensor 기본 개념

### 2-1. Tensor

- **Tensor**: 숫자를 여러 축으로 묶어 표현하는 자료 구조. 이미지와 모델의 입력·출력을 담는다.
- 이미지 한 장은 보통 `[C, H, W]`, 여러 장을 묶은 Batch는 `[N, C, H, W]` 형태를 사용한다.
- PyTorch의 `Conv2d` 입력은 기본적으로 NCHW 순서다.

| 축 | 의미 | 설명 |
|----|------|------|
| N | Batch | 한 번에 처리하는 이미지 수 |
| C | Channel | 색상 또는 특징 채널 수. RGB 이미지는 3 |
| H | Height | 이미지 높이 |
| W | Width | 이미지 너비 |

예: `[8, 3, 64, 64]`는 RGB 이미지 8장, 각 이미지 크기 64×64를 뜻한다.

### 2-2. dtype과 device

- **dtype**: Tensor에 저장된 값의 자료형. 이미지 입력은 보통 `torch.float32`, 클래스 정답은 정수형 `long`을 사용한다.
- **device**: Tensor 또는 모델이 연산되는 장치. 예: `cpu`, `cuda:0`.
- 모델과 입력 Tensor는 같은 device에 있어야 한다.

### 2-3. Tensor 축 변경

- `unsqueeze(dim)`: 지정한 위치에 크기 1인 축을 추가한다. `[C,H,W]` → `[1,C,H,W]`는 `unsqueeze(0)`.
- `permute(*dims)`: 기존 축의 순서를 바꾼다. `[C,H,W]` → `[H,W,C]`는 `permute(1,2,0)`.
- 축 순서가 바뀐 Tensor는 `Conv2d`가 기대하는 NCHW와 맞는지 확인해야 한다.

---

## 3. ANN 대신 CNN을 사용하는 이유

### MLP 방식

- 이미지를 `Flatten`으로 긴 1차원 벡터로 펼친 뒤 Linear 층에 넣는다.
- 픽셀의 위치 관계가 벡터 안에서 직접 드러나지 않는다.
- 이미지 해상도가 커지면 첫 Linear 층의 파라미터 수가 크게 증가한다.

### CNN 방식

- 작은 Kernel로 주변 픽셀을 함께 살펴본다.
- 같은 Kernel 가중치를 이미지의 여러 위치에서 공유한다.
- 이미지의 공간적 위치 관계를 Feature Map에 유지한다.
- 자료의 예시: `Conv 3→16, 3×3`은 448개 파라미터를 가진다.

### 용어

- **CNN (Convolutional Neural Network)**: 합성곱 연산을 이용해 이미지의 공간 패턴을 학습하는 신경망.
- **국소 연결(Local Connectivity)**: 한 연산이 이미지 전체가 아니라 가까운 일부 영역을 살펴보는 구조.
- **가중치 공유(Weight Sharing)**: 같은 Kernel 가중치를 입력의 여러 위치에서 반복 사용하는 방식.

---

## 4. Convolution과 Kernel

### 4-1. 합성곱(Convolution)

Kernel을 입력 이미지 위에서 이동시키며, 각 위치의 입력값과 Kernel 값을 곱해 모두 더한다. 이 계산 결과가 Feature Map의 한 위치 값이 된다.

### 4-2. 주요 용어

| 용어 | 의미 |
|------|------|
| Kernel / Filter | 입력의 작은 영역을 살펴보는 학습 가능한 가중치 묶음 |
| Feature Map | Kernel 연산 결과로 만들어지는 특징 반응 지도 |
| `kernel_size` | Kernel의 가로·세로 크기 |
| `stride` | Kernel이 한 번 이동하는 간격 |
| `padding` | 입력 가장자리에 추가하는 테두리 |
| `in_channels` | 입력 채널 수 |
| `out_channels` | 생성할 Feature Map 수이자 Kernel 개수 |
| `dilation` | Kernel 요소 사이의 간격 |
| `groups` | 입력·출력 채널 연결을 나누는 설정 |
| `bias` | 합성곱 결과에 더하는 학습 가능한 편향값 |

- RGB 입력은 `in_channels=3`이다.
- 일반적인 `groups=1`에서는 각 Kernel이 모든 입력 채널을 함께 본다.
- 가중치 Shape는 `[Cout, Cin, kH, kW]`, 출력 Shape는 `[N, Cout, Hout, Wout]`이다.
- Kernel 하나는 Feature Map 하나를 만든다. Kernel을 여러 개 쓰면 여러 Feature Map이 생성된다.
- 실제 CNN의 Kernel 값은 사람이 고정하는 것이 아니라 학습 과정에서 조정된다.

### 4-3. Conv2d 출력 크기

자료의 출력 크기 공식:

$$
H_{out}=\left\lfloor \frac{H_{in}+2P-D(K-1)-1}{S}+1 \right\rfloor
$$

- `Hin`: 입력 높이
- `P`: Padding
- `D`: Dilation
- `K`: Kernel 크기
- `S`: Stride
- 너비도 같은 방식으로 계산한다.

예시:

- 입력 32×32, Kernel 3, Stride 1, Padding 1 → 32×32 유지
- 입력 32×32, Kernel 3, Stride 2, Padding 1 → 16×16

### 4-4. Conv2d 선택 설정

- `stride > 1`: 합성곱과 Downsampling을 함께 수행할 수 있다.
- `padding="same"`: 자료 기준 Stride 1에서 공간 크기를 유지한다.
- `dilation > 1`: 파라미터 수를 늘리지 않고 수용 영역을 넓힌다.
- `groups=Cin`: Depthwise Convolution 구성에 사용한다.
- 뒤에 BatchNorm을 두는 구조에서는 `bias=False`를 자주 사용한다.

---

## 5. Pooling과 CNN 구성 요소

### 5-1. Pooling

작은 영역을 대표값으로 바꾸어 공간 크기를 줄이는 연산이다.

- **Max Pooling**: 영역 안에서 가장 큰 값을 남긴다.
- 공간 크기 H·W를 줄이고 보통 채널 수 C는 유지한다.
- 연산량과 메모리를 줄이고 작은 위치 변화에 대한 민감도를 낮출 수 있다.
- 너무 자주 적용하면 작은 물체나 세부 정보가 사라질 수 있다.

### 5-2. 주요 모듈

| 모듈 | 역할 |
|------|------|
| `nn.Conv2d` | 국소 패턴을 학습 가능한 Feature Map으로 변환 |
| `nn.BatchNorm2d` | 채널별 활성값을 안정화하고 Running 통계를 유지 |
| `nn.ReLU` | 음수 값을 0으로 바꾸는 비선형 활성화 함수 |
| `nn.MaxPool2d` | 영역의 최대값을 남겨 공간 크기 축소 |
| `nn.AdaptiveAvgPool2d(1)` | 각 채널의 공간 정보를 1×1 평균값으로 요약 |
| `nn.Sequential` | 모듈을 작성한 순서대로 실행 |
| `torch.flatten(x, 1)` | Batch 축은 유지하고 나머지 축을 펼침 |
| `nn.Linear` | 입력 특징을 클래스별 점수로 변환 |

### 5-3. CNN의 전체 구조

1. **특징 추출부(Feature Extractor)**: Conv 블록을 거치며 이미지 패턴을 추출한다.
2. **Global Pool / Flatten**: 공간 정보를 벡터 형태로 요약한다.
3. **분류 Head(Classifier Head)**: 실제 클래스 수에 맞는 Logit을 출력한다.

자료의 Shape 예시:

| 단계 | 연산 | 출력 Shape |
|------|------|------------|
| 입력 | RGB Batch | `[8,3,64,64]` |
| Block 1 | Conv 3→16 + Pool 2 | `[8,16,32,32]` |
| Block 2 | Conv 16→32 + Pool 2 | `[8,32,16,16]` |
| Global Pool | AdaptiveAvgPool2d(1) | `[8,32,1,1]` |
| Flatten | `flatten(1)` | `[8,32]` |
| 분류 | Linear 32→3 | `[8,3]` |

### 5-4. 자주 발생하는 Shape 오류

- NCHW 대신 NHWC를 Conv2d에 넣어 채널 수 불일치가 발생하는 경우
- Pooling 뒤 크기를 계산하지 않고 Linear의 `in_features`를 잘못 지정하는 경우
- `torch.flatten(x)`로 Batch 축까지 펼쳐 샘플 구분이 사라지는 경우
- 실제 클래스 폴더 수와 `num_classes`가 맞지 않는 경우

---

## 6. 이미지 데이터셋과 전처리

### 6-1. ImageFolder

- `torchvision.datasets.ImageFolder`는 클래스별 하위 폴더를 읽어 이미지와 정수 라벨을 제공한다.
- 예시 폴더 구조:

```text
data/
├── train/
│   ├── class_a/
│   └── class_b/
├── valid/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

- 폴더 이름을 정렬해 `class_to_idx`를 만들고 이미지마다 클래스 인덱스를 부여한다.
- train·valid·test에서 클래스 폴더 이름과 의미를 동일하게 유지한다.
- 데이터 분할 전 중복 이미지나 같은 출처에서 나온 이미지 그룹이 섞이지 않는지 점검한다.

### 6-2. Transforms

이미지를 모델이 처리할 수 있는 크기와 값의 형태로 바꾸는 과정이다.

| 변환 | 역할 |
|------|------|
| `transforms.Compose` | 여러 변환을 순서대로 연결 |
| `Resize` | 입력 크기 조정 |
| `CenterCrop` | 중앙 영역을 잘라냄 |
| `ToTensor` | 이미지를 Tensor `[C,H,W]`로 변환. 보통 값 범위는 `[0,1]` |
| `Normalize` | 채널별 평균·표준편차로 정규화 |

Normalize 공식:

$$
output[c]=\frac{input[c]-mean[c]}{std[c]}
$$

일반적인 `Normalize`는 Tensor에 적용하므로 `ToTensor` 다음에 둔다.

### 6-3. Data Augmentation

같은 클래스 의미를 유지하면서 다양한 입력 형태를 만들어 학습 데이터의 다양성을 높이는 방법이다.

- **Horizontal Flip**: 좌우 반전
- **Rotation**: 회전
- **Crop**: 일부 영역 자르기
- **ColorJitter**: 밝기·대비·채도·색조 변화

주의:

- 변환 후에도 클래스 정답의 의미가 유지되어야 한다.
- 좌우 방향이 클래스 의미를 바꾸는 데이터라면 Flip을 적용하면 안 될 수 있다.

### 6-4. Train과 Valid/Test 변환을 분리하는 이유

| 구분 | 변환 원칙 | 목적 |
|------|-----------|------|
| Train | 의미를 보존하는 무작위 증강 + Tensor + Normalize | 다양한 입력으로 가중치 학습 |
| Valid/Test | Resize·CenterCrop 등 결정적 변환 + Tensor + Normalize | 동일 조건으로 비교·최종 평가 |

Valid/Test에 무작위 증강을 적용하면 평가할 때마다 입력이 달라져 비교가 어려워진다.

### 6-5. DataLoader

Dataset에서 데이터를 Batch 단위로 묶어 모델에 공급한다.

- `batch_size`: 한 번에 묶는 샘플 수
- `shuffle`: 데이터 순서를 섞을지 여부. 일반적으로 Train은 `True`
- `num_workers`: 데이터 로딩에 사용하는 프로세스 수
- `pin_memory`: GPU 전송을 위한 고정 메모리 사용 설정
- `drop_last`: 마지막에 남은 작은 Batch를 버릴지 여부

출력 형태:

- images: `[B,C,H,W]`
- labels: `[B]`

---

## 7. CNN 학습과 평가

CNN의 학습 루프는 ANN과 기본 원리가 같다. 입력 데이터와 모델 구조가 이미지에 맞게 바뀐다.

### 7-1. 한 Batch의 학습 순서

1. `images`, `labels`를 device로 이동한다.
2. `logits = model(images)`로 예측 점수를 계산한다.
3. `loss_fn(logits, labels)`로 손실을 계산한다.
4. `optimizer.zero_grad()`로 이전 기울기를 초기화한다.
5. `loss.backward()`로 기울기를 계산한다.
6. `optimizer.step()`으로 가중치를 갱신한다.
7. Loss와 정답 개수를 누적한다.

### 7-2. 학습 관련 용어

| 용어 | 의미 |
|------|------|
| Logit | Softmax 적용 전의 클래스별 점수 |
| Loss | 예측과 정답의 차이를 나타내는 손실값 |
| CrossEntropyLoss | 다중 클래스 분류에서 사용하는 손실 함수 |
| Backward | Loss를 기준으로 기울기(Gradient)를 계산 |
| Optimizer | 계산된 기울기를 이용해 파라미터를 갱신 |
| Epoch | 전체 학습 데이터를 한 번 사용하는 단위 |
| Batch | 한 번의 학습 단계에서 함께 처리하는 샘플 묶음 |
| Accuracy | 전체 샘플 중 맞게 예측한 비율 |

- `CrossEntropyLoss(logits, labels)`에는 Softmax를 미리 적용하지 않는다.
- 예측 클래스는 `logits.argmax(dim=1)`로 구한다.
- 분류 라벨은 정수형 `long`이어야 한다.
- Loss 평균은 Batch 개수 기준이 아니라 샘플 수 기준으로 계산한다. 마지막 Batch 크기가 작을 수 있기 때문이다.

### 7-3. Train 모드와 Evaluation 모드

- `model.train()`: 학습 모드. Dropout·BatchNorm 등이 학습 동작을 한다.
- `model.eval()`: 평가 모드. Dropout·BatchNorm 등이 평가 동작을 한다.
- `torch.inference_mode()`: 추론·평가 중 Autograd 기록을 비활성화한다.
- `eval()`과 `inference_mode()`는 같은 기능이 아니다. 하나는 모듈 동작 모드, 다른 하나는 기울기 기록 여부를 제어한다.
- 평가 함수에서는 가중치를 갱신하지 않으며 Optimizer를 호출하지 않는다.

### 7-4. Train / Valid / Test 역할

- **Train**: 모델 가중치를 학습한다.
- **Valid**: Epoch별 성능을 확인하고 모델·설정을 선택한다.
- **Test**: 모델과 설정을 선택하는 동안 사용하지 않고 최종 평가를 위해 보관한다.

### 7-5. Best Model

- 마지막 Epoch의 모델이 반드시 가장 좋은 모델은 아니다.
- 미리 정한 Valid 지표가 가장 좋았던 시점의 가중치를 저장한다.
- 저장 시 `state_dict`, 클래스 순서, 입력 이미지 크기, Normalize의 mean/std 등을 함께 관리한다.
- 재사용 시 Checkpoint를 불러오고 `load_state_dict()`로 복원한 뒤 `eval()`로 평가한다.

### 7-6. 평가 지표

- **Accuracy**: 전체 예측 중 정답 비율
- **Precision**: 특정 클래스로 예측한 것 중 실제 해당 클래스인 비율
- **Recall**: 실제 해당 클래스 중 모델이 찾아낸 비율
- **F1-score**: Precision과 Recall을 함께 고려한 지표
- **Confusion Matrix**: 실제 클래스와 예측 클래스의 조합별 개수를 나타내는 표

Accuracy만으로 놓칠 수 있는 클래스별 오류를 Precision·Recall·F1 및 Confusion Matrix로 확인한다.

---

## 8. 전이학습(Transfer Learning)

### 8-1. 전이학습이란?

이미 대규모 데이터로 학습된 모델의 특징 표현을 새로운 데이터에 재사용하는 방법이다. 새 데이터가 적을 때 처음부터 모든 가중치를 학습하는 부담을 줄이는 출발점으로 활용한다.

### 8-2. Backbone과 Head

- **Backbone**: 이미지 특징을 추출하는 앞부분 네트워크.
- **Head**: 추출한 특징을 현재 작업의 클래스 점수로 바꾸는 분류 부분.
- 사전학습 모델의 마지막 분류 층은 현재 데이터의 클래스 수에 맞게 교체한다.

### 8-3. Freeze와 Fine-tuning

| 방식 | 학습 범위 | 설명 |
|------|-----------|------|
| Head-only / Feature Extraction | 새 Head만 학습 | Backbone은 고정하고 분류 층만 새 데이터에 맞춤 |
| Partial Fine-tuning | 마지막 Stage 등 일부 Backbone + Head | 깊은 일부 층을 열어 새 데이터에 맞게 조정 |

권장 진행 순서:

1. 먼저 Head만 학습해 기준 성능을 확인한다.
2. 필요하면 `layer4` 등 마지막 Stage를 학습 가능하게 연다.
3. 사전학습 Backbone에는 Head보다 작은 학습률을 적용하는 경우가 많다.
4. Valid 과적합과 BatchNorm 상태를 확인하고 Best Model을 저장한다.

### 8-4. ResNet18 실습 핵심

- `ResNet18_Weights.DEFAULT` 또는 자료에서 제시한 사전학습 Weight를 사용한다.
- `weights.transforms()`는 해당 Weight가 기대하는 Resize·Crop·Rescale·Normalize 전처리를 제공한다.
- `model.fc`를 현재 데이터의 클래스 수에 맞는 `nn.Linear`로 교체한다.
- Head-only에서는 Backbone 파라미터의 `requires_grad=False`로 고정하고 새 `fc`만 학습한다.
- Partial에서는 `layer4`와 `fc`를 학습 대상으로 설정할 수 있다.
- Optimizer에는 실제 학습할 파라미터를 빠뜨리지 않고 전달한다.
- Head와 Backbone에 서로 다른 학습률을 줄 수 있다. 자료 예시는 `layer4: 1e-4`, `fc: 1e-3`이다.
- BatchNorm의 Running 통계를 고정할지 정책을 정하고 매 Epoch 모드 설정에 반영한다.

### 8-5. 전이학습에서 주의할 점

- 사전학습 Weight가 기대하는 Normalize를 적용한다.
- 학습할 Head·Backbone 층이 Optimizer에 포함되어 있는지 확인한다.
- 마지막 `fc`의 출력 수와 실제 클래스 수를 맞춘다.
- Test 결과를 보고 학습 범위나 학습률을 다시 선택하지 않는다.

---

## 9. CPU·GPU 성능 비교

### 9-1. CPU, GPU, VRAM

- **CPU**: 일반 작업을 처리하는 프로세서. 비교적 적은 수의 강한 Core를 사용한다.
- **GPU**: 많은 연산 Unit으로 Tensor 연산을 병렬 처리한다.
- **VRAM**: GPU에서 사용하는 메모리. 모델·활성값·Gradient·Optimizer 상태 등이 올라간다.
- 작은 모델이나 작은 Batch에서는 데이터 전송·초기화 비용 때문에 GPU가 항상 더 빠르지는 않다.

### 9-2. 공정한 Benchmark 조건

비교할 때 다음 조건을 가능한 한 동일하게 유지하고 기록한다.

- 같은 모델과 초기 상태
- 같은 데이터 및 데이터 순서
- 같은 Batch size와 입력 이미지 크기
- 같은 Epoch 또는 반복 횟수
- 데이터 로딩 시간을 포함하는지 여부 등 측정 범위
- CPU/GPU 모델, Driver, CUDA 등 실행 환경

GPU는 비동기 실행 특성이 있으므로 측정 전후에 `torch.cuda.synchronize(device)`를 호출해 실제 연산 완료를 기다린다. 측정 시작 직전에는 `torch.cuda.reset_peak_memory_stats(device)`로 Peak 메모리 기록을 초기화한다.

### 9-3. 주요 측정값

| 항목 | 의미 |
|------|------|
| seconds / time per epoch | Epoch 1회에 걸린 시간 |
| images/s | 초당 처리한 이미지 수 |
| Peak VRAM | 측정 구간에서 PyTorch Tensor가 사용한 최대 GPU 메모리 |
| Batch size | 한 번에 처리하는 이미지 수 |

- `torch.cuda.max_memory_allocated()`는 PyTorch Tensor의 Peak allocated memory를 Byte 단위로 반환한다.
- MiB로 변환할 때는 `bytes / 1024**2`를 사용한다.
- 이 값은 시스템 전체 GPU 메모리 사용량이나 CUDA가 예약한 전체 메모리와 같지 않다.
- GPU가 없는 경우 CPU 결과와 환경을 기록하고 GPU 항목은 `N/A`로 남긴다.
- CPU 메모리는 이 자료의 CUDA 측정 함수로 측정하지 않는다.

### 9-4. Batch size 비교

자료에서는 Batch 8·16·32·64 등을 비교 대상으로 제시한다.

- Batch가 커지면 처리량이 증가할 수 있지만 VRAM 사용량도 커질 수 있다.
- 너무 큰 Batch는 OOM(Out Of Memory)을 일으킬 수 있다.
- 실제 결과를 측정하고 시간·처리량·메모리의 균형을 판단한다.

**주의:** 여기서 측정하는 것은 학습 시간이다. 서비스에서 한 장의 입력을 처리하는 추론 지연(Inference Latency)과 구분한다.

---

## 10. 프로젝트 구성과 제출 산출물

### 10-1. 프로젝트 작업 순서

1. 데이터 폴더와 클래스 이름 확인
2. Train / Valid Transform 분리
3. SmallCNN 학습 및 Best Model 저장
4. ResNet18 Head-only 또는 Partial Fine-tuning 비교
5. CPU/GPU 시간·처리량·Peak VRAM 측정
6. 결과표와 보고서 작성

### 10-2. 자료의 권장 프로젝트 구조

```text
day11_image_classification/
├── main.py                 # 학습과 평가 실행
├── models.py               # SmallCNN과 Transfer model
├── data.py                 # ImageFolder·Transforms·DataLoader
├── train.py                # train/evaluate·checkpoint·benchmark
├── outputs/
│   ├── history.csv
│   ├── metrics.csv
│   ├── confusion_matrix.*
│   └── best_model.pt
└── reports/
    └── model_comparison.md
```

### 10-3. 주요 제출물

| 산출물 | 포함할 내용 |
|--------|-------------|
| `history.csv` | Epoch별 train/valid loss·accuracy, best 표시 |
| `model_comparison.csv` | 모델, 학습 범위, Trainable Params, Valid 지표 |
| `hardware_benchmark.csv` | CPU/GPU 측정 결과와 환경. GPU 미실행은 N/A |
| `error_samples/` | 실제 오답 CSV 또는 오류 이미지. 오답 0건도 정상 |
| `best_model.pt` | state_dict, classes, image_size, Normalize 정보 |
| `README/report` | 실행법, 설정, 모델 선택 근거, 한계, 재현 조건 |

### 10-4. 자료의 기본 평가 기준

| 평가 항목 | 배점 | 확인 내용 |
|-----------|-----:|-----------|
| 데이터·Transform | 20 | 클래스 구조, Train/Valid 변환 분리, NCHW |
| CNN 구현·Shape | 20 | Conv/Pool/Global Pool/Head와 Shape 설명 |
| 학습·평가 | 25 | 학습·평가 구분, Best 저장·예측·재실행 |
| 전이학습 비교 | 15 | Freeze·전처리 설명. Weight 미준비는 미실행 기록 |
| 시간·VRAM·재현 | 20 | CPU 측정·환경·README 설명. GPU 수치 필수 아님 |

---

## 11. 핵심 개념 빠른 복습

- **CNN**: 이미지의 가까운 픽셀 관계와 공간 패턴을 학습하는 신경망.
- **Kernel**: 입력의 작은 영역을 훑으며 패턴 반응을 계산하는 학습 가능한 가중치.
- **Feature Map**: Kernel 연산으로 만들어진 특징 반응 결과.
- **Stride**: Kernel이 이동하는 간격.
- **Padding**: 입력 가장자리에 추가하는 테두리.
- **Pooling**: 작은 영역을 대표값으로 요약해 공간 크기를 줄이는 연산.
- **BatchNorm**: 채널별 활성값을 안정화하고 Running 통계를 관리하는 층.
- **ReLU**: 음수 값을 0으로 만드는 활성화 함수.
- **Global Average Pooling**: 각 채널의 공간 정보를 평균값 하나로 요약.
- **Flatten**: 여러 축을 하나의 축으로 펼치는 연산. `flatten(1)`은 Batch 축 유지.
- **Logit**: Softmax 전 클래스 점수.
- **Augmentation**: 라벨 의미를 보존하는 범위에서 학습 이미지에 변화를 주는 기법.
- **Transfer Learning**: 사전학습 모델의 특징 표현을 새 작업에 재사용.
- **Backbone**: 특징 추출부.
- **Head**: 현재 작업의 클래스 점수를 출력하는 부분.
- **Freeze**: 일부 파라미터를 학습하지 않도록 고정.
- **Fine-tuning**: 사전학습된 층 일부 또는 전체를 새 데이터에 맞게 추가 학습.
- **Best Model**: Valid 기준으로 선택해 저장한 모델.
- **Benchmark**: 정해진 조건에서 실행 시간·처리량·메모리를 측정하는 작업.
- **OOM**: 메모리가 부족해 연산을 수행하지 못하는 상태.

---

## 12. 수업 퀴즈 핵심 정답

1. 이미지를 Flatten하면 공간적으로 가까운 픽셀 관계가 벡터 안에서 직접 드러나지 않는다.
2. Dropout·BatchNorm이 있는 모델을 평가할 때 `model.eval()`을 호출한다.
3. CNN에서도 Train·Valid·Test 분리와 Best Model 원칙은 필요하다.
4. Conv2d의 출력 채널 수는 `out_channels`가 결정한다.
5. 32×32 입력에 K=3, S=1, P=1이면 공간 크기는 32×32다.
6. MaxPool2d는 보통 채널 수를 바꾸지 않고 공간 크기를 줄인다.
7. AdaptiveAvgPool2d(1)의 출력 공간 크기는 1×1이다.
8. `torch.flatten(x, 1)`의 1은 Batch 축 다음 차원부터 펼친다는 의미다.
9. ImageFolder의 클래스 인덱스는 정렬된 하위 폴더 이름의 `class_to_idx`로 정해진다.
10. Valid/Test에는 무작위 증강보다 결정적 변환을 사용한다.
11. CrossEntropyLoss에 전달하기 전에 Softmax를 직접 적용하지 않는다.
12. `eval()`은 모듈의 평가 동작, `inference_mode()`는 Autograd 기록을 제어한다.
13. 마지막 Epoch가 아니라 Valid 기준으로 Best Model을 선택한다.
14. 전이학습에서 Head-only는 새 분류 Head를 주로 학습한다.
15. GPU 시간 측정 전후 `synchronize()`가 필요한 이유는 CUDA 연산이 비동기로 실행되기 때문이다.
16. `max_memory_allocated()`는 시스템 전체 GPU 메모리가 아니라 PyTorch Tensor의 Peak allocated memory다.

---

## 13. 학습 범위 메모

이 README는 제공된 2026-11-02 수업 자료의 내용을 학습용으로 재구성한 것이다. 자료에서 다루지 않은 세부 구현이나 외부 이론은 임의로 보충하지 않았다.

다음 수업(자료 기준 2026-11-03) 예고:

- RNN·LSTM: 시계열·문장처럼 순서가 있는 입력 처리
- Autoencoder: 입력을 압축하고 다시 복원하는 학습
- Transformer: Attention 구조
