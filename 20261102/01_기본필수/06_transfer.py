"""
실습 06 · 실제 ImageNet 사전학습 가중치로 Head만 학습한다.

[새로운 개념 정리]

1. 전이학습(Transfer Learning)
   - 기존에 학습된 모델의 지식을 새로운 문제에 재사용하는 방법.
   - 처음부터 모델을 학습하는 것보다 적은 데이터와 시간으로 학습할 수 있다.

2. 사전학습 모델(Pretrained Model)
   - 대규모 데이터셋으로 미리 학습해 둔 모델.
   - 이번 실습에서는 ImageNet으로 학습된 ResNet18을 사용한다.

3. ImageNet
   - 다양한 이미지와 이미지 분류 정보를 포함하는 대규모 데이터셋.
   - ResNet18은 ImageNet 이미지로 미리 학습된 가중치를 가지고 있다.

4. ResNet18
   - 이미지 분류에 사용하는 CNN 모델.
   - 여러 합성곱 계층을 통해 이미지의 특징을 추출한다.
   - 18개의 계층으로 구성된 ResNet 모델이다.

5. Backbone
   - 모델에서 이미지의 특징을 추출하는 부분.
   - 이번 실습에서는 기존 ResNet18의 분류층을 제외한 부분이다.

6. Head
   - Backbone이 추출한 특징을 바탕으로 최종 분류를 수행하는 부분.
   - 이번 실습에서는 기존 Head를 제거하고 2개 클래스를 구분하는
     새로운 Linear 계층을 연결한다.

7. Feature Extraction
   - 기존 모델의 가중치를 고정하고 새로운 Head만 학습하는 방식.
   - 기존 모델이 학습한 이미지 특징을 그대로 활용한다.

8. requires_grad
   - 해당 파라미터의 기울기를 계산할지 결정하는 속성.
   - False: 기울기 계산 대상에서 제외한다.
   - True: 기울기를 계산하고 학습에 사용할 수 있다.

9. weights.transforms()
   - 사전학습 가중치에 맞는 이미지 전처리 방법을 제공한다.
   - 이미지 크기 조정, 자르기, 정규화 등을 포함한다.

10. BatchNorm
    - 신경망의 학습을 안정적으로 진행하도록 돕는 정규화 계층.
    - eval() 모드에서는 학습 때와 다른 평가 방식으로 동작한다.

11. logits
    - 모델이 최종 분류층에서 출력하는 클래스별 점수.
    - argmax()로 가장 높은 점수의 클래스를 선택할 수 있다.

12. CrossEntropyLoss
    - 분류 모델의 예측 결과와 실제 정답 사이의 차이를 계산하는 손실 함수.

13. optimizer
    - 손실 함수의 역전파로 계산된 기울기를 이용해 모델의 가중치를 갱신한다.

14. Feature Extraction과 Fine-tuning의 차이
    - Feature Extraction: 기존 Backbone을 고정하고 Head만 학습한다.
    - Fine-tuning: 기존 Backbone의 일부 또는 전체 가중치도 추가 학습한다.

최초 1회 공식 가중치 다운로드 및 캐시가 필요하다.
인터넷이 없으면 교사에게 요청한다.
weights=None으로 변경하면 사전학습 가중치를 사용하지 않으므로
이번 실습의 전이학습 조건에 해당하지 않는다.
"""


# 경로 관련 기능을 사용하기 위한 모듈
from pathlib import Path

# PyTorch 기본 기능
import torch

# 신경망 계층과 손실 함수를 사용하기 위한 모듈
from torch import nn

# 데이터를 배치 단위로 불러오기 위한 DataLoader
from torch.utils.data import DataLoader

# 이미지 폴더를 데이터셋으로 변환하는 ImageFolder
from torchvision import datasets

# ResNet18 모델과 ImageNet 사전학습 가중치
from torchvision.models import resnet18, ResNet18_Weights


# 난수 시드를 고정하여 실험 결과의 재현성을 높인다.
torch.manual_seed(42)

# CPU 연산에 사용할 스레드 수를 1개로 설정한다.
torch.set_num_threads(1)


# 현재 파이썬 파일이 위치한 폴더를 프로젝트 경로로 지정한다.
root = Path(__file__).resolve().parent


# ImageNet으로 사전학습된 ResNet18 가중치를 지정한다.
weights = ResNet18_Weights.IMAGENET1K_V1

# 지정한 가중치에 맞는 이미지 전처리 방법을 가져온다.
# 이미지 크기 조정, Crop, Normalize 등이 포함된다.
transform = weights.transforms()


# 학습용 이미지 데이터셋을 불러온다.
# ImageFolder는 하위 폴더 이름을 클래스 이름으로 사용한다.
train = datasets.ImageFolder(
    root / "data" / "train",
    transform=transform
)

# 검증용 이미지 데이터셋을 불러온다.
valid = datasets.ImageFolder(
    root / "data" / "valid",
    transform=transform
)


# 학습 데이터를 배치 단위로 불러올 DataLoader를 생성한다.
# batch_size=4: 한 번에 이미지 4장을 가져온다.
# shuffle=False: 데이터 순서를 섞지 않는다.
# num_workers=0: 별도의 데이터 로딩 프로세스를 사용하지 않는다.
loader = DataLoader(
    train,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# ImageNet 사전학습 가중치를 적용한 ResNet18 모델을 생성한다.
# weights=weights를 사용하므로 기존에 학습된 가중치를 가져온다.
model = resnet18(weights=weights)


# 해설 B06: 기존 Backbone의 가중치를 고정하고 새로운 Head를 학습한다.

# TODO BEGIN B06

# 모델에 포함된 모든 학습 파라미터를 하나씩 가져온다.
for parameter in model.parameters():

    # 기존 파라미터의 기울기 계산을 비활성화한다.
    # 기존 Backbone의 가중치가 학습 과정에서 변경되지 않도록 한다.
    parameter.requires_grad = False

# END B06


# 기존 ResNet18의 마지막 분류층(fc)을 새로운 분류층으로 교체한다.
# model.fc.in_features: 기존 분류층에 입력되는 특징의 개수
# 2: 새롭게 구분할 클래스의 개수
model.fc = nn.Linear(model.fc.in_features, 2)


# AdamW 최적화 알고리즘을 생성한다.
# model.fc.parameters(): 새로 만든 분류층의 파라미터만 학습 대상으로 지정한다.
# lr=0.001: 학습률을 0.001로 설정한다.
# weight_decay=0.0: 가중치 감쇠를 사용하지 않는다.
optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=0.001,
    weight_decay=0.0
)


# 분류 문제에 사용할 손실 함수를 생성한다.
loss_fn = nn.CrossEntropyLoss()


# eval()은 모델을 평가 모드로 전환한다.
# BatchNorm 등의 계층이 평가 방식으로 동작하도록 한다.
# 역전파를 비활성화하는 기능은 아니다.
# Backbone의 BatchNorm 통계도 학습 중 변경되지 않도록 한다.
model.eval()

# 새로 만든 Head는 학습 모드로 설정한다.
model.fc.train()


# 총 3번의 학습을 진행한다.
for epoch in range(3):

    # DataLoader에서 이미지와 정답을 배치 단위로 가져온다.
    for images, labels in loader:

        # 이전 배치에서 계산된 기울기를 초기화한다.
        optimizer.zero_grad()

        # 이미지를 모델에 전달하여 예측 결과를 얻는다.
        # 예측 결과와 실제 정답을 비교하여 손실을 계산한다.
        loss = loss_fn(model(images), labels)

        # 손실을 역전파하여 새 Head의 기울기를 계산한다.
        loss.backward()

        # 계산된 기울기를 이용하여 새 Head의 가중치를 갱신한다.
        optimizer.step()


# 학습이 끝난 모델을 평가 모드로 전환한다.
model.eval()


# 검증 과정에서는 기울기를 계산하지 않는다.
with torch.no_grad():

    # 검증 데이터 전체를 하나의 배치로 가져온다.
    images, labels = next(
        iter(DataLoader(valid, batch_size=len(valid)))
    )

    # 모델의 예측 결과와 실제 정답을 비교한다.
    # argmax(1): 각 이미지에서 가장 높은 점수를 가진 클래스 선택
    # == labels: 예측 클래스와 실제 정답이 같은지 비교
    # float(): True와 False를 숫자로 변환
    # mean(): 정답 비율의 평균 계산
    accuracy = (
        (model(images).argmax(1) == labels)
        .float()
        .mean()
        .item()
    )


# 실제 사용한 사전학습 가중치 이름을 출력한다.
print("실제 사전학습:", weights.name)

# 학습 가능한 파라미터의 개수를 출력한다.
# 기존 Backbone은 고정되어 있으므로 새 Head의 파라미터만 집계된다.
print(
    "학습 파라미터:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)

# 검증에 사용한 이미지 배치의 크기를 출력한다.
print("입력 크기:", tuple(images.shape))

# 검증 데이터의 정확도를 출력한다.
print("valid 정확도:", round(accuracy, 4))