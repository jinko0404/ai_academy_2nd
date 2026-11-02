"""
실습 A06 · Head만 학습하는 방식과 Backbone 일부까지 학습하는 방식 비교

[기본 개념 정리]

1. 전이학습(Transfer Learning)
   - 기존에 학습된 모델의 지식을 새로운 문제에 재사용하는 방법.
   - 이번 실습에서는 ImageNet으로 사전학습된 ResNet18을 사용한다.

2. ResNet18
   - 이미지 분류에 사용하는 CNN 모델.
   - 여러 합성곱 계층을 통해 이미지의 특징을 추출한다.

3. Backbone
   - 이미지의 특징을 추출하는 모델의 앞부분.
   - 이번 실습에서는 ResNet18의 layer1, layer2, layer3, layer4가 해당한다.

4. Head
   - Backbone이 추출한 특징을 이용해 최종 분류를 수행하는 부분.
   - 이번 실습에서는 model.fc가 Head에 해당한다.

5. Feature Extraction
   - 기존 Backbone의 가중치를 고정하고 새로운 Head만 학습하는 방법.
   - 기존 모델이 학습한 특징을 그대로 활용한다.
   - 이번 실습의 mode="head"에 해당한다.

6. Fine-tuning
   - 기존 모델의 일부 또는 전체 가중치도 새로운 데이터에 맞게 추가 학습하는 방법.
   - 이번 실습의 mode="partial"은 layer4까지 학습한다.

7. requires_grad
   - 해당 파라미터의 기울기를 계산할지 결정하는 속성.
   - False: 학습 대상에서 제외한다.
   - True: 기울기를 계산하고 학습할 수 있다.

8. Optimizer
   - 손실 함수의 역전파로 계산한 기울기를 이용해 모델의 가중치를 갱신한다.
   - 이번 실습에서는 AdamW를 사용한다.

9. Learning Rate(학습률)
   - 가중치를 얼마나 크게 변경할지 결정하는 값.
   - 학습률이 크면 가중치가 크게 변하고, 작으면 조금씩 변한다.

10. Parameter Groups
    - 하나의 Optimizer에 서로 다른 파라미터 그룹을 등록하는 방법.
    - 각 그룹에 서로 다른 학습률을 지정할 수 있다.
    - 이번 실습에서는 Head와 layer4에 다른 학습률을 적용한다.

11. BatchNorm
    - 신경망의 학습을 안정적으로 돕는 정규화 계층.
    - train()에서는 학습용 통계를 갱신하고,
      eval()에서는 저장된 통계를 사용한다.
    - 이번 실습에서는 layer4를 학습하더라도 BatchNorm 통계는 고정한다.

12. DataLoader
    - Dataset에서 데이터를 가져와 배치 단위로 제공한다.
    - batch_size=4이면 한 번에 이미지 4장을 가져온다.

13. Epoch
    - 전체 학습 데이터를 한 번 사용하는 단위.
    - 이번 실습은 학습 경로 확인을 위해 1 Epoch만 진행한다.

14. torch.no_grad()
    - 기울기 계산을 비활성화하는 기능.
    - 검증 과정에서 불필요한 기울기 계산을 방지한다.

15. CrossEntropyLoss
    - 분류 모델의 예측 결과와 실제 정답 사이의 차이를 계산하는 손실 함수.

16. valid_loss와 valid_accuracy
    - valid_loss: 검증 데이터에 대한 손실값.
    - valid_accuracy: 검증 데이터에서 정답을 맞힌 비율.

17. time.perf_counter()
    - 코드 실행에 걸린 시간을 측정하는 기능.
    - 이번 실습에서는 학습 소요 시간을 측정한다.

18. CSV
    - 표 형태의 데이터를 저장하는 파일 형식.
    - 이번 실습에서는 두 가지 학습 방식의 결과를 저장한다.

[두 가지 학습 방식]

1. head
   - Backbone 전체 고정
   - 새로운 Head만 학습
   - Head 학습률: 0.001

2. partial
   - Backbone의 layer1~layer3 고정
   - layer4와 새로운 Head 학습
   - Head 학습률: 0.001
   - layer4 학습률: 0.0001

주의:
- 두 방식 모두 ImageNet 사전학습 가중치를 사용한다.
- 1 Epoch만으로 일반화 성능의 우열을 단정할 수 없다.
"""

# 파일 및 폴더 경로를 다루기 위한 모듈
from pathlib import Path

# 모델의 가중치 등을 복사할 때 사용하는 모듈
import copy

# CSV 파일 저장 및 처리에 사용하는 모듈
import csv

# JSON 데이터 처리를 위한 모듈
import json

# 실행 시간 측정을 위한 모듈
import time

# PyTorch 기본 기능
import torch

# 신경망 계층과 손실 함수 사용
from torch import nn

# 데이터를 배치 단위로 불러오기 위한 DataLoader
from torch.utils.data import DataLoader

# 이미지 데이터셋과 전처리 기능
from torchvision import datasets, transforms

# 표 형태의 데이터 처리 및 CSV 저장
import pandas as pd


# 난수 시드를 고정하여 실험 결과의 재현성을 높인다.
torch.manual_seed(42)

# CPU 연산에 사용할 스레드 수를 1개로 설정한다.
torch.set_num_threads(1)

# 현재 파이썬 파일이 위치한 폴더를 프로젝트 경로로 지정한다.
root = Path(__file__).resolve().parent

# 실험 결과를 저장할 outputs 폴더 경로를 지정한다.
out = root / "outputs"

# outputs 폴더가 없으면 생성한다.
# exist_ok=True: 폴더가 이미 있어도 오류가 발생하지 않는다.
out.mkdir(exist_ok=True)


# ResNet18 모델과 ImageNet 사전학습 가중치를 가져온다.
from torchvision.models import resnet18, ResNet18_Weights


# ImageNet으로 사전학습된 ResNet18 가중치를 지정한다.
weights = ResNet18_Weights.IMAGENET1K_V1


# 사전학습 가중치에 맞는 이미지 전처리 방법을 가져온다.
# 이미지 크기 조정, 자르기, 정규화 등이 포함된다.
transform = weights.transforms()

# 학습용 이미지 데이터셋을 불러온다.
train = datasets.ImageFolder(
    root / "data" / "train",
    transform=transform
)

# 검증용 이미지 데이터셋을 불러온다.
valid = datasets.ImageFolder(
    root / "data" / "valid",
    transform=transform
)

rows = []   # 두 가지 학습 방식의 결과를 저장할 빈 리스트를 생성한다.

for mode in ["head", "partial"]:# head 방식과 partial 방식을 차례대로 실험한다.
    torch.manual_seed(42)       # 각 실험을 동일한 난수 시드로 시작한다.

    # ImageNet 사전학습 가중치를 적용한 ResNet18 모델을 생성한다.
    model = resnet18(weights=weights)

    # 모델의 모든 파라미터를 가져온다.
    for param in model.parameters():

        # 기존 모델의 모든 가중치를 우선 고정한다.
        param.requires_grad = False

    # 기존 ResNet18의 마지막 분류층을 새로운 분류층으로 교체한다.
    # 기존 분류층의 입력 특징 개수는 유지하고 출력 클래스는 2개로 설정한다.
    model.fc = nn.Linear(model.fc.in_features, 2)


    # Optimizer에 등록할 파라미터 그룹을 생성한다.
    # 새로 만든 Head의 학습률은 0.001로 설정한다.
    groups = [
        {
            "params": model.fc.parameters(),
            "lr": 0.001
        }
    ]


    # partial 방식일 때만 Backbone의 layer4를 추가 학습한다.
    if mode == "partial":

        # TODO BEGIN A06

        # layer4에 포함된 파라미터를 가져온다.
        for param in model.layer4.parameters():

            # layer4의 기울기 계산을 활성화한다.
            # 따라서 layer4도 학습 과정에서 가중치가 갱신된다.
            param.requires_grad = True

        # END A06

        # layer4의 파라미터를 Optimizer에 추가한다.
        # 기존 가중치를 크게 바꾸지 않도록 학습률을 0.0001로 설정한다.
        groups.append(
            {
                "params": model.layer4.parameters(),
                "lr": 0.0001
            }
        )


    # AdamW 최적화 알고리즘을 생성한다.
    # groups에 등록된 Head와 layer4만 학습 대상으로 사용한다.
    # weight_decay=0.0001: 가중치 감쇠를 적용한다.
    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=0.0001
    )


    # 학습 데이터를 배치 단위로 불러올 DataLoader를 생성한다.
    # batch_size=4: 한 번에 이미지 4장을 가져온다.
    # shuffle=True: 학습 데이터 순서를 섞는다.
    # num_workers=0: 별도 데이터 로딩 프로세스를 사용하지 않는다.
    # generator: 데이터 순서를 섞을 때 사용할 난수 시드를 고정한다.
    train_loader = DataLoader(
        train,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(42)
    )


    # 모델 전체를 평가 모드로 설정한다.
    # 특히 Backbone의 BatchNorm 통계가 갱신되지 않도록 한다.
    model.eval()

    # 새로 만든 Head는 학습 모드로 설정한다.
    model.fc.train()


    # partial 방식일 때만 layer4를 학습 모드로 전환한다.
    if mode == "partial":

        # layer4의 일반적인 계층은 학습 모드로 설정한다.
        model.layer4.train()

        # layer4 내부의 모든 하위 계층을 확인한다.
        for layer in model.layer4.modules():

            # 현재 계층이 BatchNorm2d인지 확인한다.
            if isinstance(layer, nn.BatchNorm2d):

                # BatchNorm의 통계는 갱신하지 않고 평가 모드로 유지한다.
                layer.eval()


    # 학습 시작 시간을 기록한다.
    start = time.perf_counter()


    # 학습 데이터를 배치 단위로 가져온다.
    for images, labels in train_loader:

        # 이전 배치에서 계산된 기울기를 초기화한다.
        optimizer.zero_grad()

        # 모델에 이미지를 전달하고 예측 결과를 얻는다.
        # 예측 결과와 실제 정답을 비교하여 손실을 계산한다.
        # backward()를 통해 손실을 역전파한다.
        nn.functional.cross_entropy(
            model(images),
            labels
        ).backward()

        # 계산된 기울기를 이용하여 학습 대상 가중치를 갱신한다.
        optimizer.step()


    # 학습 종료 시간을 기록하고 걸린 시간을 계산한다.
    training_seconds = time.perf_counter() - start


    # 학습이 끝난 모델을 평가 모드로 전환한다.
    model.eval()

    # 검증 손실, 정답 수, 전체 데이터 수를 저장할 변수를 초기화한다.
    total_loss, correct, count = 0.0, 0, 0


    # 검증 과정에서는 기울기를 계산하지 않는다.
    with torch.no_grad():

        # 검증 데이터를 배치 크기 4로 가져온다.
        for images, labels in DataLoader(valid, batch_size=4):

            # 모델에 검증 이미지를 전달하여 예측 점수를 얻는다.
            logits = model(images)

            # 배치의 평균 손실에 이미지 개수를 곱해 누적한다.
            # 이렇게 하면 마지막 배치 크기가 달라도 정확한 평균을 계산할 수 있다.
            total_loss += (
                nn.functional.cross_entropy(logits, labels).item()
                * len(labels)
            )

            # 각 이미지에서 가장 높은 점수를 가진 클래스를 선택한다.
            # 실제 정답과 비교하여 맞힌 개수를 누적한다.
            correct += (
                (logits.argmax(1) == labels).sum().item()
            )

            # 지금까지 확인한 검증 이미지 개수를 누적한다.
            count += len(labels)


    # 현재 학습 방식의 결과를 딕셔너리로 정리한다.
    rows.append(
        dict(
            mode=mode,  # 학습 방식: head 또는 partial
            pretrained=weights.name,  # 사용한 사전학습 가중치 이름
            image_size=224,  # ResNet18의 입력 이미지 크기
            epochs=1,  # 학습 횟수
            # 학습 가능한 파라미터의 전체 개수를 계산한다.
            trainable_params=sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            ),
            # 전체 검증 손실의 평균을 계산한다.
            valid_loss=total_loss / count,

            # 검증 데이터의 정확도를 계산한다.
            valid_accuracy=correct / count,

            # 학습에 걸린 시간을 기록한다.
            training_seconds=training_seconds
        )
    )


# 두 가지 학습 방식의 결과를 표 형태로 출력한다.
# training_seconds는 출력 표에서 제외한다.
# round(4): 소수점 넷째 자리까지 반올림한다.
print(
    pd.DataFrame(rows)
    .drop(columns="training_seconds")
    .round(4)
    .to_string(index=False)
)


# 두 가지 학습 방식의 실험 결과를 CSV 파일로 저장한다.
pd.DataFrame(rows).to_csv(
    out / "transfer_comparison.csv",
    index=False
)