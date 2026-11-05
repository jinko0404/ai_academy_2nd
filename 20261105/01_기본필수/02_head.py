"""
[Day14 - B02 기본 개념: Backbone 동결 및 Linear Head 교체]

1. 전이학습(Transfer Learning)
   - 다른 데이터로 학습한 가중치를 새로운 작업에 활용하는 방법
   - 이 파일은 별도 합성 이미지로 학습한 교육용 가중치를 사용함
   - Backbone은 이미지 특징을 추출하고 Head는 클래스 점수를 계산함

2. nn.Sequential
   - 여러 신경망 층을 등록한 순서대로 실행하는 컨테이너
   - model[인덱스]로 특정 층에 접근하거나 교체할 수 있음
   - 이 모델의 0~8번 층은 Backbone, 9번 층은 Head

3. CNN 구성 요소
   - nn.Conv2d(): 합성곱으로 이미지의 공간적 특징을 추출
   - nn.BatchNorm2d(): 채널별 값을 정규화
   - nn.ReLU(): 음수를 0으로 바꾸어 비선형성을 추가
   - nn.MaxPool2d(): 영역별 최댓값을 선택하여 공간 크기를 축소
   - nn.AdaptiveAvgPool2d(): 평균을 사용하여 지정한 공간 크기로 변환

4. Tensor와 shape
   - Tensor: 다차원 수치 데이터를 표현하는 자료구조
   - 이미지 배치 shape: (배치 크기, 채널 수, 높이, 너비)
   - nn.Flatten(): 배치 축을 유지하면서 나머지 차원을 펼침
   - 이 모델은 (N, 16, 4, 4)를 (N, 256)으로 변환함

5. nn.Linear(in_features, out_features, bias=True)
   - 입력 특징에 가중치를 곱하고 편향을 더하는 완전연결층
   - in_features: 입력 특징 수, out_features: 출력 점수 수
   - 입력·출력 특징 수는 필수이며 bias는 생략하면 True
   - nn.Linear(256, 3): 256개 특징에서 3개 클래스 점수를 계산

6. logit
   - 확률로 변환하기 전의 클래스별 점수
   - 3클래스 분류에서는 이미지마다 3개 logit이 필요함
   - logit 자체는 합이 1인 확률이 아님

7. parameters()와 requires_grad
   - parameters(): 모델에 등록된 파라미터를 순회
   - requires_grad: 기울기 계산 여부를 나타내는 속성으로 함수가 아님
   - False로 설정하면 해당 파라미터의 기울기를 계산하지 않음
   - 동결 이후 생성한 새 Linear의 파라미터는 기본적으로 True

8. state_dict()와 load_state_dict()
   - state_dict(): 가중치, 편향, BatchNorm 통계 등 모델 상태를 이름별로 반환
   - load_state_dict(): 준비한 상태를 모델에 적용
   - 딕셔너리의 update(): 같은 이름의 값을 덮어쓰고 새로운 이름은 추가

9. np.load()와 torch.from_numpy()
   - np.load(): NumPy 파일에서 배열을 읽는 함수
   - .npz: 여러 배열을 이름과 함께 저장할 수 있는 파일 형식
   - allow_pickle=False: pickle로 저장된 객체 배열의 로드를 허용하지 않음
   - copy(): 배열 복사, torch.from_numpy(): 배열을 Tensor로 변환
   - with 문: 사용을 마치면 열린 파일을 자동으로 닫음

10. eval()과 파라미터 동결
    - eval(): BatchNorm 등의 동작을 평가 모드로 전환
    - eval() 자체는 기울기 계산을 끄지 않음
    - 파라미터 동결과 평가 모드 전환은 별개의 설정
    - 동결해도 train 모드에서는 BatchNorm 이동 통계가 갱신될 수 있음

11. numel()과 sum()
    - numel(): Tensor의 전체 원소 수를 반환
    - sum(): 여러 값을 합산
    - 새 Head의 파라미터 수: 256×3개 가중치 + 3개 편향 = 771개

12. resnet18()과 fc
    - torchvision에서 ResNet-18을 생성하는 함수
    - weights=None: 사전 학습 가중치를 사용하지 않음
    - weights는 이름을 붙여 전달하는 키워드 인자
    - fc: 마지막 완전연결층, fc.in_features: 해당 층의 입력 특징 수
    - 이 파일의 ResNet은 구조 확인용이며 사전 학습 성능을 평가하지 않음

13. Path와 이미지 전처리
    - Path: 파일과 폴더 경로를 관리하는 클래스
    - resolve(): 절대 경로로 변환, parent: 상위 폴더 경로
    - tr.Compose(): 여러 변환을 순서대로 묶음
    - tr.Resize(): 이미지 크기 변경
    - tr.ToTensor(): 일반적인 8비트 이미지를 [0, 1] 범위 Tensor로 변환

14. 실행 설정과 재현성
    - torch.set_num_threads(1): CPU 연산 스레드 수를 1개로 설정
    - torch.manual_seed(14): PyTorch 난수 시드를 고정하여 재현성을 높임
    - 이 파일은 가중치 로드와 구조를 확인하며 실제 학습 반복문은 없음
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""

# 파일 및 폴더 경로를 관리하는 Path 클래스
from pathlib import Path

# JSON 처리 라이브러리이며 현재 파일에서는 사용하지 않음
import json

# 제공된 가중치 배열 파일을 읽는 수치 연산 라이브러리
import numpy as np

# 표 데이터 처리 라이브러리이며 현재 파일에서는 사용하지 않음
import pandas as pd

# Tensor 연산과 모델 실행을 담당하는 딥러닝 라이브러리
import torch

# 신경망 층과 모델 컨테이너를 제공하는 모듈
from torch import nn

# 폴더 기반 이미지 데이터셋 클래스이며 현재 파일에서는 사용하지 않음
from torchvision.datasets import ImageFolder

# 이미지 크기 변경 및 Tensor 변환을 위한 모듈
from torchvision import transforms as tr

# 분류 평가 함수들이며 현재 파일에서는 사용하지 않음
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Path는 경로 처리, numpy는 배열 파일 읽기, torch와 nn은 텐서와 신경망 구성에 사용합니다.
# transforms는 이미지 전처리를 제공합니다. json, pandas, ImageFolder와 평가 함수들은
# 현재 파일에서는 사용하지 않으며, 후속 데이터 처리·평가 코드에서 사용할 수 있습니다.

# CPU 연산 스레드 수를 1개로 설정합니다.
torch.set_num_threads(1)

# 가중치 초기화 등에 쓰이는 PyTorch 난수 시드를 고정합니다.
torch.manual_seed(14)

# 이 소스 파일이 위치한 폴더입니다.
BASE = Path(__file__).resolve().parent

# 현재 폴더 아래의 data 폴더를 가리킵니다.
DATA = BASE / "data"

# 출력 경로를 지정합니다. 현재 파일에는 저장 코드가 없습니다.
OUT = BASE


# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
# Compose는 변환을 순서대로 적용합니다. Resize는 이미지 크기를 32×32로 맞춥니다.
# ToTensor는 일반적인 8비트 이미지를 [0, 1] 범위의 (채널, 높이, 너비) 텐서로 만듭니다.
# 여기서는 변환을 정의하기만 하며, 실제 이미지에 적용하는 코드는 없습니다.
# 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
model = nn.Sequential(
    # Sequential은 층을 등록한 순서대로 실행합니다. N은 배치 크기입니다.
    # [0] RGB 3채널 → 8채널, (N, 8, 32, 32)
    nn.Conv2d(3, 8, 3, padding=1, bias=False),

    # [1] 8개 채널을 정규화합니다.
    nn.BatchNorm2d(8),

    # [2] 음수를 0으로 바꾸는 활성화 함수입니다.
    nn.ReLU(),

    # [3] 2×2 영역의 최댓값으로 축소합니다. (N, 8, 16, 16)
    nn.MaxPool2d(2),

    # [4] 8채널 → 16채널, (N, 16, 16, 16)
    nn.Conv2d(8, 16, 3, padding=1, bias=False),

    # [5] 16개 채널을 정규화합니다.
    nn.BatchNorm2d(16),

    # [6] 비선형성을 추가합니다.
    nn.ReLU(),

    # [7] 공간 크기를 4×4로 맞춥니다. (N, 16, 4, 4)
    nn.AdaptiveAvgPool2d((4, 4)),

    # [8] 배치 축을 유지하고 특징을 펼칩니다. (N, 256)
    nn.Flatten(),

    # [9] 256개 특징 → 3개 클래스 점수, (N, 3)
    nn.Linear(256, 3),

)
# 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
    # npz는 여러 배열을 담는 파일입니다. pickle 객체 로드를 막고, with 종료 시 파일을 닫습니다.
    # 현재 모델의 상태를 가져옵니다.
    state = model.state_dict()

    # 파일에 있는 이름의 상태를 덮어씁니다. 파일에 없는 상태는 기존 값을 유지합니다.
    # copy()로 NumPy 배열을 복사한 뒤 from_numpy()로 PyTorch 텐서로 변환합니다.
    state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
# 준비한 상태를 실제 모델에 적용합니다.
model.load_state_dict(state)


# 현재 모델의 모든 파라미터를 순회합니다.
for p in model.parameters():

    # 기존 Backbone과 Head의 파라미터를 동결합니다.
    p.requires_grad = False

# TODO B02: 256개 특징에서 3개 logit을 출력하는 새 Linear Head를 만드세요.
# BEGIN B02
# 9번 층을 3출력 Head로 교체합니다. bias=True는 기본값입니다.
model[9] = nn.Linear(256, 3)

# 새 Head는 동결 이후 생성했으므로 가중치와 편향의 requires_grad가 True입니다.
# END B02
# 가중치의 출처를 표시합니다.
print("source 종류: 별도 합성 이미지로 사전학습한 TinyShapeCNN")

# 256×3 + 3 = 771개
print("학습 파라미터:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# numel()은 텐서의 원소 수입니다. requires_grad=True인 파라미터만 합산합니다.
# 예상 출력: (2, 3)
print("출력 shape:", tuple(model.eval()(torch.zeros(2, 3, 32, 32)).shape))

# zeros로 가상 RGB 이미지 2장을 만들고 평가 모드에서 모델에 통과시킵니다.
# PPT ResNet18의 구조도 다운로드 없이 확인합니다. 이것은 '전이학습 성능'이 아닙니다.
# ResNet-18 생성 함수를 가져옵니다.
from torchvision.models import resnet18

# 사전 학습 가중치 없이 생성합니다. 다운로드하지 않습니다.
resnet = resnet18(weights=None)

# 기존 ResNet의 모든 파라미터를 순회합니다.
for p in resnet.parameters():

    # 기존 파라미터를 동결합니다.
    p.requires_grad = False

# 기존 입력 특징 수(512)를 유지하고 새 Head를 연결합니다.
resnet.fc = nn.Linear(resnet.fc.in_features, 3)

# 512×3 + 3 = 1,539개
print("ResNet18 구조 확인용 Head 파라미터:", sum(p.numel() for p in resnet.parameters() if p.requires_grad))

print("주의: 위 ResNet18은 구조만 확인했습니다. ImageNet 학습 결과가 아닙니다.")
