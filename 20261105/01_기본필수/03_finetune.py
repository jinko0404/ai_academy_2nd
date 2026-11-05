"""
[Day14 - B03 기본 개념: Head 학습과 Partial Fine-tuning 비교]

1. 전이학습과 Fine-tuning
   - 다른 데이터로 학습한 가중치를 새로운 작업에 활용하는 방법
   - Fine-tuning: 사전학습한 파라미터의 일부 또는 전체를 새 데이터로 조정
   - 이 파일은 별도 합성 이미지로 학습한 소형 source 가중치를 사용함
   - ImageNet으로 사전학습한 공식 ResNet18을 학습하는 코드가 아님

2. Backbone과 Head
   - Backbone: 이미지의 특징을 추출하는 부분으로 model[0]~model[8]
   - Head: 추출된 특징으로 클래스별 점수를 계산하는 model[9]
   - head 전략: Head만 학습하며 Backbone 파라미터는 고정
   - partial 전략: 두 번째 Conv인 model[4]와 Head를 학습
   - 이 기본필수 파일에서는 BatchNorm의 가중치와 편향도 고정함

3. ImageFolder와 전처리
   - 폴더 이름을 기준으로 클래스와 숫자 라벨을 생성하는 이미지 데이터셋
   - dataset[i]: (변환된 이미지, 정답 라벨)을 반환
   - dataset.targets: 이미지 순서에 대응하는 정답 라벨 목록
   - tr.Compose(): 여러 변환을 순서대로 적용
   - Resize((32, 32))와 ToTensor(): RGB 이미지를 [0, 1] 범위 Tensor로 변환

4. Tensor와 torch.stack()
   - Tensor: 다차원 수치 데이터를 표현하는 자료구조
   - torch.stack(): 같은 크기의 Tensor들을 새로운 차원으로 쌓음
   - 이미지 전체를 (이미지 수, 3, 32, 32) 형태의 배치로 구성
   - torch.long: CrossEntropyLoss의 클래스 번호 정답에 사용하는 정수 자료형
   - 이 코드는 전체 train 데이터를 한 번에 처리하는 full-batch 학습을 수행함

5. nn.Sequential과 CNN
   - nn.Sequential(): 등록한 층을 순서대로 실행하는 컨테이너
   - Conv2d: 공간적 특징 추출, BatchNorm2d: 채널별 정규화
   - ReLU: 비선형성 추가, MaxPool2d: 공간 크기 축소
   - AdaptiveAvgPool2d: 공간 크기를 지정된 4×4로 변환
   - Flatten: (N, 16, 4, 4)를 (N, 256)으로 펼침
   - Linear(256, 3): 256개 특징에서 3개 logit을 계산

6. state_dict와 가중치 복원
   - state_dict(): 파라미터와 BatchNorm 통계 등 모델 상태를 이름별로 반환
   - np.load(): 제공된 .npz 배열 묶음을 읽음
   - allow_pickle=False: pickle 객체 배열 로드를 허용하지 않음
   - torch.from_numpy(): NumPy 배열을 Tensor로 변환
   - load_state_dict(): 준비한 상태를 모델에 적용

7. requires_grad와 파라미터 동결
   - parameters(): 모델 또는 특정 층의 파라미터를 순회
   - requires_grad: 기울기 계산 여부를 나타내는 속성
   - False로 전체를 동결한 뒤 학습할 층만 True로 변경
   - head의 학습 파라미터: 256×3 + 3 = 771개
   - partial은 Conv 가중치 16×8×3×3 = 1,152개를 더해 총 1,923개

8. eval()과 BatchNorm 통계
   - eval(): BatchNorm 등의 동작을 평가 모드로 전환
   - 이 코드에서는 학습 중에도 BatchNorm의 이동 평균·분산을 고정하기 위해 사용
   - eval()은 기울기 계산을 끄지 않으므로 backward()와 학습이 가능함
   - requires_grad=False와 eval()은 서로 다른 설정

9. AdamW와 학습률
   - optimizer: 기울기를 이용하여 파라미터를 갱신하는 도구
   - AdamW(params, lr, weight_decay): 학습할 파라미터와 갱신 설정을 지정
   - lr=0.003: 학습률, weight_decay=0.0: 가중치 감쇠를 적용하지 않음
   - 전체 파라미터를 전달해도 이 코드의 동결 파라미터는 기울기가 없어 갱신되지 않음

10. CrossEntropyLoss와 학습 순서
    - CrossEntropyLoss: 클래스별 logit과 정답 클래스 번호로 분류 손실을 계산
    - 입력에 별도로 Softmax를 적용할 필요가 없음
    - zero_grad(): 이전 기울기 초기화
    - backward(): 손실의 기울기 계산
    - step(): 계산한 기울기로 파라미터 갱신
    - epoch: 학습 데이터 전체를 한 번 사용하는 단위이며 여기서는 8회 반복

11. 검증과 accuracy_score()
    - torch.no_grad(): 검증 시 기울기를 기록하지 않아 연산·메모리 부담을 줄임
    - argmax(1): 각 이미지에서 가장 큰 logit의 클래스 번호를 선택
    - accuracy_score(): 전체 예측 중 정답 비율을 계산
    - valid로 두 전략을 비교하며 test는 읽지 않음

12. 재현성과 결과 표
    - 각 전략 시작 시 같은 난수 시드를 설정하여 같은 초기값으로 비교
    - 데이터, source 가중치, 학습 횟수, 학습률 등 비교 조건을 동일하게 유지
    - numel(): 파라미터 원소 수, sum(): 원소 수 합산
    - pd.DataFrame(): 결과를 표로 변환
    - round(4): 소수점 넷째 자리까지 반올림
    - to_string(index=False): 행 번호 없이 표를 문자열로 출력
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""

# 파일 및 폴더 경로를 관리하는 클래스
from pathlib import Path

# JSON 처리 라이브러리이며 현재 파일에서는 사용하지 않음
import json

# 제공된 가중치 배열 파일을 읽는 수치 연산 라이브러리
import numpy as np

# 전략별 결과를 표로 정리하는 라이브러리
import pandas as pd

# Tensor 연산과 자동 미분을 지원하는 딥러닝 라이브러리
import torch

# 신경망 층과 손실 함수를 제공하는 모듈
from torch import nn

# 폴더 이름을 기준으로 이미지와 정답 라벨을 불러오는 클래스
from torchvision.datasets import ImageFolder

# 이미지 크기 변경과 Tensor 변환을 위한 모듈
from torchvision import transforms as tr

# 분류 평가 함수들 중 이 파일에서는 accuracy_score만 사용
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# CPU 연산 스레드 수를 1개로 설정
torch.set_num_threads(1)

# PyTorch 난수 시드를 고정하여 재현성을 높임
torch.manual_seed(14)

# 현재 파이썬 파일이 위치한 폴더 경로
BASE = Path(__file__).resolve().parent

# 원본 데이터와 source 가중치가 저장된 폴더 경로
DATA = BASE / "data"

# 출력 경로를 현재 폴더로 지정함. 이 파일에서는 결과를 화면에만 출력
OUT = BASE

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# 이미지를 32×32로 맞추고 [0, 1] 범위의 (3, 32, 32) Tensor로 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# 학습 이미지와 검증 이미지를 같은 전처리로 불러옴
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)

# 각 항목의 이미지인 [0]을 가져와 전체 학습 이미지 배치로 쌓음
X_train = torch.stack([train[i][0] for i in range(len(train))])

# 학습 정답 클래스 번호를 정수 Tensor로 변환
y_train = torch.tensor(train.targets, dtype=torch.long)

# 검증 이미지도 데이터셋 순서대로 하나의 배치로 쌓음
X_valid = torch.stack([valid[i][0] for i in range(len(valid))])

# 검증 이미지와 같은 순서로 정답 Tensor를 생성
y_valid = torch.tensor(valid.targets, dtype=torch.long)

# 전략 이름, 학습 파라미터 수, 검증 정확도를 저장할 리스트
results = []

# Head만 학습하는 전략과 일부 Backbone까지 학습하는 전략을 순서대로 실행
for strategy in ["head", "partial"]:

    # 두 전략이 같은 초기값에서 시작하도록 모델 생성 전에 시드를 다시 고정
    torch.manual_seed(14)

    # 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
    model = nn.Sequential(
        # [0] 3채널 → 8채널 합성곱. padding=1로 공간 크기 32×32 유지
        nn.Conv2d(3, 8, 3, padding=1, bias=False),

        # [1] 8개 채널을 정규화
        nn.BatchNorm2d(8),

        # [2] 음수를 0으로 바꾸는 활성화 함수
        nn.ReLU(),

        # [3] 공간 크기를 절반으로 축소하여 (N, 8, 16, 16) 생성
        nn.MaxPool2d(2),

        # [4] 두 번째 Conv. 8채널 → 16채널이며 partial에서만 학습
        nn.Conv2d(8, 16, 3, padding=1, bias=False),

        # [5] 16개 채널을 정규화하며 두 전략 모두 파라미터와 이동 통계 고정
        nn.BatchNorm2d(16),

        # [6] 비선형 활성화 적용
        nn.ReLU(),

        # [7] 공간 크기를 4×4로 맞추어 (N, 16, 4, 4) 생성
        nn.AdaptiveAvgPool2d((4, 4)),

        # [8] 이미지마다 16×4×4 = 256개 특징으로 펼침
        nn.Flatten(),

        # [9] 두 전략 모두 학습하는 3클래스 분류 Head
        nn.Linear(256, 3),
    )

    # 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
    with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:

        # 현재 모델의 상태를 가져와 파일에 없는 항목은 기존 값을 유지
        state = model.state_dict()

        # 파일의 배열을 복사하여 Tensor로 바꾼 뒤 같은 이름의 모델 상태를 덮어씀
        state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})

    # 준비한 source 상태를 모델에 적용
    model.load_state_dict(state)

    # 우선 모델의 모든 파라미터를 동결
    for p in model.parameters():
        p.requires_grad = False

    # TODO B03: 두 전략에서 Head(9)를 열고, partial일 때 두 번째 Conv(4)도 여세요.
    # BEGIN B03
    # 두 전략 모두 Head의 가중치와 편향을 학습하도록 설정
    for p in model[9].parameters():
        p.requires_grad = True

    # partial 전략일 때만 두 번째 Conv의 가중치도 학습하도록 설정
    if strategy == "partial":
        for p in model[4].parameters():
            p.requires_grad = True
    # END B03

    # BatchNorm의 이동 통계를 고정함. eval 상태에서도 학습 파라미터의 미분은 가능
    model.eval()

    # 기울기가 계산되는 파라미터를 AdamW로 갱신하며 가중치 감쇠는 적용하지 않음
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)

    # 전체 학습 데이터를 사용한 갱신을 8회 수행
    for epoch in range(8):

        # 이전 반복에서 계산된 기울기를 초기화
        optimizer.zero_grad()

        # (N, 3) logit과 (N,) 정답 클래스 번호로 분류 손실 계산
        loss = nn.CrossEntropyLoss()(model(X_train), y_train)

        # 손실을 역전파하여 학습 가능한 파라미터의 기울기 계산
        loss.backward()

        # 계산한 기울기로 파라미터를 실제 갱신
        optimizer.step()

    # 검증에서는 기울기 기록 없이 예측만 수행
    with torch.no_grad():

        # 클래스 축인 1번 차원에서 가장 높은 점수의 클래스 번호 선택
        pred = model(X_valid).argmax(1)

    # 현재 전략의 학습 파라미터 수와 마지막 epoch의 검증 정확도를 저장
    results.append([strategy, sum(p.numel() for p in model.parameters() if p.requires_grad), accuracy_score(y_valid, pred)])

# 두 전략의 결과를 표로 만들고 소수점 넷째 자리까지 반올림하여 행 번호 없이 출력
print(pd.DataFrame(results, columns=["strategy", "trainable", "valid_accuracy"]).round(4).to_string(index=False))
