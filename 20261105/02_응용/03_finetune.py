"""
[Day14 - A03 기본 개념: Head-only와 Partial Fine-tuning 비교]

1. 전이학습과 학습 전략
   - 다른 데이터로 학습한 가중치를 새 작업에 활용하는 방법
   - source는 별도 합성 이미지로 학습한 소형 CNN이며 ImageNet ResNet18이 아님
   - head: Head(9)만 학습, partial: 두 번째 Conv(4)와 Head(9)를 학습
   - 이 파일의 실제 구현은 BatchNorm의 파라미터를 두 전략 모두 동결함

2. experiments와 튜플 언패킹
   - 리스트에 (학습 전략, 불균형 대응 방식) 튜플을 실험별로 저장
   - [("head", "none"), ("partial", "none")]: 두 실험을 지정
   - for strategy, balance in experiments: 튜플의 두 값을 각각 변수에 대입
   - none은 클래스 가중치를 사용하지 않는 조건이며 학습 전략만 바꾸어 비교함

3. ImageFolder와 전처리
   - 하위 폴더 이름을 기준으로 이미지의 클래스와 숫자 라벨을 생성
   - dataset[i]: (이미지, 정답 라벨), dataset.targets: 정답 라벨 목록
   - Compose: 변환을 순서대로 적용, Resize: 32×32로 크기 변경
   - ToTensor: 일반적인 8비트 RGB 이미지를 [0, 1] 범위 Tensor로 변환
   - train과 valid의 클래스 이름·번호 매핑은 일치해야 함

4. Tensor와 배치
   - torch.stack(): 이미지 Tensor들을 새로운 배치 차원으로 쌓음
   - 이미지 shape: (이미지 수, 3, 32, 32), 정답 shape: (이미지 수,)
   - torch.long: 클래스 번호를 표현하는 정수 자료형
   - 전체 train 데이터를 한 번에 사용하는 full-batch 학습을 수행함

5. nn.Sequential과 CNN
   - Sequential: 등록한 층을 순서대로 실행하는 컨테이너
   - Conv2d: 특징 추출, BatchNorm2d: 채널별 정규화, ReLU: 비선형성 추가
   - MaxPool2d: 공간 크기 축소, AdaptiveAvgPool2d: 지정한 공간 크기로 변환
   - Flatten: (N, 16, 4, 4)를 (N, 256)으로 펼침
   - Linear(256, 3): 256개 특징으로 확률 변환 전 점수인 logit 3개를 계산

6. 가중치 복원과 state_dict()
   - np.load(): 제공된 .npz 배열 묶음을 읽음
   - allow_pickle=False: pickle 객체 배열의 로드를 허용하지 않음
   - state_dict(): 파라미터와 BatchNorm 통계 등 모델 상태를 이름별로 반환
   - torch.from_numpy(): NumPy 배열을 Tensor로 변환
   - load_state_dict(): 준비한 상태를 모델에 적용

7. requires_grad와 eval()
   - requires_grad: 파라미터의 기울기 계산 여부를 나타내는 속성
   - 전체를 False로 동결한 뒤 학습할 층만 True로 설정
   - eval(): 평가 모드로 전환하여 BatchNorm의 이동 통계를 고정
   - eval() 자체는 기울기 계산을 끄지 않으므로 이 상태에서도 학습 가능
   - head는 771개, partial은 Conv의 1,152개를 더한 1,923개 파라미터를 학습

8. AdamW와 학습 과정
   - 학습 가능한 파라미터만 optimizer에 전달
   - lr=0.003: 학습률, weight_decay=0.0001: 가중치 감쇠 설정
   - zero_grad(): 기울기 초기화, backward(): 기울기 계산, step(): 파라미터 갱신
   - range(1, 9): epoch 1부터 8까지 반복

9. 클래스 가중치와 CrossEntropyLoss
   - bincount(): 클래스 번호별 개수 계산, minlength=3: 최소 3개 항목 확보
   - 클래스 개수의 역수로 희소 클래스에 더 큰 가중치를 부여
   - inverse / inverse.mean(): 가중치의 평균이 1이 되도록 조정
   - CrossEntropyLoss: logit과 정답 클래스 번호로 분류 손실 계산
   - 현재 실험의 balance는 모두 none이므로 계산한 클래스 가중치를 적용하지 않음

10. 검증 손실과 최적 epoch 선택
    - no_grad(): 검증 과정에서 기울기를 기록하지 않음
    - item(): 원소 하나인 Tensor를 Python 숫자로 변환
    - valid 손실이 가장 낮은 epoch의 모델 상태를 보관
    - detach(): 미분 그래프에서 분리, clone(): 값을 별도 복사
    - 학습 후 최적 상태를 복원하여 지표를 계산하며 마지막 epoch와 다를 수 있음
    - valid로 모델을 선택하고 test는 사용하지 않음

11. Softmax와 평가 지표
    - softmax(1): 클래스 축의 logit을 합이 1인 확률로 변환
    - argmax(1): 가장 높은 확률의 클래스 번호 선택
    - accuracy_score(): 전체 예측 중 정답 비율
    - macro F1: 클래스별 F1을 동일 가중치로 평균
    - Recall: 실제 해당 클래스 중 올바르게 예측한 비율
    - classification_report(output_dict=True): 지표를 딕셔너리로 반환
    - labels=[0,1,2]: 클래스 순서 고정, zero_division=0: 분모가 0인 지표는 0 처리
    - report['2']['recall']은 클래스 번호 2가 triangle이라는 매핑을 전제로 함

12. 기록과 재현성
    - 각 실험 시작 시 같은 seed를 설정하여 모델 초기값을 맞춤
    - 데이터, 전처리, source, epoch, 학습률 등을 고정하여 비교
    - perf_counter(): 경과 시간 측정에 사용하는 시계이며 시간은 기기별로 다름
    - DataFrame: 결과 표, to_csv(index=False): 행 번호 없이 CSV 저장
    - all_history는 epoch별 검증 손실을 메모리에 보관하며 이 파일에서 저장하지 않음
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""

# 파일 및 폴더 경로 관리
from pathlib import Path

# JSON 처리 도구이며 현재 파일에서는 사용하지 않음
import json

# source 가중치 배열 파일 읽기
import numpy as np

# 결과 표 작성과 CSV 저장
import pandas as pd

# Tensor 연산과 자동 미분
import torch

# 신경망 층과 손실 함수
from torch import nn

# 폴더 구조로 이미지와 정답 라벨 생성
from torchvision.datasets import ImageFolder

# 이미지 전처리 모듈
from torchvision import transforms as tr

# 분류 평가 함수들. confusion_matrix는 현재 파일에서 사용하지 않음
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# CPU 연산 스레드를 1개로 설정
torch.set_num_threads(1)

# PyTorch 난수 시드 고정
torch.manual_seed(14)

# 현재 파일의 폴더를 기준 경로로 사용
BASE = Path(__file__).resolve().parent

# 원본 데이터와 source 가중치 폴더
DATA = BASE / "data"

# 결과 CSV를 저장할 폴더
OUT = BASE / "outputs"

# outputs 폴더가 없으면 생성
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# 소형 source 학습 당시 입력 규칙에 맞춰 크기와 픽셀 범위를 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# 학습 구간의 경과 시간 측정
import time

# 학습용과 검증용 이미지에 동일한 전처리 적용
train = ImageFolder(DATA / "train", transform=transform)
valid = ImageFolder(DATA / "valid", transform=transform)

# 각 항목의 이미지인 [0]을 가져와 전체 학습 배치로 쌓음
X_train = torch.stack([train[i][0] for i in range(len(train))])

# 학습 정답을 클래스 번호 정수 Tensor로 변환
y_train = torch.tensor(train.targets, dtype=torch.long)

# 검증 이미지를 데이터셋 순서대로 배치로 쌓음
X_valid = torch.stack([valid[i][0] for i in range(len(valid))])

# 검증 이미지 순서에 대응하는 정답 Tensor 생성
y_valid = torch.tensor(valid.targets, dtype=torch.long)

# TODO A03: Head-only, Partial 두 전략만 비교하세요. 데이터와 epoch는 고정합니다.
# BEGIN A03
# 클래스 가중치 사용 여부는 none으로 고정하고 학습 전략만 두 가지로 비교
experiments = [("head", "none"), ("partial", "none")]
# END A03

# 실험별 최종 지표를 저장할 리스트
rows = []

# 모든 실험의 epoch별 검증 손실을 저장할 리스트
all_history = []

# 각 튜플에서 전략과 불균형 대응 방식을 꺼내 실험 실행
for strategy, balance in experiments:

    # 모델 생성 전에 시드를 다시 고정하여 실험별 초기값을 동일하게 설정
    torch.manual_seed(14)
    # 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
    model = nn.Sequential(
        # [0~3] 3채널 → 8채널 특징 추출 후 공간 크기를 32×32에서 16×16으로 축소
        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        # [4~6] 8채널 → 16채널 특징 추출. partial에서는 Conv(4)만 추가 학습
        nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
        # [7~9] 4×4 평균 풀링 → 256개 특징으로 펼침 → 클래스 점수 3개 출력
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 3),
    )
    # 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
    with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
        # 현재 상태를 가져와 파일에 없는 항목은 초기값 유지
        state = model.state_dict()
        # 저장된 배열을 복사하고 Tensor로 변환하여 같은 이름의 상태를 덮어씀
        state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
    # source 상태를 실제 모델에 적용
    model.load_state_dict(state)

    # 전체 파라미터를 먼저 동결
    for p in model.parameters():
        p.requires_grad = False

    # 두 전략 모두 Head의 가중치와 편향 학습 허용
    for p in model[9].parameters():
        p.requires_grad = True

    # partial 전략에서만 두 번째 Conv의 가중치 학습 허용
    if strategy == "partial":
        for p in model[4].parameters():
            p.requires_grad = True

    # 학습 중에도 BatchNorm의 이동 통계를 고정하며 기울기 계산은 유지
    model.eval()

    # 학습 가능한 파라미터만 AdamW에 등록
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003, weight_decay=0.0001)

    # train의 클래스별 이미지 개수를 실수 Tensor로 계산
    counts = torch.bincount(y_train, minlength=3).float()

    # 적은 클래스에 더 큰 값이 부여되도록 개수의 역수 계산
    inverse = 1.0 / counts

    # 클래스 가중치의 평균을 1로 조정
    class_weights = inverse / inverse.mean()

    # weight 조건일 때만 가중치를 사용함. 현재 두 실험은 모두 일반 교차 엔트로피 사용
    criterion = nn.CrossEntropyLoss(weight=class_weights if balance == "weight" else None)

    # 첫 검증 손실을 최적값으로 기록할 수 있도록 양의 무한대로 초기화
    best_loss = float("inf")

    # 학습과 epoch별 검증을 포함하는 구간의 시작 시간 기록
    started = time.perf_counter()

    # 1~8 epoch 학습
    for epoch in range(1, 9):

        # 이전 기울기 초기화
        optimizer.zero_grad()

        # 전체 train 배치의 logit과 정답으로 손실 계산
        loss = criterion(model(X_train), y_train)

        # 학습 가능한 파라미터의 기울기 계산
        loss.backward()

        # 파라미터 갱신
        optimizer.step()

        # 검증 손실 계산에서는 기울기를 기록하지 않음
        with torch.no_grad():
            # train 가중치 적용 여부와 관계없이 일반 교차 엔트로피로 valid를 평가
            vl = nn.CrossEntropyLoss()(model(X_valid), y_valid).item()

        # 실험 설정과 현재 epoch의 검증 손실을 기록
        all_history.append({"strategy":strategy,"balance":balance,"epoch":epoch,"valid_loss":vl})

        # 이전 최적값보다 검증 손실이 낮을 때만 최적 모델 갱신
        if vl < best_loss:
            # 가장 낮은 검증 손실과 해당 epoch 번호 보관
            best_loss=vl;best_epoch=epoch
            # 이후 학습으로 값이 바뀌지 않도록 모델 상태를 분리하여 복사
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}

    # 학습 구간의 경과 시간 계산. 아래 최적 상태 복원과 최종 평가 시간은 제외
    elapsed = time.perf_counter()-started

    # 검증 손실이 가장 낮았던 epoch의 상태 복원
    model.load_state_dict(best_state)

    # 복원한 모델의 검증 예측 수행
    with torch.no_grad():
        # 클래스별 확률로 변환하고 각 이미지의 예측 클래스 번호 선택
        prob=model(X_valid).softmax(1);pred=prob.argmax(1)

    # 클래스별 지표를 딕셔너리로 생성하며 분모가 0인 지표는 0으로 처리
    report=classification_report(y_valid,pred,labels=[0,1,2],output_dict=True,zero_division=0)

    # 전략, 학습 원소 수, 최적 epoch, 정확도, macro F1, triangle Recall, 시간을 기록
    # report['2']는 현재 데이터의 클래스 번호 2인 triangle에 해당
    rows.append({"strategy":strategy,"balance":balance,"params":sum(p.numel() for p in model.parameters() if p.requires_grad),"best_epoch":best_epoch,"accuracy":accuracy_score(y_valid,pred),"macro_f1":f1_score(y_valid,pred,labels=[0,1,2],average="macro",zero_division=0),"triangle_recall":report['2']['recall'],"seconds":elapsed})

# 결과를 표로 변환하고 행 번호 없이 CSV로 저장. seconds 열도 파일에 포함
table=pd.DataFrame(rows);table.to_csv(OUT/"finetune_comparison.csv",index=False)

# 화면에서는 시간 열을 제외하고 소수점 넷째 자리까지 반올림하여 출력
print(table.drop(columns="seconds").round(4).to_string(index=False))

# 기기별로 달라지는 시간은 CSV에서 확인하도록 안내
print("시간은 finetune_comparison.csv에 기록: 기기별로 달라짐")
