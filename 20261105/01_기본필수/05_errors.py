"""
[Day14 - B05 기본 개념: 혼동행렬과 오답 분석]

1. ImageFolder와 클래스 매핑
   - 클래스별 하위 폴더에서 이미지와 정답 라벨을 불러오는 클래스
   - 폴더 이름을 정렬하여 클래스 번호를 자동으로 부여
   - 현재 데이터는 circle=0, square=1, triangle=2로 매핑됨
   - class_to_idx: 클래스 이름→번호 딕셔너리, classes: 번호순 클래스 이름 목록
   - targets: 이미지별 정답 번호, samples: (이미지 경로, 정답 번호) 목록
   - train과 valid의 클래스 이름·번호 매핑은 동일해야 함

2. 전처리와 Tensor 배치
   - Compose: 변환을 순서대로 적용, Resize: 이미지 크기를 32×32로 변경
   - ToTensor: 일반적인 8비트 RGB 이미지를 float [0, 1] Tensor로 변환
   - torch.stack(): 이미지들을 (N, 3, 32, 32) 배치로 쌓음
   - torch.long: 정답 클래스 번호에 사용하는 정수 자료형
   - 정답과 예측의 순서가 같은 이미지 순서에 대응해야 함

3. 전이학습과 모델 구조
   - 별도 합성 이미지로 학습한 소형 source Backbone을 사용
   - Sequential: 등록한 층을 순서대로 실행하는 컨테이너
   - Conv2d: 특징 추출, BatchNorm2d: 정규화, ReLU: 비선형성 추가
   - Pooling으로 공간 크기를 줄이고 Flatten으로 특징을 펼침
   - Backbone(0~8)이 만든 256개 특징을 Head(9)가 3개 logit으로 변환

4. 가중치 복원과 파라미터 동결
   - np.load(): .npz 배열 묶음을 읽음. allow_pickle=False로 객체 배열 로드 금지
   - state_dict(): 파라미터와 BatchNorm 통계 등 모델 상태를 이름별로 반환
   - torch.from_numpy(): 배열을 Tensor로 변환, load_state_dict(): 상태 적용
   - requires_grad=False: 해당 파라미터의 기울기를 계산하지 않음
   - Head만 True로 설정하여 771개 파라미터를 학습
   - eval(): BatchNorm 이동 통계를 고정하며 기울기 계산 자체를 끄지는 않음

5. 학습과 검증
   - AdamW: 기울기로 파라미터를 갱신하는 optimizer
   - CrossEntropyLoss: logit과 정답 클래스 번호로 분류 손실 계산
   - zero_grad(): 기울기 초기화, backward(): 기울기 계산, step(): 갱신
   - 전체 train 배치를 한 번 학습한 초기 모델로 오류를 관찰함
   - torch.no_grad(): 검증 시 기울기를 기록하지 않음
   - 07번의 최종 모델과 구분하며 test 데이터는 읽지 않음

6. logit, Softmax, argmax
   - logit: 확률로 변환하기 전 클래스별 점수
   - softmax(dim=1): 클래스 축의 점수를 합이 1인 확률로 변환
   - argmax(dim=1): 가장 높은 점수의 클래스 번호 선택
   - logit과 Softmax 확률에서 argmax로 선택한 클래스는 같음
   - 확률이 높다고 항상 올바른 예측이거나 잘 보정된 확신도인 것은 아님

7. confusion_matrix()
   - 기본 문법: confusion_matrix(실제 정답, 예측값, labels=[0, 1, 2])
   - 행은 실제 클래스, 열은 예측 클래스
   - labels는 행·열의 클래스 순서를 고정하고 누락된 클래스도 포함
   - 대각선은 올바른 예측 개수, 나머지 칸은 잘못 분류한 개수
   - matrix[0, 2]: 실제 circle을 triangle로 예측한 개수

8. classification_report()
   - 클래스별 Precision, Recall, F1, support와 요약 지표를 출력
   - Precision: 해당 클래스로 예측한 것 중 실제 정답의 비율
   - Recall: 실제 해당 클래스 중 올바르게 예측한 비율
   - F1: Precision과 Recall의 조화평균, support: 실제 클래스별 샘플 수
   - target_names는 labels와 같은 순서의 클래스 이름이어야 함
   - zero_division=0: 분모가 0인 지표를 0으로 처리
   - digits=3: 지표를 소수점 셋째 자리까지 표시

9. 오답 인덱스 추출
   - predictions != y_valid: 예측과 정답이 다르면 True인 Tensor 생성
   - nonzero(): True인 위치를 반환, flatten(): 위치 목록을 1차원으로 펼침
   - tolist(): Tensor를 Python 리스트로 변환
   - errors[:3]: 오답 중 앞의 최대 3개를 선택
   - 오답이 없으면 빈 리스트가 되어 반복문은 실행되지 않음

10. 오답 해석과 공통 설정
    - samples[i][0]으로 이미지 경로를 찾고 classes[번호]로 이름을 확인
    - Path(...).name: 경로에서 파일 이름만 추출
    - 실제·예측은 관찰 결과이며 틀린 원인은 이미지 확인을 거쳐 가설로 구분해야 함
    - 이 파일은 오답 이미지 파일이나 확률을 출력·저장하지 않음
    - Path.resolve().parent: 현재 파일의 폴더 경로
    - set_num_threads(1): CPU 스레드 설정, manual_seed(14): 난수 시드 고정
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""

# 파일 및 폴더 경로 관리
from pathlib import Path
# JSON 처리 도구이며 현재 파일에서는 사용하지 않음
import json
# source 가중치 배열 파일 읽기
import numpy as np
# 표 데이터 처리 도구이며 현재 파일에서는 사용하지 않음
import pandas as pd
# Tensor 연산과 자동 미분
import torch
# 신경망 층과 손실 함수
from torch import nn
# 폴더 기반 이미지 데이터셋과 자동 클래스 매핑
from torchvision.datasets import ImageFolder
# 이미지 전처리 도구
from torchvision import transforms as tr
# 혼동행렬과 분류 보고서 사용. accuracy_score와 f1_score는 여기서는 직접 사용하지 않음
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# CPU 연산 스레드 수를 1개로 설정
torch.set_num_threads(1)
# PyTorch 난수 시드를 고정하여 초기값 재현성을 높임
torch.manual_seed(14)
# 현재 소스 파일이 위치한 폴더 경로
BASE = Path(__file__).resolve().parent
# 원본 이미지와 source 가중치가 있는 폴더
DATA = BASE / "data"
# 공통 출력 경로이며 이 파일에서는 결과 파일을 저장하지 않음
OUT = BASE

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# 소형 source의 입력 규칙에 맞게 이미지 크기와 픽셀 범위를 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# train의 하위 폴더 이름으로 클래스 이름·번호 매핑을 자동 생성
train = ImageFolder(DATA / "train", transform=transform)
# valid도 자체적으로 클래스 매핑 생성. valid.class_to_idx로 확인 가능
valid = ImageFolder(DATA / "valid", transform=transform)
# 각 항목의 이미지인 [0]을 가져와 전체 학습 이미지 배치로 쌓음
X_train = torch.stack([train[i][0] for i in range(len(train))])
# train의 자동 매핑에 따른 정답 번호를 정수 Tensor로 변환
y_train = torch.tensor(train.targets, dtype=torch.long)
# 검증 이미지를 데이터셋 순서대로 하나의 배치로 쌓음
X_valid = torch.stack([valid[i][0] for i in range(len(valid))])
# valid의 자동 매핑에 따른 정답 번호를 이미지와 같은 순서로 가져옴
y_valid = torch.tensor(valid.targets, dtype=torch.long)
# 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
model = nn.Sequential(
    # [0~3] 3채널 → 8채널 특징 추출과 정규화·활성화 후 16×16으로 축소
    nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
    # [4~6] 8채널 → 16채널 특징 추출과 정규화·활성화
    nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
    # [7~9] 4×4 평균 풀링 → 256개 특징으로 펼침 → 3클래스 logit 출력
    nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, 3),
)
# 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
    # 현재 모델 상태를 가져와 파일에 없는 항목의 초기값은 유지
    state = model.state_dict()
    # 제공된 배열을 복사하고 Tensor로 변환하여 같은 이름의 상태를 덮어씀
    state.update({name: torch.from_numpy(saved[name].copy()) for name in saved.files})
# 준비한 source 상태를 실제 모델에 적용
model.load_state_dict(state)
# 전체 모델의 파라미터를 먼저 동결
for p in model.parameters():
    p.requires_grad = False
# Head의 가중치와 편향만 학습 허용
for p in model[9].parameters():
    p.requires_grad = True
# BatchNorm의 이동 통계를 고정. 평가 모드에서도 Head의 기울기 계산은 가능
model.eval()
# Head만 optimizer에 등록하며 학습률은 0.01, 가중치 감쇠는 사용하지 않음
optimizer = torch.optim.AdamW(model[9].parameters(), lr=0.01, weight_decay=0.0)
# 정답 클래스 번호와 logit으로 계산하는 분류 손실 함수 생성
criterion = nn.CrossEntropyLoss()
# 오류 관찰용 초기 1회 학습. 07번 최종 8회 학습 모델과 구분합니다.
for epoch in range(1):
    # 이전 기울기를 초기화
    optimizer.zero_grad()
    # 전체 train 이미지의 예측 logit과 정답으로 손실 계산
    loss = criterion(model(X_train), y_train)
    # Head 파라미터의 기울기 계산
    loss.backward()
    # 계산한 기울기로 Head 갱신
    optimizer.step()
# 검증에서는 기울기 기록 없이 예측
with torch.no_grad():
    # 검증 이미지마다 클래스별 점수 3개 계산
    logits = model(X_valid)
    # 점수를 클래스별 확률로 변환. 현재 파일에서는 이후 직접 사용하지 않음
    probabilities = logits.softmax(dim=1)
    # 이미지별로 가장 높은 점수의 클래스 번호 선택
    predictions = logits.argmax(dim=1)

# TODO B05: 실제 y_valid와 predictions로 혼동행렬을 만드세요. 행=실제, 열=예측.
# BEGIN B05
# 실제 정답을 첫 인자로 전달하고 circle·square·triangle 순서의 3×3 행렬 생성
matrix = confusion_matrix(y_valid, predictions, labels=[0, 1, 2])
# END B05
# 행=실제, 열=예측인 혼동행렬 출력
print("혼동행렬:\n", matrix)
# 같은 클래스 순서의 이름과 함께 클래스별 평가 지표 출력
print(classification_report(y_valid, predictions, labels=[0, 1, 2], target_names=valid.classes, zero_division=0, digits=3))
# 정답과 다른 예측의 위치를 찾아 1차원 Python 인덱스 리스트로 변환
errors = (predictions != y_valid).nonzero().flatten().tolist()
# 전체 오답 수 출력. 오답이 없으면 0으로 보고
print("오답 개수:", len(errors))
# 오답 중 최대 3개의 파일 이름과 실제·예측 클래스 이름 확인
for i in errors[:3]:
    # samples는 검증 배치와 같은 순서이며 classes에 번호를 넣어 클래스 이름 조회
    print(Path(valid.samples[i][0]).name, "실제:", valid.classes[y_valid[i]], "예측:", valid.classes[predictions[i]])
