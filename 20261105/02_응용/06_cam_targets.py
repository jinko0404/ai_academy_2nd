"""
[Day14 - A06 기본 개념: target별 Grad-CAM과 기울기 계산]

1. Grad-CAM과 Heatmap
   - 선택한 클래스 점수에 대한 특징맵의 기울기를 이용하여 공간 반응을 시각화
   - Heatmap은 위치별 반응 크기를 색으로 표현한 지도
   - 예측 원인·정답의 증명이나 물체의 정확한 경계·탐지 박스가 아님

2. target과 이번 실험
   - target은 분석 기준으로 선택한 클래스 번호
   - 이 코드는 같은 valid 이미지 한 장에 target 0과 1을 각각 지정
   - 현재 매핑에서는 circle=0, square=1, triangle=2
   - 예측 클래스와 관계없이 분석할 target을 직접 선택할 수 있음
   - 같은 이미지라도 target이 바뀌면 기울기와 Heatmap이 달라질 수 있음

3. ImageFolder와 Tensor 배치
   - 폴더 이름을 정렬하여 클래스 이름과 번호를 자동 생성
   - classes: 번호순 이름, targets: 정답 번호, samples: 이미지 경로·정답 목록
   - train과 valid의 클래스 매핑은 동일해야 함
   - Resize와 ToTensor로 RGB 32×32, float [0, 1] 입력 생성
   - stack(): 이미지를 (N, 3, 32, 32) 배치로 쌓음

4. 전이학습과 모델 구조
   - 별도 합성 이미지로 학습한 소형 source Backbone을 사용
   - Sequential은 층을 등록 순서대로 실행하며 Head(9)만 8회 학습
   - Conv: 특징 추출, BatchNorm: 정규화, ReLU: 비선형성 추가
   - Pooling과 Flatten으로 256개 특징을 만들고 Linear로 3개 logit 계산
   - state_dict와 load_state_dict로 제공된 모델 상태를 복원

5. eval(), no_grad(), requires_grad
   - 파라미터의 requires_grad=False는 해당 파라미터의 기울기 계산을 막음
   - eval()은 BatchNorm 이동 통계를 고정하며 미분 자체를 끄지는 않음
   - no_grad()는 일반 검증 예측에서 기울기 기록을 끔
   - Grad-CAM은 no_grad 밖에서 입력의 requires_grad를 켜고 새로 계산
   - 입력 미분 기록을 켜는 것은 Backbone 파라미터를 다시 학습시키는 것과 다름

6. features, out, logits의 차이
   - features=model[:7](image): 0~6번 층의 특징맵, shape (1, 16, 16, 16)
   - out=model[7:](features): 해당 특징맵에서 계산한 logit, shape (1, 3)
   - logits는 앞의 no_grad에서 계산한 전체 검증 예측이므로 A06에 사용하지 않음
   - 미분할 점수는 현재 features로부터 기울기를 기록하며 계산된 out에서 선택해야 함

7. A06: out[0, target]
   - out의 첫 축은 이미지 번호, 두 번째 축은 클래스 번호
   - 0은 현재 한 장짜리 배치의 첫 이미지, target은 분석할 클래스
   - out[0, target]은 미분할 점수 하나인 scalar Tensor
   - target 자체는 단순 클래스 번호이므로 미분할 점수가 아님
   - out[target]은 이미지 축을 선택하므로 클래스 선택과 다름

8. A06: torch.autograd.grad()와 마지막 [0]
   - 기본 문법: torch.autograd.grad(미분할 점수, 미분 대상 Tensor)
   - out[0, target]을 features의 각 원소에 대해 미분
   - 반환값은 Tensor가 아니라 입력별 기울기를 담은 튜플
   - 미분 대상이 features 하나이므로 (기울기 Tensor,) 형태로 반환
   - 마지막 [0]으로 첫 기울기 Tensor를 꺼내야 mean()을 사용할 수 있음
   - gradient의 shape는 features와 같은 (1, 16, 16, 16)

9. 채널 가중합과 ReLU
   - gradient.mean((2,3), keepdim=True): 공간 축을 평균하여 채널별 가중치 계산
   - (1, 16, 1, 1) 가중치를 features에 브로드캐스팅하여 곱함
   - sum(1): 채널을 합산하여 (1, 16, 16) 지도 생성
   - relu(): 음수를 0으로 바꾸어 양의 반응만 유지
   - 양의 반응이 없으면 CAM이 모두 0일 수 있음

10. 보간과 표시용 정규화
    - unsqueeze(1): 채널 축 추가, interpolate(): 지도를 입력 크기 32×32로 확대
    - [0,0]: 배치·채널 축 제거, detach().numpy(): 그래프에서 분리하여 배열 변환
    - 최대값으로 나누어 표시 범위를 맞추고 1e-8로 0 나눗셈 방지
    - 각 지도별 정규화이므로 색이나 mean CAM으로 클래스의 절대 중요도를 비교할 수 없음

11. OpenCV 시각화
    - permute(1,2,0): CHW → HWC 축 변환, uint8: 0~255 정수 이미지로 변환
    - applyColorMap(): CAM에 색상표 적용, OpenCV 색상 영상은 BGR 순서
    - cvtColor(): 원본 RGB를 BGR로 변환하여 Heatmap과 색상 순서 일치
    - addWeighted(): 같은 크기·좌표에서 원본 60%와 Heatmap 40% 합성
    - imwrite(): target별 이미지 저장. 256×256 확대는 표시용임

12. 학습과 재현성
    - AdamW: 파라미터 갱신, CrossEntropyLoss: logit과 정답으로 손실 계산
    - zero_grad → backward → step: 기울기 초기화 → 계산 → 갱신
    - manual_seed(14): PyTorch 난수 고정, set_num_threads(1): CPU 스레드 설정
    - train으로 학습하고 valid 이미지로 시각화하며 test는 읽지 않음
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
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# 소형 source의 입력 규칙에 맞게 이미지 크기와 픽셀 범위를 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
# Heatmap 색상 적용, 원본 합성, 이미지 저장에 사용하는 OpenCV
import cv2
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
# requires_grad는 가중치 갱신, eval()은 BatchNorm 통계 사용 방식을 제어합니다.
# 전체 모델의 파라미터를 먼저 동결
for p in model.parameters():
    p.requires_grad = False
# Head의 가중치와 편향만 학습 허용
for p in model[9].parameters():
    p.requires_grad = True
model.eval()  # 고정 Backbone의 running 통계도 보존
# Head만 optimizer에 등록하며 학습률은 0.01, 가중치 감쇠는 사용하지 않음
optimizer = torch.optim.AdamW(model[9].parameters(), lr=0.01, weight_decay=0.0)
# 정답 클래스 번호와 logit으로 계산하는 분류 손실 함수 생성
criterion = nn.CrossEntropyLoss()
# 제공된 8회 학습. 평가용 valid를 backward에 넣지 않습니다.
for epoch in range(8):
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

# 검증 데이터의 첫 번째 이미지를 두 target에서 공통으로 사용
index=0
# 이미지를 바꾸지 않고 circle(0), square(1) 기준의 Grad-CAM을 각각 생성
for target in [0,1]:
    # 배치 축을 유지하여 이미지를 복사하고 동결 Backbone 내부 연산의 미분 기록 활성화
    image=X_valid[index:index+1].clone().requires_grad_(True)
    # 앞 7개 층의 특징맵과 그 특징맵에서 이어지는 클래스 점수를 새로 계산
    # target마다 그래프도 새로 생성하므로 이전 그래프를 유지할 필요가 없음
    features=model[:7](image);out=model[7:](features)
    # TODO A06: target logit의 features에 대한 gradient를 계산하세요.
    # BEGIN A06
    # out[0, target]: 첫 이미지에서 선택한 클래스의 logit 하나
    # features: 그 점수를 어떤 Tensor에 대해 미분할지 지정
    # 반환 예: (기울기 Tensor,) → 끝의 [0]으로 Tensor만 추출
    # 앞의 out[0, target] 안의 0과 끝의 [0]은 서로 다른 역할임.
    # 끝의 [0]을 생략하면 gradient가 튜플이 되어 아래 mean()에서 오류가 발생함.
    gradient = torch.autograd.grad(out[0, target], features)[0]
    # END A06
    # 공간 평균 기울기를 채널 가중치로 곱하고 채널 합산 후 양의 반응만 유지
    cam=(gradient.mean((2,3),keepdim=True)*features).sum(1).relu()
    # 채널 축 추가 → 32×32 보간 → 2차원 선택 → 그래프에서 분리하여 배열 변환
    cam=torch.nn.functional.interpolate(cam.unsqueeze(1),(32,32),mode="bilinear",align_corners=False)[0,0].detach().numpy()
    # 최대 반응을 기준으로 정규화하고 모두 0일 때 0 나눗셈 방지
    cam=cam/max(float(cam.max()),1e-8)
    # 같은 입력 이미지의 축과 값 범위를 표시용 RGB 형식으로 변환
    rgb=np.uint8(X_valid[index].permute(1,2,0).numpy()*255)
    # CAM을 0~255로 바꾸고 JET 색상표 적용
    heat=cv2.applyColorMap(np.uint8(cam*255),cv2.COLORMAP_JET)
    # 원본을 BGR로 바꿔 동일 좌표의 Heatmap과 가중 합성
    overlay=cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),.6,heat,.4,0)
    # outputs에 target_0_cam.png, target_1_cam.png로 각각 저장
    cv2.imwrite(str(OUT/f"target_{target}_cam.png"),cv2.resize(overlay,(256,256)))
    # 분석 클래스, 해당 logit, 정규화된 CAM 평균 출력. 평균은 확률이나 정확도가 아님
    print("target:",valid.classes[target],"logit:",round(out[0,target].item(),4),"mean CAM:",round(float(cam.mean()),4))


"""
결론: 같은 이미지에서 circle의 logit은 1.5821로 square의 -1.351보다 높았다.
target을 바꾸었을 때 정규화된 CAM 평균도 서로 다르게 나타났다.
이는 같은 이미지라도 분석 기준 클래스에 따라 반응 지도가 달라질 수 있음을 보여준다.
다만 각 CAM은 개별적으로 정규화되므로 평균값이 더 높다고 해당 클래스의 예측 점수가 더 높다는 뜻은 아니다.
구체적인 반응 위치는 저장된 두 Heatmap을 직접 비교하여 확인해야 한다.
"""