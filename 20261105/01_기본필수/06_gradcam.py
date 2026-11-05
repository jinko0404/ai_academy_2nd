"""
[Day14 - B06 기본 개념: Grad-CAM과 양의 반응 시각화]

1. Grad-CAM과 Heatmap
   - 선택한 클래스 점수에 대한 특징맵의 기울기로 공간적 반응을 시각화하는 방법
   - Heatmap: 위치별 반응 크기를 색으로 표시한 지도
   - 예측 원인이나 정답을 증명하거나 물체의 정확한 경계를 검출하는 것은 아님

2. target과 logit
   - target: Grad-CAM 계산의 기준으로 선택한 클래스
   - logit: 확률로 변환하기 전의 클래스별 점수
   - 이 코드에서는 가장 큰 logit의 예측 클래스를 target으로 선택
   - 정답 클래스와 다를 수 있으며 다른 클래스를 직접 target으로 선택할 수도 있음
   - 같은 이미지도 target이 바뀌면 기울기와 Heatmap이 달라질 수 있음

3. 데이터셋과 전처리
   - ImageFolder: 폴더 이름을 정렬하여 클래스 번호와 정답을 자동 생성
   - classes: 클래스 이름 목록, targets: 정답 번호, samples: 이미지 경로·정답 목록
   - train과 valid의 클래스 매핑은 동일해야 함
   - Resize와 ToTensor: RGB 이미지를 32×32, float [0, 1]로 변환
   - torch.stack(): 이미지를 (N, 3, 32, 32) 배치로 쌓음

4. 전이학습과 모델 분리
   - 별도 합성 이미지로 학습한 소형 source Backbone을 사용
   - Head만 초기 1회 학습하며 07번 최종 모델과 구분
   - model[:7]: 0~6번 층으로 특징맵을 계산
   - model[7:]: 평균 풀링, Flatten, Linear로 클래스 점수를 계산
   - 이 코드의 features는 (1, 16, 16, 16)이며 두 번째 축이 채널임

5. 파라미터 동결과 기울기 기록
   - requires_grad=False: 해당 파라미터의 기울기를 계산하지 않음
   - eval(): BatchNorm 이동 통계를 고정하며 기울기 기록 자체를 끄지는 않음
   - no_grad(): 일반 검증 예측 시 기울기를 기록하지 않음
   - Grad-CAM은 미분이 필요하므로 no_grad 밖에서 다시 순전파함
   - 입력의 requires_grad_(True)로 동결된 Backbone을 통과하는 연산도 기록
   - 이 설정은 Backbone 파라미터를 다시 학습하도록 여는 것과 다름

6. 정답·오답 사례 선택
   - predictions == y_valid: 정답 위치, !=: 오답 위치를 True로 표시
   - nonzero().flatten().tolist(): True인 위치를 1차원 Python 인덱스 리스트로 변환
   - 각 종류에서 첫 번째 사례를 최대 1개씩 선택
   - 사례가 없으면 없다고 보고하며 임의로 생성하지 않음

7. torch.autograd.grad()
   - 기본 문법: torch.autograd.grad(미분할 점수, 미분 대상 Tensor)
   - one[0, target]: 선택한 이미지의 target logit 하나
   - 점수가 features의 각 원소 변화에 얼마나 민감한지 기울기로 계산
   - 반환값이 튜플이므로 [0]으로 features에 대한 기울기를 가져옴

8. 채널 가중치 계산
   - gradient.mean(dim=(2, 3), keepdim=True): 높이·너비 방향으로 기울기 평균
   - 채널마다 하나의 가중치를 얻으며 shape는 (1, 16, 1, 1)
   - keepdim=True로 차원을 유지하여 features에 채널별로 곱할 수 있음
   - 이 weights는 학습된 Linear 가중치가 아니라 현재 target의 Grad-CAM 가중치임

9. B06: 가중합과 ReLU
   - weights * features: 브로드캐스팅으로 채널별 가중치를 모든 공간 위치에 곱함
   - sum(dim=1): 16개 채널을 합산하여 (1, 16, 16) 공간 지도를 만듦
   - relu(): 음수는 0으로 바꾸고 양의 반응만 남김
   - 채널 가중치는 음수일 수도 있으며 ReLU는 채널 합산 후 적용
   - 양의 반응이 없으면 지도가 모두 0일 수 있음

10. 크기 보간과 정규화
    - unsqueeze(1): 채널 축을 추가하여 (1, 1, 16, 16)으로 변환
    - interpolate(): bilinear 방식으로 공간 지도를 입력 크기 32×32로 확대
    - [0, 0]: 배치·채널 축 제거, detach().numpy(): 미분 그래프에서 분리하여 배열 변환
    - 최대값으로 나눠 표시 범위를 맞추며 1e-8로 0 나눗셈을 방지
    - 이미지마다 정규화하므로 색의 세기를 서로 다른 이미지의 절대 중요도로 비교하면 안 됨

11. OpenCV 합성과 저장
    - permute(1, 2, 0): (채널, 높이, 너비)를 (높이, 너비, 채널)로 변환
    - uint8: 0~255 정수 이미지 자료형
    - applyColorMap(): CAM 값에 색상표 적용
    - OpenCV 색상 영상은 BGR 순서이므로 원본 RGB를 BGR로 변환
    - addWeighted(): 같은 크기·좌표의 원본과 Heatmap을 가중 합성
    - imwrite(): 합성 영상을 파일로 저장. 256×256 확대는 표시용임

12. 공통 실행 설정과 학습
    - Path.resolve().parent: 현재 파일 폴더, manual_seed(14): 난수 시드 고정
    - np.load와 load_state_dict: 제공된 source 상태를 모델에 적용
    - AdamW: 파라미터 갱신, CrossEntropyLoss: logit과 정답으로 손실 계산
    - zero_grad → backward → step: 기울기 초기화 → 계산 → 갱신
    - train으로 학습하고 valid에서 사례를 선택하며 test는 읽지 않음
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
# 오류 관찰용 초기 1회 학습. 07번 최종 8회 학습 모델과 구분합니다.
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

# 실제 정답/오답 예를 각각 최대 1개 고릅니다. 없으면 억지로 만들지 않습니다.
selected = []
# 정답·오답 조건을 각각 검사하여 첫 사례만 선택
for kind, mask in [("correct", predictions == y_valid), ("wrong", predictions != y_valid)]:
    # 조건이 True인 검증 데이터의 위치를 Python 리스트로 변환
    indices = mask.nonzero().flatten().tolist()
    # 리스트가 비어 있지 않으면 해당 종류의 첫 사례를 기록
    if indices:
        selected.append((kind, indices[0]))
    else:
        # 정답 또는 오답 사례가 없으면 그 사실을 그대로 보고
        print(kind, "사례 없음: 지표 그대로 보고")
# 선택한 정답·오답 사례마다 Grad-CAM 생성
for kind, index in selected:
    # 슬라이스로 배치 차원을 유지하여 이미지 한 장을 복사하고 미분 기록 활성화
    image = X_valid[index:index + 1].clone().requires_grad_(True)
    # 0~6번 층으로 두 번째 ReLU까지의 특징맵 계산
    features = model[:7](image)
    # 나머지 층으로 (1, 3) 클래스 logit 계산
    one = model[7:](features)
    # 이번 이미지의 예측 클래스를 분석 target으로 선택
    target = int(one.argmax(1).item())
    # 선택한 logit의 특징맵에 대한 기울기를 반환 튜플에서 추출
    gradient = torch.autograd.grad(one[0, target], features)[0]
    # 공간 축을 평균하여 (1, 16, 1, 1) 채널별 가중치 계산
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    # TODO B06: 채널 가중합에 relu()를 적용해 양의 반응만 남기세요.
    # BEGIN B06
    # 1. (1, 16, 1, 1) 가중치를 (1, 16, 16, 16) 특징맵에 채널별로 곱함.
    #    브로드캐스팅으로 각 채널의 모든 위치에 같은 채널 가중치를 적용함.
    # 2. sum(dim=1)로 채널을 합산하여 (1, 16, 16) 공간 지도를 만듦.
    #    배치 축인 dim=0이나 공간 축인 dim=2를 합산하는 것이 아님.
    # 3. relu()로 합산 결과의 음수를 0으로 바꿔 양의 반응만 남김.
    cam = (weights * features).sum(dim=1).relu()
    # END B06
    # 보간에 필요한 채널 축 추가 → 32×32 확대 → 2차원 선택 → NumPy 배열 변환
    cam = torch.nn.functional.interpolate(cam.unsqueeze(1), size=(32,32), mode="bilinear", align_corners=False)[0,0].detach().numpy()
    # 최대값으로 정규화하며 모두 0일 때도 0으로 나누지 않도록 처리
    cam = cam / max(float(cam.max()), 1e-8)
    # 원본 입력을 HWC 축 순서의 0~255 RGB 이미지로 변환
    rgb = (X_valid[index].permute(1,2,0).numpy()*255).astype(np.uint8)
    # 0~255 CAM에 JET 색상표를 적용하여 BGR Heatmap 생성
    heat = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
    # 원본을 BGR로 맞추고 원본 60%와 Heatmap 40%를 같은 좌표에서 합성
    overlay = cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),0.6,heat,0.4,0)
    # 표시용으로 확대하여 cam_correct.png 또는 cam_wrong.png에 저장
    cv2.imwrite(str(OUT / f"cam_{kind}.png"), cv2.resize(overlay,(256,256)))
    # 사례 종류, 원본 파일 이름, 분석 target 이름과 CAM 크기 출력
    print(kind, Path(valid.samples[index][0]).name, "target:", valid.classes[target], "CAM:", cam.shape)
# 이 그림은 점수에 대한 반응입니다. 원인·정답 증명이 아닙니다.
