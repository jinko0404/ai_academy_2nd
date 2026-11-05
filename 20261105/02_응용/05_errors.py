"""
[Day14 - A05 기본 개념: 오답 인덱스 추출과 이미지 저장]

1. ImageFolder와 클래스 매핑
   - 폴더 이름을 정렬하여 클래스 번호와 정답 라벨을 자동 생성
   - 현재 데이터는 circle=0, square=1, triangle=2
   - classes: 번호순 클래스 이름, targets: 이미지별 정답 번호
   - samples: (이미지 경로, 정답 번호) 목록
   - train과 valid의 클래스 매핑은 동일해야 함

2. 이미지 전처리와 배치
   - Compose: 여러 변환을 순서대로 적용
   - Resize((32, 32)): 크기 변경, ToTensor(): 일반적인 8비트 이미지를 [0, 1]로 변환
   - torch.stack(): 이미지 Tensor들을 (N, 3, 32, 32)로 쌓음
   - torch.long: 정답 클래스 번호에 사용하는 정수 자료형
   - 이미지, 정답, 예측, samples의 순서가 일치해야 같은 행을 비교할 수 있음

3. Backbone과 Head
   - 별도 합성 이미지로 사전학습한 소형 source Backbone을 사용
   - Sequential: 등록한 층을 순서대로 실행
   - Conv2d: 특징 추출, BatchNorm2d: 정규화, ReLU: 비선형성 추가
   - Pooling으로 공간 크기를 줄이고 Flatten으로 256개 특징을 만듦
   - Linear(256, 3): 클래스별 logit 3개를 출력하는 Head

4. 상태 복원과 파라미터 동결
   - np.load(): 배열 파일 읽기, allow_pickle=False: pickle 객체 배열 로드 금지
   - state_dict(): 파라미터와 BatchNorm 통계 등 모델 상태를 이름별로 반환
   - torch.from_numpy(): 배열을 Tensor로 변환, load_state_dict(): 상태 복원
   - requires_grad=False: 해당 파라미터의 기울기를 계산하지 않음
   - Head만 True로 설정하여 학습
   - eval(): BatchNorm 이동 통계를 고정하며 기울기 계산 자체를 끄지는 않음

5. 학습과 검증
   - AdamW: 기울기로 파라미터 갱신, CrossEntropyLoss: 분류 손실 계산
   - zero_grad(): 기울기 초기화, backward(): 기울기 계산, step(): 갱신
   - 전체 train 배치로 8회 학습하며 valid를 역전파에 사용하지 않음
   - no_grad(): 검증 시 기울기를 기록하지 않음. test는 읽지 않음

6. logit, 확률, 예측 클래스
   - logit: 확률 변환 전 점수, softmax(dim=1): 클래스별 확률로 변환
   - argmax(dim=1): 각 이미지에서 가장 높은 점수의 클래스 번호 선택
   - probabilities의 shape는 (N, 3), predictions와 y_valid는 각각 (N,)
   - 최대 확률은 모델의 점수이며 정답 보장이나 보정된 확신도를 의미하지 않음

7. A05: 원소별 비교와 Boolean Tensor
   - predictions != y_valid: 같은 위치의 예측과 정답을 하나씩 비교
   - 다르면 True(오답), 같으면 False(정답)가 됨
   - 예: 정답 [0, 1, 2, 1], 예측 [0, 0, 2, 2]
   - 비교 결과: [False, True, False, True]
   - 여기서 필요한 것은 틀린 클래스 번호가 아니라 틀린 샘플의 위치임

8. A05: nonzero(), flatten(), tolist()
   - nonzero(): 0이 아닌 원소의 좌표를 반환. Boolean에서는 True의 위치를 반환
   - 위 예제의 결과: tensor([[1], [3]])이며 shape는 (오답 수, 1)
   - flatten(): 좌표들을 1차원으로 펼쳐 tensor([1, 3])으로 변환
   - tolist(): Tensor를 Python 리스트 [1, 3]으로 변환
   - 인덱스는 0부터 시작하므로 두 번째와 네 번째 이미지가 오답이라는 뜻
   - 모두 맞혔다면 결과는 []이며 오답 이미지 저장 반복문이 실행되지 않음

9. 인덱스로 원본 데이터 찾기
   - 같은 i로 valid.samples[i], y_valid[i], predictions[i]를 조회
   - valid.samples[i][0]: 해당 이미지 경로
   - valid.classes[번호]: 실제·예측 번호를 클래스 이름으로 변환
   - indices[:8]: 오답 중 앞의 최대 8개를 선택하며 전체 오답 수와 다를 수 있음

10. PIL 이미지와 텍스트 저장
    - Image.open(): 이미지 열기, convert("RGB"): 3채널 변환
    - resize((192, 192)): 표시용 확대이며 모델 입력 크기는 여전히 32×32
    - Image.new(): 흰색 캔버스 생성, paste(): 이미지 붙이기
    - ImageDraw.Draw(): 그림·글씨 도구 생성, text(): 위치에 문자열 그리기
    - save(): 완성한 타일을 이미지 파일로 저장

11. 파일 관리와 결과 표
    - glob("error_*.png"): 이전 오답 이미지 검색, unlink(): 해당 파일 삭제
    - 이 코드는 재실행 시 outputs의 error_*.png 파일을 먼저 삭제함
    - f"error_{i:03d}.png": 인덱스를 최소 세 자리 숫자로 표시
    - item(): 단일 원소 Tensor를 Python 숫자로 변환, .3f: 소수점 세 자리 표시
    - DataFrame과 to_csv(index=False): 저장한 오답 정보를 행 번호 없이 CSV로 저장
    - 오답이 없어도 errors.csv는 열 이름을 가진 빈 표로 저장됨

12. 해석과 공통 설정
    - 실제·예측·확률은 관찰 결과이며 오답 원인은 이미지 확인 후 가설로 구분
    - 이번 8회 학습 모델의 결과는 기본필수 05의 1회 학습 모델과 구분
    - Path.resolve().parent: 현재 파일 폴더, mkdir(exist_ok=True): 출력 폴더 생성
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
OUT = BASE / "outputs"
# 결과 저장 폴더가 없으면 생성하고 이미 있으면 그대로 사용
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# 소형 source의 입력 규칙에 맞게 이미지 크기와 픽셀 범위를 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
# 이미지 읽기와 이미지 위 텍스트 그리기 도구
from PIL import Image, ImageDraw
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

# TODO A05: 실제값과 예측값이 다른 행의 인덱스를 구하세요.
# BEGIN A05
# 목표: 예측이 틀린 이미지의 위치를 Python 리스트로 구함.
# 아래는 이해를 위한 중간 단계 예시이며 실행 코드는 마지막 한 줄임.
#
# 1단계: (predictions != y_valid)
#   정답이 tensor([0, 1, 2, 1]), 예측이 tensor([0, 0, 2, 2])라면
#   결과는 tensor([False, True, False, True])임. True가 오답 위치를 표시함.
#
# 2단계: .nonzero()
#   True인 위치를 좌표 형태로 찾아 tensor([[1], [3]])을 반환함.
#   클래스 번호가 1과 3이라는 뜻이 아니라 인덱스 1과 3의 이미지가 틀렸다는 뜻임.
#
# 3단계: .flatten()
#   tensor([[1], [3]])을 tensor([1, 3])으로 펼침.
#   뒤의 for i in indices에서 이미지 위치를 하나씩 사용할 수 있는 형태로 정리함.
#
# 4단계: .tolist()
#   tensor([1, 3])을 일반 Python 리스트 [1, 3]으로 변환함.
#   실제 실행에서 모두 맞혔다면 []가 나오는 것이 정상이며 오답을 억지로 만들지 않음.
indices = (predictions != y_valid).nonzero().flatten().tolist()
# END A05
# 실제로 저장한 오답 이미지의 정보를 모을 리스트
records=[]
# 이전 실행의 오답 PNG를 삭제하여 이번 결과와 섞이지 않도록 처리
for old in OUT.glob("error_*.png"):old.unlink()
# 전체 오답 중 앞의 최대 8개만 이미지로 저장
for i in indices[:8]:
    # 같은 인덱스의 원본 이미지를 RGB로 읽어 표시용으로 192×192 확대
    image=Image.open(valid.samples[i][0]).convert("RGB").resize((192,192))
    # 아래쪽에 설명 공간을 둔 흰색 캔버스를 만들고 이미지와 그리기 도구 준비
    tile=Image.new("RGB",(192,240),"white");tile.paste(image,(0,0));draw=ImageDraw.Draw(tile)
    # 정답 번호에 대응하는 실제 클래스 이름 표시
    draw.text((5,195),f"true: {valid.classes[y_valid[i]]}",fill="black")
    # 예측 번호에 대응하는 클래스 이름 표시
    draw.text((5,210),f"pred: {valid.classes[predictions[i]]}",fill="black")
    # 예측 클래스의 확률을 Python 숫자로 바꾸어 소수점 세 자리까지 표시
    draw.text((5,225),f"p={probabilities[i].max().item():.3f}",fill="black")
    # 검증 데이터의 인덱스를 파일 이름에 넣어 저장
    tile.save(OUT/f"error_{i:03d}.png")
    # 저장한 이미지의 파일 이름, 실제·예측 번호와 최대 확률 기록
    records.append({"file":Path(valid.samples[i][0]).name,"actual":int(y_valid[i]),"predicted":int(predictions[i]),"confidence":probabilities[i].max().item()})
# 저장된 오답 정보를 CSV로 작성. 오답이 없어도 열 이름은 유지
pd.DataFrame(records,columns=["file","actual","predicted","confidence"]).to_csv(OUT/"errors.csv",index=False)
# 전체 오답 수와 실제 저장 수를 구분하여 출력. 오답이 없으면 둘 다 0
print("전체 오답:",len(indices),"저장:",len(records),"없으면 생성하지 않음")
