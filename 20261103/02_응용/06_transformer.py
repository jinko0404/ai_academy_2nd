
"""
[실습 개념: Transformer를 이용한 시퀀스 분류]

1. Sequence (시퀀스)
   - 순서가 있는 데이터의 묶음
   - 예: 시간에 따른 센서 측정값, 문장의 단어 배열

2. Padding (패딩)
   - 시퀀스마다 데이터 길이가 다를 때 동일한 길이로 맞추는 작업
   - 짧은 데이터 뒤에 0을 추가하여 길이를 맞춘다.
   - Padding 값은 실제 데이터가 아니므로 별도로 구분해야 한다.

3. Padding Mask (패딩 마스크)
   - 실제 데이터와 Padding 위치를 구분하는 True/False 배열
   - True: Padding 위치
   - False: 실제 데이터 위치
   - Transformer가 Padding을 참고하지 않도록 전달한다.

4. Projection (프로젝션)
   - 입력 데이터의 Feature 차원을 원하는 크기로 변환하는 과정
   - 여기서는 입력 Feature 3개를 8차원으로 변환한다.

5. Embedding (임베딩)
   - 데이터를 신경망이 처리할 수 있는 벡터 형태로 표현하는 것
   - Position Embedding은 데이터의 위치 정보를 벡터로 표현한다.

6. Position Embedding (위치 임베딩)
   - 시퀀스에서 데이터가 몇 번째 위치인지 나타내는 정보
   - Transformer는 입력 순서를 자체적으로 알지 못하므로 위치 정보를 추가한다.
   - 값 임베딩과 위치 임베딩을 더해서 사용한다.

7. Transformer (트랜스포머)
   - Attention을 이용하여 시퀀스 내부 데이터 간 관계를 학습하는 신경망
   - 특정 위치가 다른 위치의 정보를 얼마나 참고할지 계산한다.

8. Attention (어텐션)
   - 입력 데이터의 각 위치 사이의 관계와 중요도를 계산하는 방법
   - Padding Mask를 사용하면 Padding 위치를 Key로 참고하지 않는다.

9. Encoder (인코더)
   - 입력 데이터를 받아 Attention 등의 연산으로 특징을 추출하는 부분
   - Transformer Encoder는 여러 Encoder Layer를 쌓아서 구성할 수 있다.

10. Masked Mean (마스킹 평균)
    - Padding을 제외하고 실제 데이터만 평균을 계산하는 방법
    - Attention에서 Padding Key를 제외하는 것과는 별개의 처리다.

11. Classification Head (분류 헤드)
    - 추출된 특징을 이용하여 최종 클래스를 예측하는 신경망 계층
    - 여기서는 8차원 특징을 2개 클래스의 점수로 변환한다.

12. CrossEntropyLoss (교차 엔트로피 손실)
    - 모델의 예측 점수와 실제 정답 사이의 차이를 계산한다.
    - 분류 모델의 학습에 사용하는 손실 함수다.

13. Backpropagation (역전파)
    - 손실을 기준으로 각 파라미터의 Gradient를 계산하는 과정

14. Optimizer (최적화 알고리즘)
    - 계산된 Gradient를 이용하여 모델의 가중치를 업데이트한다.
    - 여기서는 Adam을 사용한다.

15. Epoch (에포크)
    - 전체 학습 데이터를 한 번 학습하는 단위

16. Accuracy (정확도)
    - 전체 예측 중 실제 정답을 맞힌 비율
"""

# ============================================================
# 1. 라이브러리 및 실행 환경 설정
# ============================================================

import torch
from torch import nn
torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
torch.manual_seed(42)     # 난수 시드를 고정하여 실행 결과를 재현할 수 있게 한다.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent  # 현재 Python 파일이 위치한 폴더
OUT = ROOT / "outputs"                  # 결과 저장 폴더
OUT.mkdir(exist_ok=True)                # 폴더가 없으면 생성

import json
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# 2. 데이터 불러오기 및 학습/검증 데이터 분리
# ============================================================

# JSON: 데이터를 Key-Value 형태로 저장하는 파일 형식
# read_text(): 파일 내용을 문자열로 읽는다.
# json.loads(): JSON 문자열을 Python 객체로 변환한다.

items = json.loads(
    (ROOT / "data" / "sequences.json").read_text(encoding="utf-8")
)

tr = [r for r in items if r["split"] == "train"] # train: 모델 학습에 사용하는 데이터
va = [r for r in items if r["split"] == "valid"] # valid: 학습한 모델의 성능을 확인하는 데이터

max_len = 12  # 위치 임베딩에서 사용할 최대 시퀀스 길이


# ============================================================
# 3. Padding을 이용한 시퀀스 길이 맞추기
# ============================================================

# Sequence: 순서가 있는 데이터의 묶음
# 데이터마다 길이가 다르면 하나의 Batch로 처리하기 어렵다.
#
# Padding: 짧은 시퀀스 뒤에 0을 추가하여 길이를 맞추는 작업
# pad_sequence(): 여러 시퀀스의 길이를 가장 긴 시퀀스에 맞춘다.
#
# batch_first=True:
# 데이터의 차원을 (Batch, Sequence, Feature) 순서로 구성한다.

xt = pad_sequence(
    [torch.tensor(r["values"]) for r in tr],
    batch_first=True
)

xv = pad_sequence(
    [torch.tensor(r["values"]) for r in va],
    batch_first=True
)


# ============================================================
# 4. 실제 시퀀스 길이와 정답 레이블 분리
# ============================================================

# Length: Padding을 제외한 실제 데이터의 길이
# Label: 모델이 맞혀야 하는 정답 클래스

lt = torch.tensor([r["length"] for r in tr])
lv = torch.tensor([r["length"] for r in va])

yt = torch.tensor([r["label"] for r in tr])
yv = torch.tensor([r["label"] for r in va])


# ============================================================
# 5. Padding Mask 생성
# ============================================================

# Padding Mask: 실제 데이터와 Padding 위치를 구분하는 배열
#
# torch.arange(): 0부터 지정한 길이 직전까지 위치 번호 생성
# unsqueeze(): 기존 Tensor에 차원을 추가
#
# 위치 번호 >= 실제 길이인 경우 True가 된다.
#
# True  = Padding 위치
# False = 실제 데이터 위치
#
# 예: 실제 길이가 3이고 전체 길이가 5라면
# [False, False, False, True, True]

mt = torch.arange(xt.size(1)).unsqueeze(0) >= lt.unsqueeze(1)
mv = torch.arange(xv.size(1)).unsqueeze(0) >= lv.unsqueeze(1)


# ============================================================
# 6. Transformer 모델 구성
# ============================================================

# ModuleDict: 여러 신경망 계층을 이름으로 구분하여 관리하는 자료구조

parts = nn.ModuleDict({

    # Projection: 입력 데이터의 Feature 차원을 변환한다.
    # 입력 Feature 3개 → 출력 Feature 8개
    "projection": nn.Linear(3, 8),

    # Position Embedding: 시퀀스에서 각 데이터의 위치 정보를 표현한다.
    # 최대 12개 위치를 각각 8차원 벡터로 변환한다.
    "position": nn.Embedding(12, 8),

    # Transformer Encoder:
    # Attention을 이용해 시퀀스 내부 데이터 간 관계를 학습한다.
    "encoder": nn.TransformerEncoder(

        # Transformer Encoder Layer 하나를 구성한다.
        nn.TransformerEncoderLayer(
            8,                  # d_model: 입력 및 출력 벡터의 차원
            2,                  # nhead: Attention Head 개수
            dim_feedforward=16, # Feed Forward 신경망의 내부 차원
            dropout=0,          # Dropout: 학습 중 일부 값을 무작위로 제거하는 기법
            batch_first=True   # 입력 차원을 (Batch, Sequence, Feature)로 사용
        ),

        1,  # Encoder Layer를 1개 쌓는다.

        enable_nested_tensor=False
    ),

    # Classification Head:
    # 추출된 특징을 최종 클래스별 점수로 변환한다.
    # 8차원 특징 → 2개 클래스 점수
    "head": nn.Linear(8, 2)
})


# ============================================================
# 7. Optimizer 설정
# ============================================================

# Optimizer: 모델의 가중치를 업데이트하는 최적화 알고리즘
# Adam: Gradient를 이용하여 학습률을 조정하면서 가중치를 업데이트한다.
#
# parts.parameters():
# 모델 내부의 모든 학습 가능한 파라미터를 전달한다.

optimizer = torch.optim.Adam(
    parts.parameters(),
    lr=0.01  # Learning Rate: 가중치 업데이트 크기를 조절하는 학습률
)


# ============================================================
# 8. Transformer 모델 학습
# ============================================================

# Epoch: 전체 학습 데이터를 한 번 학습하는 단위
# 여기서는 전체 학습 데이터를 30번 반복한다.

for epoch in range(30):

    parts.train()             # 모델을 학습 모드로 전환
    optimizer.zero_grad()     # 이전 학습에서 계산된 Gradient 초기화

    # Projection: 입력 Feature를 8차원으로 변환
    # Position Embedding: 각 위치의 정보를 8차원으로 변환
    #
    # 두 벡터를 더해서 값 정보와 위치 정보를 함께 전달한다.
    embedded = (
        parts["projection"](xt)
        + parts["position"](torch.arange(xt.size(1)))
    )

    # TODO A06: Transformer에 실제 길이로 만든 Padding Mask를 전달한다.
    # 위치 임베딩은 값 임베딩과 더한다. masked mean에서는 Padding query도 제외한다.
    # BEGIN A06

    # Attention: 시퀀스 내부 데이터 간 관계와 중요도를 계산한다.
    #
    # src_key_padding_mask:
    # Padding 위치를 Key로 참고하지 않도록 차단하는 Mask
    # True = Padding이므로 Attention에서 제외
    #
    # 주의: Padding Query의 출력까지 자동으로 0이 되는 것은 아니다.
    encoded = parts["encoder"](
        embedded,
        src_key_padding_mask=mt
    )

    # END A06

    # ========================================================
    # 9. Masked Mean: Padding을 제외한 평균 계산
    # ========================================================

    # Attention에서 Padding Key를 제외했더라도
    # Padding Query에 해당하는 출력은 별도로 제외해야 한다.
    #
    # ~mt: Mask의 True/False를 반전
    # True = 실제 데이터, False = Padding
    #
    # unsqueeze(-1): Feature 차원을 추가하여 곱셈이 가능하게 한다.
    # sum(1): Sequence 방향으로 실제 데이터만 합산한다.
    # lt.unsqueeze(-1): 실제 시퀀스 길이로 나누어 평균을 계산한다.

    pooled = (
        encoded * (~mt).unsqueeze(-1)
    ).sum(1) / lt.unsqueeze(-1)

    # ========================================================
    # 10. 손실 계산 및 가중치 업데이트
    # ========================================================

    # Classification Head:
    # 시퀀스의 대표 특징을 2개 클래스의 점수(Logits)로 변환한다.
    #
    # CrossEntropyLoss:
    # 예측 점수와 실제 정답의 차이를 계산한다.
    loss = nn.CrossEntropyLoss()(
        parts["head"](pooled),
        yt
    )

    # Backpropagation: 손실을 기준으로 Gradient 계산
    loss.backward()

    # Optimizer: 계산된 Gradient를 이용하여 가중치 업데이트
    optimizer.step()


# ============================================================
# 11. 검증 데이터 평가
# ============================================================

parts.eval()  # 모델을 평가 모드로 전환

# Inference Mode:
# 검증 및 예측 과정에서는 Gradient 계산을 생략한다.
# 학습에 필요한 연산을 줄여 추론을 효율적으로 수행한다.

with torch.inference_mode():

    # 학습 데이터와 동일하게 입력값과 위치 정보를 결합한다.
    embedded = (
        parts["projection"](xv)
        + parts["position"](torch.arange(xv.size(1)))
    )

    # 검증 데이터의 Padding Mask를 Transformer에 전달한다.
    encoded = parts["encoder"](
        embedded,
        src_key_padding_mask=mv
    )

    # Padding을 제외하고 실제 데이터만 평균 계산한다.
    pooled = (
        encoded * (~mv).unsqueeze(-1)
    ).sum(1) / lv.unsqueeze(-1)

    # Classification Head로 클래스별 점수를 계산한다.
    # argmax(1): 각 데이터에서 가장 높은 점수의 클래스 인덱스를 선택한다.
    pred = parts["head"](pooled).argmax(1)


# ============================================================
# 12. 평가 결과 출력 및 CSV 저장
# ============================================================

# 학습 데이터와 검증 데이터 개수 및 최대 위치 길이 출력
print("train/valid:", len(tr), len(va), "위치 최대 길이:", 12)

# 모델의 학습 가능한 파라미터 개수 출력
# numel(): Tensor 안에 들어 있는 전체 원소 개수
print("params:", sum(p.numel() for p in parts.parameters()))

# Accuracy: 전체 검증 데이터 중 정답을 맞힌 비율
# pred == yv: 예측값과 실제 정답 비교
# float(): True와 False를 1.0과 0.0으로 변환
# mean(): 정답 비율의 평균 계산
print(
    "valid accuracy:",
    round((pred == yv).float().mean().item(), 3)
)

# DataFrame: 표 형태로 데이터를 관리하는 pandas 자료구조
# 검증 데이터의 ID, 실제 정답, 예측 결과를 표로 생성한다.
#
# to_csv(): CSV 파일로 저장한다.
# index=False: 불필요한 행 번호를 파일에 저장하지 않는다.

pd.DataFrame({
    "id": [r["id"] for r in va],
    "actual": yv.tolist(),
    "prediction": pred.tolist()
}).to_csv(
    OUT / "transformer_predictions.csv",
    index=False
)

