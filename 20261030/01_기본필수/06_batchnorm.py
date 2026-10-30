import torch
from torch import nn

"""
==================================================
기본 개념 정리
==================================================

# BatchNorm은 Batch 통계로 activation을 정규화한다
# BatchNorm은 한 번에 들어온 데이터 묶음(Batch)을 기준으로 신경망의 중간 출력값을 일정한 기준으로 맞춰주는 기능
# Activation은 신경망의 한 층을 통과한 결과값
# 정규화가 필요한 가장 큰 이유는 데이터의 값이 지나치게 크거나 분포가 불안정하면 모델의 학습이 어려워질 수 있기 때문이다.
# 정규화(Normalization)란 데이터의 값이나 분포를 일정한 기준에 맞게 조정하는 과정
# [Training Mode]
# 현재 학습 데이터(Batch)의 평균과 분산을 계산하여 정규화에 사용
# 계산한 평균과 분산을 바탕으로 Running Mean과 Running Var를 갱신
# [Evaluation Mode]
# 학습 과정에서 누적해 둔 Running Mean과 Running Var를 사용하여 정규화
# 현재 Batch의 평균과 분산을 새로 계산하거나 Running 통계값을 갱신하지 않음

# Batch Mean	 현재 Batch 데이터의 평균
# Batch Var	현재  Batch 데이터의 분산
# Running Mean	 학습 중 누적한 평균
# Running Var	 학습 중 누적한 분산

# 인공 신경망 (ANN, Artificial Neural Network)
# 인공 신경망은 사람의 뇌에 있는 신경세포(뉴런)의 연결 구조를 모방하여 만든 컴퓨터 학습 모델이야.
# 쉽게 말하면 데이터를 입력받고, 여러 계산 과정을 거쳐 결과를 예측하도록 만든 구조

# Input Layer : 데이터를 입력받는 층
# Hidden Layer : 입력 데이터를 계산하고 특징을 학습하는 층
# Output Layer : 최종 예측 결과를 출력하는 층

# Weight : 입력 데이터가 결과에 미치는 영향력
# Bias : 계산 결과를 조정하는 값
# Activation Function : 비선형적인 관계를 학습할 수 있도록 돕는 함수

# Weight Decay (가중치 감소)
# 가중치가 지나치게 커지는 것을 억제하는 정규화 기법
# 큰 가중치에 페널티를 부여하여 과적합을 방지하는 데 도움을 줌
"""

torch.manual_seed(42)   # 난수 시드 고정
X = torch.tensor([[1.,2.],[2.,1.],[3.,4.],[4.,3.]]) # 입력 데이터 (4개 데이터, 특성 2개)
bn = nn.BatchNorm1d(num_features=8) # 특성 8개에 대한 BatchNorm 생성
model = nn.Sequential(
    nn.Linear(2, 8),  # 입력 특성 2개를 8개로 변환
    bn,               # 8개 특성의 평균과 분산을 이용해 정규화
    nn.ReLU()         # 음수 값을 0으로 변환
)
model.train() # 학습 모드 활성화
# model에 X를 넣어 output을 구하세요. 입력 (4,2)는 Linear에서 (4,8)이 됩니다.
# BatchNorm1d(8)은 8개 특성의 통계를 유지합니다.
# >>> B07
# TODO B07: 위 요구사항에 맞게 이 블록만 완성합니다.
output = model(X) # X를 모델에 입력하여 전체 신경망 실행
# <<< B07

print("입력:", tuple(X.shape), "출력:", tuple(output.shape))
print("running_mean:", tuple(bn.running_mean.shape))

before = bn.running_mean.clone()    # 평가 전 누적 평균 복사
model.eval()    # 평가 모드로 변경

with torch.no_grad():  # 기울기 계산 비활성화
    model(X)  # 평가 모드로 모델 실행
    
print("eval에서 통계 유지:", torch.equal(before, bn.running_mean))
