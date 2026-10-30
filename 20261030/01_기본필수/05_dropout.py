import torch
from torch import nn

# Regularization(정규화)는 머신러닝 모델이 학습 데이터에만 지나치게 맞춰지는 과적합(Overfitting)을 방지하는 기법
# 핵심 목적은 과적합 방지와 일반화 성능 향상

# Dropout(드롭아웃)은 신경망 학습 중에 일부 뉴런을 무작위로 비활성화하는 정규화 기법
# training 모드에서만 mask가 적용되고, evaluation 모드에서는 입력을 그대로 전달한다.
# 학습할 때는 과적합을 막고, 평가할 때는 정상적인 예측을 하기 위해서다.
# 즉, 학습할 때는 일부러 방해해서 과적합을 줄이고, 평가할 때는 전체 뉴런을 사용해서 예측하는 것

# 난수 시드를 고정하여 실행할 때마다 동일한 난수 결과를 얻도록 설정
torch.manual_seed(42)

# 1로 채워진 2행 4열 텐서 생성
X = torch.ones(2, 4)

# Dropout 객체 생성
# p=0.5는 학습 중 각 뉴런이 50% 확률로 비활성화된다는 의미
dropout = nn.Dropout(p=0.5)

# Dropout을 학습 모드로 설정
# 학습 모드에서는 일부 값을 무작위로 0으로 만들고 나머지 값은 2배로 조정
dropout.train()

# Dropout을 3번 실행하여 무작위 마스크 적용 결과 확인
for i in range(3):
    print("train", i + 1, dropout(X).tolist())
# Dropout을 평가 모드로 바꾸세요. dropout.eval()은 mode만 바꿉니다.
# 평가 모드의 Dropout은 입력을 그대로 전달하며, gradient 비활성화와는 별개의 기능입니다.
# >>> B06
# TODO B06: 위 요구사항에 맞게 이 블록만 완성합니다.
dropout.eval() # 평가 모드로 변경하여 Dropout이 입력값을 변경하지 않도록 설정
# <<< B06

# 평가 모드에서 Dropout을 3번 실행하여 입력값이 그대로 유지되는지 확인
for i in range(3):
    print("eval", i + 1, dropout(X).tolist())
