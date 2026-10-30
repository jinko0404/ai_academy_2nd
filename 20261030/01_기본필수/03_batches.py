import torch
from torch.utils.data import TensorDataset, DataLoader

# 학습용 공통 데이터와 별개인 '배치 개수만 확인하는' 20개 숫자입니다.
# 1. 테스트용 데이터 생성
# 0부터 39까지 숫자를 생성한 뒤, 2개씩 묶어 20행 2열로 변환
X = torch.arange(40, dtype=torch.float32).reshape(20, 2)
y = torch.arange(20) % 2        # 0과 1이 반복되는 정답 데이터 20개 생성
dataset = TensorDataset(X, y)   # 입력 데이터 X와 정답 데이터 y를 하나의 데이터셋으로 묶음
# DataLoader(dataset, batch_size=6, shuffle=False)를 만드세요.
# 20개를 순서대로 나누면 6·6·6·2개입니다. dataset은 필수, batch_size와 shuffle은 선택 인자입니다.
# >>> B04
# TODO B04: 위 요구사항에 맞게 이 블록만 완성합니다.
# 2. DataLoader 생성
# 데이터를 6개씩 묶어서 전달
# shuffle=False이므로 데이터 순서를 섞지 않음
loader = DataLoader(dataset, batch_size=6, shuffle=False)
#학습할 때에는 순서에 따른 편향을 줄이고 다양한 순서로 데이터를 학습시키기위해
#shuffle을 True로 둔다.
# <<< B04
# 3. 배치 단위로 데이터 확인
# enumerate()로 반복 횟수와 데이터를 함께 가져옴
# start=1은 반복 번호를 1부터 시작하도록 설정
for iteration, (xb, yb) in enumerate(loader, start=1):

    # 현재 배치 번호와 입력 데이터(X), 정답 데이터(y)의 크기 출력
    print(
        "iteration:", iteration,
        "X:", tuple(xb.shape),
        "y:", tuple(yb.shape)
    )

# 전체 데이터가 몇 개의 배치로 나뉘었는지 출력
# 데이터 20개를 6개씩 나누므로 총 4번 반복
print("한 epoch의 iteration:", len(loader))
