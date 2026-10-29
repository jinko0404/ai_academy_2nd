"""실습 08 추가: pytest 없이 표준 unittest로 두 검증을 직접 완성합니다."""

import sys
from pathlib import Path
import unittest
import copy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 현재 테스트 파일의 상위 폴더를 Python 모듈 검색 경로에 추가
# 상위 폴더에 있는 model.py와 train.py를 불러오기 위해 사용
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 직접 만든 신경망 모델과 학습 및 평가 함수 가져오기
from model import SimpleANN
from train import train_one_epoch, evaluate


class TrainingChecks(unittest.TestCase):
    # 각 테스트를 실행하기 전에 필요한 환경과 데이터를 준비
    def setUp(self):
        # 난수 결과를 일정하게 유지해 테스트 결과를 재현할 수 있도록 설정
        torch.manual_seed(42)

        # CPU 연산에 사용할 스레드 수를 1개로 설정
        torch.set_num_threads(1)

        # 테스트에 사용할 신경망 모델 생성
        self.model = SimpleANN()

        # 입력 데이터 7건: 각 데이터는 숫자 2개로 구성
        self.X = torch.tensor([
            [1., 2.],
            [-1., 0.],
            [2., 1.],
            [0., 0.],
            [1., -1.],
            [-2., 1.],
            [3., 2.]
        ])

        # 입력 데이터 각각에 대응하는 실제 정답
        self.y = torch.tensor([1, 0, 1, 0, 1, 0, 1])

    def test_last_batch_weighting(self):
        # 입력 데이터 X와 실제 정답 y를 하나의 데이터셋으로 묶음
        # 전체 7건을 batch_size=3으로 나누면 3, 3, 1건이 됨
        loader = DataLoader(
            TensorDataset(self.X, self.y),
            batch_size=3,
            shuffle=False  # 데이터 순서를 섞지 않음
        )

        # 모델의 예측과 실제 정답을 비교해 손실을 계산하는 함수
        loss_fn = nn.CrossEntropyLoss()

        # 가중치를 업데이트하는 최적화 도구 생성
        # lr=0.0은 학습률이 0이므로 가중치가 변경되지 않음
        # 이번 테스트에서는 배치 분할에 따른 손실 계산만 비교
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0)

        # 전체 7건을 한 번에 모델에 넣어 손실을 계산
        # 배치로 나누지 않은 손실을 비교 기준 expected에 저장
        # item()은 텐서에서 일반 파이썬 숫자를 꺼냄
        expected = loss_fn(self.model(self.X), self.y).item()

        # 데이터를 3, 3, 1건으로 나눠 학습 함수를 실행
        # 함수가 반환한 손실을 actual에 저장
        actual = train_one_epoch(
            self.model,
            loader,
            loss_fn,
            optimizer,
            torch.device("cpu")
        )

        # H11 검증:
        # 배치별 손실을 합칠 때 각 배치의 데이터 개수를 올바르게 반영했는지 확인
        # 전체 데이터를 한 번에 계산한 손실과 배치별 계산 결과가 같아야 함
        # places=5는 소수점 아래 5자리까지 비교한다는 뜻
        self.assertAlmostEqual(expected, actual, places=5)

    def test_evaluation_does_not_learn(self):
        # 모델의 출력값을 모두 더한 뒤 역전파 실행
        # 각 파라미터에 비교할 gradient를 만들어 둠
        self.model(self.X).sum().backward()

        # 평가 전 모델의 가중치와 편향을 복사해 보관
        # 평가 후 값과 비교하기 위해 별도로 복사함
        before = copy.deepcopy(self.model.state_dict())

        # 평가 전 각 파라미터의 gradient도 복사해 보관
        grads = [p.grad.clone() for p in self.model.parameters()]

        # 모델 평가 실행
        # loss: 손실값, accuracy: 정확도, prediction: 예측 결과
        loss, accuracy, prediction = evaluate(
            self.model,
            self.X,
            self.y,
            nn.CrossEntropyLoss(),
            torch.device("cpu")
        )

        # H12 검증 1:
        # 평가가 끝난 현재 모델의 가중치와 편향을 가져옴
        after = self.model.state_dict()

        # 평가 전과 평가 후의 gradient를 하나씩 비교
        # p는 현재 파라미터, old_grad는 평가 전 gradient
        # 두 값이 같으면 평가 과정에서 gradient가 변경되지 않은 것
        for p, old_grad in zip(self.model.parameters(), grads):
            self.assertTrue(torch.equal(old_grad, p.grad))

        # H12 검증 2:
        # 예측 결과가 gradient 추적 대상이 아닌지 확인
        # requires_grad가 False이면 gradient를 추적하지 않음
        self.assertFalse(prediction.requires_grad)

        # H12 검증 3:
        # 평가 전과 평가 후의 각 파라미터 값을 이름별로 비교
        for name, old_value in before.items():
            # 같은 이름을 가진 평가 후 파라미터를 가져옴
            new_value = after[name]

            # 평가 전후의 파라미터 값이 정확히 같은지 검사
            self.assertTrue(torch.equal(old_value, new_value))


# 이 파일을 직접 실행했을 때만 테스트 실행
if __name__ == "__main__":
    # 등록된 테스트를 실행하고 결과를 자세하게 출력
    unittest.main(verbosity=2)
