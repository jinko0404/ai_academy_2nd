"""
[Day14 - B04 기본 개념: 클래스 불균형과 Macro F1]

1. 클래스 불균형
   - 클래스마다 데이터 개수가 서로 다른 상태
   - 이 예제의 정답은 circle 12개, square 6개, triangle 3개로 구성됨
   - 많은 클래스만 예측해도 Accuracy가 어느 정도 높게 나올 수 있음
   - 소수 클래스의 성능을 확인하려면 클래스별 Recall과 Macro F1도 함께 봐야 함

2. y_true와 y_pred
   - y_true: 실제 정답 클래스 번호 목록
   - y_pred: 예측 클래스 번호 목록
   - 같은 위치의 정답과 예측이 같은 샘플을 나타내며 두 목록의 길이는 같아야 함
   - 이 파일은 지표 이해를 위한 수작업 예제이며 실제 모델 예측이 아님

3. 리스트 반복과 연결
   - [0] * 12: 값 0이 12개 들어 있는 리스트 생성
   - 리스트 + 리스트: 두 리스트를 순서대로 연결
   - [0] * 12 + [1] * 6 + [2] * 3: 총 21개의 정답 생성
   - [0] * 21: 모든 샘플을 circle로 예측하는 상황 구성

4. count()와 리스트 컴프리헨션
   - 리스트.count(값): 해당 값이 등장하는 횟수를 계산
   - range(3): 0, 1, 2를 순서대로 생성
   - [y_true.count(i) for i in range(3)]: 클래스별 개수를 리스트로 계산

5. Accuracy
   - 전체 샘플 중 정답을 맞힌 비율
   - accuracy_score(y_true, y_pred): 정답과 예측을 비교하여 정확도 계산
   - 이 예제에서는 21개 중 circle 12개만 맞혀 12/21 ≈ 0.5714
   - Accuracy만 보면 square와 triangle을 전혀 맞히지 못한 문제가 가려질 수 있음

6. Precision, Recall, F1
   - 특정 클래스를 양성으로 보고 나머지 클래스를 음성으로 취급하여 계산
   - TP: 해당 클래스를 올바르게 예측한 수
   - FP: 다른 클래스를 해당 클래스로 잘못 예측한 수
   - FN: 해당 클래스를 다른 클래스로 잘못 예측한 수
   - Precision = TP / (TP + FP): 해당 클래스로 예측한 것 중 실제 정답 비율
   - Recall = TP / (TP + FN): 실제 해당 클래스 중 올바르게 예측한 비율
   - F1 = 2×TP / (2×TP + FP + FN): Precision과 Recall의 조화평균

7. f1_score()와 Macro F1
   - 기본 문법: f1_score(y_true, y_pred, average, labels, zero_division)
   - average="macro": 클래스별 F1을 구한 뒤 동일한 비중으로 평균
   - labels=[0,1,2]: 평균에 포함할 클래스를 명시
   - zero_division=0: 분모가 0인 지표를 0으로 처리
   - circle F1은 24/33 ≈ 0.7273이며 square와 triangle F1은 각각 0
   - Macro F1 = (0.7273 + 0 + 0) / 3 ≈ 0.2424

8. classification_report()
   - 클래스별 Precision, Recall, F1, support와 요약 지표를 출력
   - support: 정답에 포함된 해당 클래스의 샘플 수
   - labels와 target_names는 같은 순서로 대응해야 함
   - digits=3: 보고서의 지표를 소수점 셋째 자리까지 표시
   - macro avg: 클래스별 동일 비중 평균
   - weighted avg: 정답 클래스별 support로 가중한 평균

9. round()와 출력
   - round(숫자, 4): 소수점 넷째 자리까지 반올림
   - print(): 계산 결과를 화면에 출력
   - round()는 출력용 반올림이며 원래 macro 변수의 값을 변경하지 않음

10. 공통 준비 코드
    - Path: 경로 관리, resolve(): 절대 경로 변환, parent: 상위 폴더
    - torch.set_num_threads(1): CPU 연산 스레드 수를 1개로 설정
    - torch.manual_seed(14): PyTorch 난수 시드 고정
    - Compose, Resize, ToTensor: 소형 source의 이미지 전처리 정의
    - 이 파일은 실제 이미지를 읽거나 모델을 학습하지 않으며 전처리도 적용하지 않음
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""

# 파일 및 폴더 경로를 관리하는 클래스
from pathlib import Path

# JSON 처리 라이브러리이며 현재 파일에서는 사용하지 않음
import json

# 수치 연산 라이브러리이며 현재 파일에서는 사용하지 않음
import numpy as np

# 표 데이터 처리 라이브러리이며 현재 파일에서는 사용하지 않음
import pandas as pd

# 공통 실행 환경 설정에 사용하는 딥러닝 라이브러리
import torch

# 신경망 구성 모듈이며 현재 파일에서는 사용하지 않음
from torch import nn

# 폴더 기반 이미지 데이터셋 클래스이며 현재 파일에서는 사용하지 않음
from torchvision.datasets import ImageFolder

# 공통 이미지 전처리를 정의하는 모듈
from torchvision import transforms as tr

# 정확도, F1, 클래스별 보고서 계산에 사용함. confusion_matrix는 현재 사용하지 않음
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# PyTorch CPU 연산에 사용하는 스레드 수를 1개로 설정
torch.set_num_threads(1)

# PyTorch 난수 시드 고정. 아래 수작업 지표 예제에는 난수가 사용되지 않음
torch.manual_seed(14)

# 현재 소스 파일이 위치한 폴더 경로
BASE = Path(__file__).resolve().parent

# 공통 데이터 경로를 지정하지만 이 파일에서는 데이터를 읽지 않음
DATA = BASE / "data"

# 공통 출력 경로를 지정하지만 이 파일에서는 파일을 저장하지 않음
OUT = BASE

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# 이미지 크기를 32×32로 맞추고 Tensor로 변환하는 공통 전처리 정의
# 이 파일에서는 실제 이미지에 적용하지 않음
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# 손으로 읽는 지표 예제입니다. 모델의 실제 예측이라고 주장하지 않습니다.
# 정답을 circle(0) 12개, square(1) 6개, triangle(2) 3개 순서로 생성
y_true = [0] * 12 + [1] * 6 + [2] * 3

# 모든 샘플을 가장 많은 클래스인 circle(0)로 예측한 상황 구성
y_pred = [0] * 21

# 정답 목록에서 0, 1, 2의 개수를 각각 세어 출력. 예상: [12, 6, 3]
print("클래스 수:", [y_true.count(i) for i in range(3)])

# 전체 정답 비율을 소수점 넷째 자리까지 반올림하여 출력. 예상: 0.5714
print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))

# TODO B04: f1_score에 average="macro", labels=[0,1,2], zero_division=0을 넣으세요.
# BEGIN B04
# 세 클래스의 F1을 동일 비중으로 평균하고 분모가 0인 경우는 0으로 처리
macro = f1_score(y_true, y_pred, average="macro", labels=[0,1,2], zero_division=0)
# END B04

# Macro F1을 소수점 넷째 자리까지 반올림하여 출력. 예상: 0.2424
print("Macro F1:", round(macro, 4))

# 0, 1, 2에 대응하는 클래스 이름과 함께 상세 지표를 출력
# square와 triangle의 Recall은 0이므로 소수 클래스를 놓치는 문제를 확인할 수 있음
print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=["circle", "square", "triangle"], zero_division=0, digits=3))
