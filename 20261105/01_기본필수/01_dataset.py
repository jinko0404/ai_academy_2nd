"""
[Day14 - B01 기본 개념: ImageFolder]

1. ImageFolder란?
   - torchvision에서 제공하는 이미지 데이터셋 클래스
   - 폴더 이름을 기준으로 클래스와 정답 라벨을 자동 생성함

2. ImageFolder 기본 문법
   - ImageFolder(root=폴더경로, transform=변환방법)
   - root: 이미지 폴더 경로 (필수)
   - transform: 이미지 전처리 방법 (선택)

3. 주요 속성
   - classes: 클래스 이름 목록
   - class_to_idx: 클래스 이름과 숫자 라벨의 매핑 정보
   - targets: 이미지별 정답 라벨 목록
   - samples: (이미지 경로, 정답 라벨) 목록

4. Path
   - 파일 및 폴더 경로를 관리하는 클래스
   - __file__: 현재 파이썬 파일 경로
   - resolve(): 절대 경로로 변환
   - parent: 상위 폴더 경로
   - / 연산자: 폴더 경로 연결

5. transform
   - Resize((32, 32)): 이미지 크기를 32×32로 변경
   - ToTensor(): 이미지를 Tensor로 변환
   - 일반적인 8비트 RGB 이미지의 픽셀값을 [0,1]로 변환

6. 데이터셋 분리
   - train: 모델 학습용 데이터
   - valid: 모델 검증용 데이터
   - test: 최종 성능 평가용 데이터
   - 세 데이터셋의 클래스 매핑이 일치하는지 확인해야 함
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""

# 파일 및 폴더 경로를 관리하는 Path 클래스
from pathlib import Path

# JSON 데이터 처리 라이브러리
import json

# 수치 연산 라이브러리
import numpy as np

# 표 형태의 데이터 처리 라이브러리
import pandas as pd

# 딥러닝 연산 라이브러리
import torch

# 신경망 구성에 사용하는 모듈
from torch import nn

# 폴더 구조를 기준으로 이미지와 정답 라벨을 불러오는 클래스
from torchvision.datasets import ImageFolder

# 이미지 크기 변경 및 Tensor 변환을 위한 모듈
from torchvision import transforms as tr

# 모델 평가 지표를 계산하는 함수들
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# PyTorch CPU 연산에 사용하는 스레드 수를 1개로 설정
torch.set_num_threads(1)

# 난수 시드를 고정하여 실험의 재현성을 높임
torch.manual_seed(14)

# 현재 파이썬 파일이 위치한 폴더 경로
BASE = Path(__file__).resolve().parent

# 이미지 데이터가 저장된 data 폴더 경로
DATA = BASE / "data"

# 결과 파일을 저장할 기본 경로
OUT = BASE

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.

# 이미지 크기를 32×32로 변경한 후 Tensor로 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])


# TODO B01: ImageFolder로 train 폴더를 읽으세요. root는 필수 경로입니다.
# BEGIN B01

# DATA 폴더 안의 train 폴더에서 이미지 데이터셋을 불러옴
# 하위 폴더 이름을 기준으로 클래스와 숫자 라벨을 자동 생성
train = ImageFolder(DATA / "train")  # train 폴더로 바꾸세요.

# END B01


# 학습 데이터의 클래스 이름과 숫자 라벨 매핑 출력
print("class_to_idx:", train.class_to_idx)

# train, valid, test 데이터셋을 순서대로 확인
for split in ["train", "valid", "test"]:

    # 현재 데이터셋 폴더에서 이미지 경로와 정답 라벨을 불러옴
    dataset = ImageFolder(DATA / split)

    # 클래스 0, 1, 2에 해당하는 이미지 개수를 각각 계산
    counts = [dataset.targets.count(i) for i in range(3)]

    # 데이터셋별 이미지 개수와 클래스 매핑 일치 여부 출력
    # True: train과 클래스 매핑이 동일함
    # False: train과 클래스 매핑이 다름
    print(split, "개수:", counts, "같은 매핑:", dataset.class_to_idx == train.class_to_idx)

# test는 파일명·라벨 매핑만 확인했습니다. 모델 예측에는 사용하지 않습니다.