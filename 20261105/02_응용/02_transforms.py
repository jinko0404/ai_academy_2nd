"""
[Day14 - A02 기본 개념: 소형 source와 공식 ResNet 입력 전처리 비교]

1. 이미지 전처리
   - 모델에 입력하기 전에 이미지 크기, 자료형, 값의 범위를 맞추는 작업
   - 사전학습 당시 사용한 입력 규칙과 일치해야 학습된 특징을 활용할 수 있음
   - 이 파일은 같은 valid 이미지에 두 가지 전처리를 적용하여 비교함
   - 모델 학습이나 예측 성능 평가를 수행하는 코드는 아님

2. tr.Compose()
   - 여러 이미지 변환을 등록한 순서대로 적용하는 클래스
   - 기본 문법: tr.Compose([변환1, 변환2, ...])
   - 소형 source는 Resize 후 ToTensor를 적용함

3. tr.Resize()
   - 이미지 크기를 변경하는 변환
   - tr.Resize((32, 32)): 높이와 너비를 각각 32로 맞춤
   - 공식 전처리의 Resize 256은 짧은 변을 256으로 맞추며 종횡비를 유지함

4. tr.ToTensor()
   - 일반적인 8비트 PIL 이미지를 float Tensor로 변환
   - 이미지 축 순서를 (높이, 너비, 채널)에서 (채널, 높이, 너비)로 바꿈
   - 픽셀값 0~255를 0~1 범위로 변환

5. ResNet18_Weights
   - torchvision에서 제공하는 ResNet-18 사전학습 가중치 선택용 Enum
   - IMAGENET1K_V1: ImageNet 1,000개 클래스 분류로 학습된 가중치 설정
   - Enum 선택과 transforms() 생성만으로 가중치를 다운로드하지 않음
   - 이 파일에서는 모델이나 사전학습 가중치를 실제로 불러오지 않음

6. weights.transforms()
   - 선택한 가중치에 맞는 전처리 객체를 반환하는 메서드
   - IMAGENET1K_V1은 Resize 256 → CenterCrop 224 → Tensor 변환·Normalize 적용
   - 반환된 객체를 official(image)처럼 호출하여 이미지를 변환
   - resize_size, crop_size, mean, std 속성으로 전처리 설정을 확인
   - 이미 구성된 객체이므로 별도의 Compose로 감쌀 필요가 없음
   - Compose로 감싸면 official.mean 같은 속성을 직접 조회할 수 없음

7. CenterCrop과 Normalize
   - CenterCrop: 이미지 중앙을 지정한 크기로 잘라내는 변환
   - Normalize: 채널별로 (픽셀값 - mean) / std를 계산하는 변환
   - mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
   - 위 값은 RGB 순서이며 정규화 이후의 값은 [0, 1] 범위를 벗어날 수 있음
   - 소형 source의 [0, 1] 입력과 ImageNet 정규화를 서로 섞지 않아야 함

8. PIL.Image
   - 이미지 파일을 읽고 변환하는 라이브러리의 모듈
   - Image.open(): 이미지 파일을 열어 PIL 이미지 객체를 반환
   - convert("RGB"): 이미지를 빨강·초록·파랑의 3채널로 변환

9. Path, glob(), sorted()
   - Path: 파일 및 폴더 경로를 관리하는 클래스
   - resolve(): 절대 경로로 변환, parent: 상위 폴더 경로
   - glob("*.png"): 해당 폴더의 PNG 파일 경로를 찾음
   - sorted(): 경로를 정렬, [0]: 정렬된 목록의 첫 번째 항목 선택
   - mkdir(exist_ok=True): 폴더가 없으면 생성하고 이미 있으면 그대로 사용

10. Tensor의 shape와 tuple()
    - shape: Tensor의 각 차원 크기를 나타내는 속성
    - tuple(): 크기 정보를 튜플로 변환하여 출력
    - 소형 source 입력: (3, 32, 32), 공식 ResNet 입력: (3, 224, 224)
    - 이미지 한 장을 변환하므로 배치 차원은 포함하지 않음

11. 실행 설정과 데이터 분할
    - torch.set_num_threads(1): CPU 연산 스레드 수를 1개로 설정
    - torch.manual_seed(14): PyTorch 난수 시드를 고정하여 재현성을 높임
    - 이 파일은 valid 이미지 한 장으로 전처리 결과를 확인하며 test는 읽지 않음
"""

"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""

# 파일 및 폴더 경로를 관리하는 Path 클래스
from pathlib import Path

# JSON 처리 라이브러리이며 현재 파일에서는 사용하지 않음
import json

# 수치 연산 라이브러리이며 현재 파일에서는 사용하지 않음
import numpy as np

# 표 데이터 처리 라이브러리이며 현재 파일에서는 사용하지 않음
import pandas as pd

# Tensor 연산과 실행 환경 설정을 위한 딥러닝 라이브러리
import torch

# 신경망 구성 모듈이며 현재 파일에서는 사용하지 않음
from torch import nn

# 폴더 기반 이미지 데이터셋 클래스이며 현재 파일에서는 사용하지 않음
from torchvision.datasets import ImageFolder

# 이미지 크기 변경 및 Tensor 변환을 위한 모듈
from torchvision import transforms as tr

# 모델 평가 함수들이며 현재 파일에서는 사용하지 않음
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# PyTorch CPU 연산에 사용하는 스레드 수를 1개로 설정
torch.set_num_threads(1)

# 난수 시드를 고정하여 실험의 재현성을 높임
torch.manual_seed(14)

# 현재 파이썬 파일이 위치한 폴더 경로
BASE = Path(__file__).resolve().parent

# 원본 데이터가 저장된 data 폴더 경로
DATA = BASE / "data"

# 결과를 저장할 outputs 폴더 경로
OUT = BASE / "outputs"

# outputs 폴더가 없으면 생성함. 이 파일은 결과 파일을 저장하지는 않음
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.

# 이미지를 32×32로 변경한 후 [0, 1] 범위의 (3, 32, 32) Tensor로 변환
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# ResNet-18 사전학습 가중치 설정과 대응하는 전처리를 제공하는 Enum
from torchvision.models import ResNet18_Weights

# 이미지 파일 읽기와 색상 모드 변환을 위한 모듈
from PIL import Image

# ImageNet 사전학습 가중치 설정을 선택함. 이 선택만으로 다운로드하지 않음
weights = ResNet18_Weights.IMAGENET1K_V1

# TODO A02: Weight가 제공하는 transforms()를 가져오세요.
# BEGIN A02
# 선택한 가중치에 맞는 공식 전처리 객체를 가져옴
official = weights.transforms()
# END A02

# valid/circle 폴더의 PNG 경로를 정렬하여 첫 번째 이미지를 RGB로 읽음
# 해당 폴더에 PNG 이미지가 한 장 이상 있어야 [0]으로 선택할 수 있음
image = Image.open(sorted((DATA/"valid"/"circle").glob("*.png"))[0]).convert("RGB")

# 소형 source 전처리를 적용하고 Tensor 크기를 출력함. 예상: (3, 32, 32)
# "범위 [0,1]"은 전처리 규칙을 표시한 문자열이며 최솟값·최댓값을 측정한 결과는 아님
print("소형 source 입력:",tuple(transform(image).shape),"범위 [0,1]")

# 같은 이미지에 공식 전처리를 적용하고 Tensor 크기를 출력함. 예상: (3, 224, 224)
print("공식 ResNet 입력:",tuple(official(image).shape))

# 공식 전처리의 Resize와 중앙 Crop 설정을 출력함. 예상: [256] [224]
print("공식 Resize/Crop:", official.resize_size,official.crop_size)

# RGB 채널별 정규화에 사용하는 평균과 표준편차를 출력
print("공식 mean/std:", official.mean,official.std)

# 모델의 사전학습 출처에 맞는 입력 규칙을 사용해야 함을 안내
print("작은 source와 ImageNet의 Normalize를 서로 섞지 않습니다.")
