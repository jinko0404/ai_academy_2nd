
"""
실습 목적:
OpenCV로 읽어온 컬러 이미지를 Gray와 RGB 형식으로 각각 변환하고,
이미지 크기를 224x224로 변경한다.

핵심 개념:
- BGR: OpenCV에서 사용하는 기본 색상 채널 순서
- Gray: 색상 정보 없이 밝기만 표현하는 흑백 이미지
- RGB: 빨강, 초록, 파랑 순서의 색상 표현
- cv2.cvtColor(): 이미지 색상 형식 변환
- cv2.resize(): 이미지 크기 변경
- INTER_AREA: 이미지 축소에 적합한 보간 방식
"""

from pathlib import Path
import os
import cv2
import numpy as np

# 현재 실행 중인 Python 파일이 위치한 폴더를 작업 폴더로 설정
# __file__: 현재 실행 중인 Python 파일의 경로
# resolve(): 절대 경로로 변환
# parent: 파일이 위치한 폴더
os.chdir(Path(__file__).resolve().parent)

# scene.png 이미지 읽기
# OpenCV는 기본적으로 컬러 이미지를 BGR 순서로 읽어옴
image = cv2.imread("scene.png")

# >>> TODO B02
# image를 Gray와 RGB로 각각 변환하여 gray, rgb에 넣으세요.
# 두 번째 인자는 변환 코드입니다.

# 컬러 이미지(BGR)를 흑백 이미지(Gray)로 변환
# 결과에는 색상 채널 3개가 아닌 밝기 정보만 남음
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 컬러 이미지(BGR)를 RGB 순서로 변환
# 원본 image는 유지하고 변환 결과를 rgb에 저장
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# <<< TODO B02

# 원본 이미지를 224x224 크기로 축소 또는 확대
# (224, 224): (너비, 높이) 순서
# INTER_AREA: 이미지 축소 시 주로 사용하는 보간 방식
small = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

# 변환한 흑백 이미지 저장
cv2.imwrite("p02_gray.png", gray)

# 크기를 변경한 컬러 이미지 저장
cv2.imwrite("p02_small.png", small)

# RGB 배열은 화면 표시용
# OpenCV의 imwrite()에는 BGR 이미지를 전달하는 것이 기본
# 따라서 RGB 배열은 여기서 저장하지 않음

# Gray 이미지의 배열 크기 출력
# 일반적으로 (높이, 너비) 형태로 출력됨
print("Gray shape:", gray.shape)

# RGB 이미지의 30행 30열 픽셀 색상값 출력
# RGB 순서이므로 [R, G, B] 형태
print("RGB 빨강:", rgb[30, 30].tolist())

# 크기 변경한 이미지의 배열 크기 출력
# (높이, 너비, 채널) 형태
print("Resize shape:", small.shape)
