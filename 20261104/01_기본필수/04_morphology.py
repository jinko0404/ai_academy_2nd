
"""
[실습 B04] Binary Image와 Morphological Opening

목적:
이미지를 흑백으로 변환하고 이진화한 다음,
Opening 연산을 적용하여 작은 흰색 잡음을 제거합니다.

전체 처리 과정:
원본 이미지 → Grayscale → Threshold → Binary
                                      ↓
                              Opening 연산
                                      ↓
                                  opened

핵심 개념:
- Threshold: 밝기값을 기준으로 픽셀을 흑백으로 구분
- Binary: 0(검정), 255(흰색)으로 구성된 이미지
- Kernel: 형태학적 연산에 사용하는 구조 요소
- Opening: 침식(Erosion) 후 팽창(Dilation)
"""

from pathlib import Path
import os
import cv2
import numpy as np


# 제공 코드:
# 현재 Python 파일이 위치한 폴더를 작업 폴더로 설정합니다.
# 실행 위치와 관계없이 현재 파일 옆에 있는 입력 이미지를 읽을 수 있습니다.
# 원본 이미지는 덮어쓰지 않습니다.
os.chdir(Path(__file__).resolve().parent)


# 원본 이미지 파일을 읽어옵니다.
# OpenCV는 기본적으로 컬러 이미지를 BGR 순서로 읽습니다.
image = cv2.imread("objects.png")


# 컬러 이미지(BGR)를 흑백 이미지(Grayscale)로 변환합니다.
# 색상 정보는 제거하고 픽셀의 밝기 정보만 남깁니다.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Threshold (임계값 처리)
# 특정 밝기값을 기준으로 픽셀을 흰색과 검은색으로 구분합니다.
#
# gray: 입력 흑백 이미지
# 127: 임계값 (Threshold Value)
# 255: 임계값을 초과한 픽셀에 지정할 값
# cv2.THRESH_BINARY: 일반적인 이진화 방식
#
# 밝기가 127보다 크면 255(흰색),
# 127 이하이면 0(검은색)으로 변환합니다.
#
# threshold_value: 실제 사용된 임계값
# binary: 이진화가 적용된 결과 이미지
threshold_value, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)


# Kernel (커널) 생성
# 형태학적 연산에서 주변 픽셀을 검사할 때 사용하는 구조 요소입니다.
#
# np.ones((3, 3)): 3×3 크기의 모든 값이 1인 행렬 생성
# dtype=np.uint8: 픽셀 연산에 사용하는 8비트 부호 없는 정수형
#
# 생성되는 커널:
# [[1, 1, 1],
#  [1, 1, 1],
#  [1, 1, 1]]
kernel = np.ones((3, 3), dtype=np.uint8)


# >>> TODO B04
# Opening (열기) 연산을 적용합니다.
#
# cv2.morphologyEx(이미지, 연산 종류, 커널)
#
# binary: 이진화된 입력 이미지
# cv2.MORPH_OPEN: Opening 연산 지정
# kernel: 3×3 크기의 구조 요소
#
# Opening 처리 순서:
# 1. Erosion (침식): 흰색 영역을 축소하여 작은 흰색 잡음 제거
# 2. Dilation (팽창): 침식된 흰색 영역을 다시 확대
#
# 결과:
# 작은 흰색 잡음이 제거된 이미지가 opened에 저장됩니다.
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# <<< TODO B04


# 흑백 변환 결과를 PNG 파일로 저장합니다.
cv2.imwrite("p04_gray.png", gray)

# 이진화 결과를 PNG 파일로 저장합니다.
cv2.imwrite("p04_binary.png", binary)

# Opening 연산 결과를 PNG 파일로 저장합니다.
cv2.imwrite("p04_open.png", opened)


# 이진화에 사용된 임계값을 출력합니다.
print("임계값:", threshold_value)


# np.count_nonzero()는 배열에서 0이 아닌 값의 개수를 계산합니다.
#
# Binary 이미지에서 흰색은 255, 검은색은 0입니다.
# 따라서 흰색 전경 픽셀의 개수를 확인할 수 있습니다.
#
# binary: Opening 적용 전 흰색 픽셀 수
# opened: Opening 적용 후 흰색 픽셀 수
#
# 두 값을 비교하면 Opening 적용 후 흰색 영역이 얼마나 줄었는지 확인할 수 있습니다.
print(
    "전경 픽셀:",
    int(np.count_nonzero(binary)),
    "->",
    int(np.count_nonzero(opened))
)


# np.unique()는 배열에 존재하는 서로 다른 값만 추출합니다.
#
# 이진 이미지가 정상적으로 처리되었다면 일반적으로 [0, 255]가 출력됩니다.
# .tolist()는 NumPy 배열을 일반 Python 리스트로 변환합니다.
print("결과 값:", np.unique(opened).tolist())

