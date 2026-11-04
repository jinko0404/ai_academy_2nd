
"""
[실습 B03] Gaussian Blur와 Canny Edge Detection

목적:
이미지에 포함된 잡음을 Gaussian Blur로 줄인 다음,
Canny 알고리즘을 이용해 물체의 윤곽선을 검출합니다.

전체 처리 과정:
원본 이미지 → 흑백 변환 → 블러 처리 → 윤곽선 검출 → 이미지 저장

핵심 개념:
- Gaussian Blur: 주변 픽셀을 이용해 이미지를 부드럽게 처리
- Kernel (5, 5): 블러 처리에 참고할 주변 영역의 크기
- Canny: 이미지의 경계선 검출
- Threshold (80, 160): 윤곽선 검출에 사용하는 하한값과 상한값
"""

from pathlib import Path
import os
import cv2
import numpy as np


# 제공 코드:
# 현재 Python 파일이 위치한 폴더를 작업 폴더로 설정합니다.
# 이렇게 하면 다른 위치에서 코드를 실행해도 현재 파일 옆의 이미지를 읽을 수 있습니다.
# 원본 파일은 수정하지 않고 결과 이미지를 별도로 저장합니다.
os.chdir(Path(__file__).resolve().parent)


# 이미지 파일을 읽어옵니다.
# OpenCV는 컬러 이미지를 기본적으로 BGR 형식으로 읽습니다.
image = cv2.imread("noisy.png")


# 컬러 이미지(BGR)를 흑백 이미지(Grayscale)로 변환합니다.
# 흑백 이미지에는 색상 정보 대신 밝기 정보만 남습니다.
# 윤곽선 검출을 위해 먼저 흑백 이미지로 변환합니다.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# >>> TODO B03
# Gaussian Blur (가우시안 블러)
# 흑백 이미지 gray에 블러 처리를 적용합니다.
#
# (5, 5): 블러 처리에 사용할 커널 크기입니다.
#         주변 5×5 영역의 픽셀을 참고합니다.
#
# sigmaX=0: 가로 방향 표준편차를 OpenCV가 자동으로 계산합니다.
#
# 결과:
# 이미지의 작은 잡음이나 불필요한 세부 정보가 줄어듭니다.
blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0)


# Canny Edge Detection (캐니 윤곽선 검출)
# 블러 처리된 이미지 blur에서 물체의 경계선을 검출합니다.
#
# 80: 하한 임계값 (Low Threshold)
# 160: 상한 임계값 (High Threshold)
#
# 두 임계값을 기준으로 경계선 후보를 판단하고,
# 불필요한 경계선을 줄여 최종 윤곽선을 만듭니다.
#
# 결과:
# 윤곽선은 흰색(255), 배경은 검은색(0)으로 표현됩니다.
edge = cv2.Canny(blur, 80, 160)
# <<< TODO B03


# 블러 처리 결과를 PNG 파일로 저장합니다.
cv2.imwrite("p03_blur.png", blur)

# 윤곽선 검출 결과를 PNG 파일로 저장합니다.
cv2.imwrite("p03_edge.png", edge)


# 사용한 블러 및 Canny 설정을 출력합니다.
print("설정: Gaussian (5,5), sigmaX=0, Canny 80/160")


# 이미지 배열의 크기(shape)를 출력합니다.
# 흑백 이미지이므로 (높이, 너비) 형태로 출력됩니다.
print("Shape:", blur.shape, edge.shape)


# 윤곽선 이미지에서 0이 아닌 픽셀의 개수를 계산합니다.
# Canny 결과에서 흰색(255)으로 표시된 윤곽선 픽셀의 개수입니다.
# int()는 NumPy 정수형 값을 일반 Python 정수형으로 변환합니다.
print("Edge 픽셀:", int(np.count_nonzero(edge)))

