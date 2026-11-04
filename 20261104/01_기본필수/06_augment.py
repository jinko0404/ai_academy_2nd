"""
[실습 B06: 이미지 반전과 밝기 조절]

1. 이미지 좌우 반전(Flip)
   - cv2.flip()은 이미지의 방향을 뒤집는 함수입니다.
   - flipCode=1 : 좌우 반전
   - flipCode=0 : 상하 반전
   - flipCode=-1 : 상하좌우 모두 반전

2. 이미지 밝기 조절(Brightness)
   - 이미지의 각 픽셀값에 숫자를 더하거나 빼서 밝기를 조절합니다.
   - 양수 값을 더하면 밝아지고, 음수 값을 더하면 어두워집니다.
   - 일반적인 이미지 픽셀값은 0~255 범위를 사용합니다.

3. NumPy 자료형과 uint8
   - OpenCV 이미지는 일반적으로 NumPy 배열로 처리됩니다.
   - uint8은 0~255 범위의 정수만 저장할 수 있는 자료형입니다.
   - uint8 상태에서 255를 초과하거나 0보다 작은 값을 계산하면
     원하는 결과와 달라질 수 있으므로 자료형 변환에 주의해야 합니다.

4. np.clip()
   - 배열의 값이 지정한 범위를 벗어나지 않도록 제한합니다.
   - np.clip(배열, 최소값, 최대값)
   - 밝기 조절 후 픽셀값을 0~255 범위로 유지할 때 사용합니다.

5. 이미지 가로 결합(hconcat)
   - cv2.hconcat()은 여러 이미지를 가로 방향으로 연결합니다.
   - 연결할 이미지들은 높이와 자료형, 채널 수가 같아야 합니다.
   - 이번 실습에서는 원본, 반전 이미지, 밝기 조절 이미지를
     하나의 비교 이미지로 만듭니다.

6. 이미지 저장 및 확인
   - cv2.imwrite() : 이미지 파일 저장
   - image.shape : 이미지의 높이, 너비, 채널 확인
   - image.min() : 이미지에서 가장 작은 픽셀값
   - image.max() : 이미지에서 가장 큰 픽셀값
"""

from pathlib import Path
import os
import cv2
import numpy as np

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
os.chdir(Path(__file__).resolve().parent)

# scene.png 이미지 파일을 불러옵니다.
# OpenCV는 이미지를 NumPy 배열 형태로 읽어옵니다.
image = cv2.imread("scene.png")

# >>> TODO B06
# cv2.flip()을 사용하여 이미지를 반전합니다.
# flipCode=1은 좌우 반전입니다.
# flipCode=0은 상하 반전입니다.
# 원본 image는 유지하고 반전된 결과를 flipped에 저장합니다.
flipped = cv2.flip(image, flipCode=1)
# <<< TODO B06

# 이미지 밝기를 30 증가시킵니다.
# astype(np.float32): uint8에서 발생할 수 있는 픽셀값 범위 문제를 방지하기 위해
# 먼저 실수형(float32)으로 변환합니다.
#
# image + 30: 모든 픽셀 채널에 30을 더합니다.
# np.clip(): 픽셀값이 0~255 범위를 벗어나지 않도록 제한합니다.
# astype(np.uint8): OpenCV 이미지에서 일반적으로 사용하는 자료형으로 다시 변환합니다.
bright = np.clip(image.astype(np.float32) + 30, 0, 255).astype(np.uint8)

# 원본, 좌우 반전, 밝기 조절 이미지를 가로 방향으로 연결합니다.
# hconcat은 Horizontal Concatenation의 약자로 가로 결합을 의미합니다.
# 세 이미지의 높이, 채널 수, 자료형이 같아야 합니다.
comparison = cv2.hconcat([image, flipped, bright])

# 비교 결과 이미지를 PNG 파일로 저장합니다.
# 저장에 성공하면 True, 실패하면 False를 반환합니다.
cv2.imwrite("p06_comparison.png", comparison)

# 이미지의 배치 순서를 출력합니다.
print("왼쪽 원본 / 가운데 좌우반전 / 오른쪽 +30 밝기")

# comparison.shape를 통해 결과 이미지의 크기를 확인합니다.
# (높이, 너비, 채널) 순서로 출력됩니다.
print("비교 그림 Shape:", comparison.shape)

# 밝기 조절 이미지에서 가장 어두운 픽셀값과 가장 밝은 픽셀값을 확인합니다.
# min()은 최솟값, max()는 최댓값을 반환합니다.
# int()는 NumPy 자료형을 일반 Python 정수로 변환합니다.
print("밝기 범위:", int(bright.min()), int(bright.max()))
