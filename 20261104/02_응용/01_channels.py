
"""
실습 목적:
OpenCV로 이미지를 읽은 다음 BGR 색상 순서를 RGB로 변환한다.
변환 전후의 이미지를 각각 저장하여 색상 차이를 비교하고,
원본 이미지의 BGR 채널별 평균값을 계산하여 CSV 파일로 저장한다.

핵심 개념:
- BGR: OpenCV에서 사용하는 기본 컬러 채널 순서
- RGB: Matplotlib에서 사용하는 컬러 채널 순서
- cv2.cvtColor(): 이미지의 색상 형식 변환
- mean(): 배열의 평균값 계산
- axis=(0, 1): 이미지의 높이와 너비 방향으로 계산
- CSV: 표 형태의 데이터를 저장하는 파일 형식
"""

from pathlib import Path
import os
import cv2
import numpy as np

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.

# 현재 실행 중인 Python 파일이 위치한 폴더를 작업 폴더로 설정
# __file__: 현재 실행 중인 Python 파일의 경로
# resolve(): 절대 경로로 변환
# parent: 파일이 들어 있는 상위 폴더
os.chdir(Path(__file__).resolve().parent)

import csv
import json
from time import perf_counter

# outputs 폴더가 없으면 생성
# exist_ok=True: 폴더가 이미 존재해도 오류가 발생하지 않음
Path("outputs").mkdir(exist_ok=True)

import matplotlib

# Matplotlib이 화면에 직접 표시하지 않고 이미지 파일로 저장하도록 설정
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# data 폴더에 있는 scene.png 이미지 읽기
# OpenCV는 이미지를 BGR 순서의 NumPy 배열로 읽어옴
image = cv2.imread("data/scene.png")

# >>> TODO A01
# Matplotlib 표시 전에 BGR을 RGB로 변환하세요. 원본 image는 그대로 유지합니다.

# 원본 image는 그대로 두고, RGB로 변환한 결과를 별도 변수에 저장
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# <<< TODO A01

# 각각 별도의 Figure입니다. 색공간 순서의 오류를 직접 비교합니다.

# 첫 번째 이미지: 원본 BGR 이미지를 RGB로 해석하여 표시
# 색상 순서가 맞지 않으므로 실제 색상과 다르게 보일 수 있음
plt.figure(figsize=(5, 4))
plt.imshow(image)
plt.title("Wrong: BGR interpreted as RGB")
plt.axis("off")  # X축과 Y축 눈금 숨기기

# 여백을 정리한 다음 이미지 파일로 저장
plt.tight_layout()
plt.savefig("outputs/p01_wrong.png", dpi=120)
plt.close()  # 첫 번째 Figure 닫기

# 두 번째 이미지: BGR에서 RGB로 변환한 이미지 표시
plt.figure(figsize=(5, 4))
plt.imshow(rgb)
plt.title("Correct: RGB display")
plt.axis("off")

# 여백을 정리한 다음 이미지 파일로 저장
plt.tight_layout()
plt.savefig("outputs/p01_correct.png", dpi=120)
plt.close()

# 원본 이미지의 BGR 채널별 평균값 계산
# image의 형태는 (높이, 너비, 채널)
# axis=(0, 1)은 높이와 너비 방향으로 평균을 계산한다는 의미
# 결과에는 B, G, R 채널의 평균값이 각각 들어감
means = image.mean(axis=(0, 1))

# CSV 파일 생성 및 저장
# encoding="utf-8-sig": 엑셀에서 한글이 깨지는 현상을 줄이는 인코딩
# newline="": CSV 저장 시 불필요한 빈 줄이 생기는 것을 방지
with open("outputs/p01_channels.csv", "w", encoding="utf-8-sig", newline="") as f:

    # CSV 파일에 데이터를 기록할 객체 생성
    w = csv.writer(f)

    # 첫 번째 행에 열 이름 기록
    w.writerow(["channel", "mean"])

    # B, G, R 채널 이름과 각각의 평균값을 묶어서 기록
    w.writerows(zip(["B", "G", "R"], means))

# BGR 채널별 평균값 출력
# np.round(means, 3): 소수점 셋째 자리까지 반올림
# tolist(): NumPy 배열을 일반 Python 리스트로 변환
print("BGR 채널평균:", np.round(means, 3).tolist())

# RGB로 변환한 이미지에서 30행 30열 픽셀의 색상값 출력
# rgb는 RGB 순서이므로 출력값도 [R, G, B]
print("변환후 빨강:", rgb[30, 30].tolist())
