
"""
실습 목적:
하나의 원본 이미지를 서로 다른 보간법으로 축소한 다음,
다시 원래 크기로 확대하여 이미지 품질 차이를 비교한다.

핵심 개념:
- Interpolation: 이미지 크기 변경 시 픽셀값을 계산하는 방법
- INTER_NEAREST: 가장 가까운 픽셀값 사용
- INTER_LINEAR: 주변 픽셀을 이용한 선형 보간
- INTER_AREA: 픽셀 영역을 고려한 보간
- MSE: 원본과 변환 이미지의 평균 제곱 오차
- float32: 이미지 오차 계산을 위한 실수 자료형
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

import csv
import json
from time import perf_counter

# outputs 폴더가 없으면 생성
# exist_ok=True: 이미 폴더가 존재해도 오류가 발생하지 않음
Path("outputs").mkdir(exist_ok=True)

# 원본 이미지 읽기
# OpenCV는 기본적으로 BGR 순서의 NumPy 배열로 이미지를 읽어옴
image = cv2.imread("data/scene.png")

# 각 보간법의 결과와 오차를 저장할 빈 리스트
rows = []

# 세 가지 보간법을 차례대로 적용
# name: 결과 파일 이름에 사용할 문자열
# method: OpenCV에서 제공하는 보간법 상수
for name, method in [
    ("nearest", cv2.INTER_NEAREST),
    ("linear", cv2.INTER_LINEAR),
    ("area", cv2.INTER_AREA)
]:

    # >>> TODO A02
    # 현재 method를 사용해 (너비80, 높이60)으로 축소한 small을 만드세요.

    # 원본 이미지를 80x60 크기로 축소
    # 반복문에서 지정한 보간법을 interpolation에 적용
    small = cv2.resize(image, (80, 60), interpolation=method)

    # <<< TODO A02

    # 축소한 이미지를 다시 320x240 크기로 확대
    # 여기서도 동일한 보간법을 적용
    large = cv2.resize(small, (320, 240), interpolation=method)

    # 축소한 이미지 저장
    # name에 따라 파일 이름이 달라짐
    cv2.imwrite(f"outputs/p02_{name}_small.png", small)

    # 다시 확대된 이미지 저장
    cv2.imwrite(f"outputs/p02_{name}_large.png", large)

    # 원본 이미지와 확대된 이미지의 픽셀 차이 계산
    # astype(np.float32): uint8의 범위 제한을 피하고 실수형으로 계산
    # ** 2: 차이를 제곱
    # np.mean(): 전체 픽셀 차이의 평균 계산
    mse = np.mean(
        (image.astype(np.float32) - large.astype(np.float32)) ** 2
    )

    # 결과 리스트에 보간법 이름, 축소 이미지의 너비와 높이, MSE 저장
    # shape[1]: 너비
    # shape[0]: 높이
    # round(): 소수점 넷째 자리에서 반올림하여 소수점 셋째 자리까지 표시
    # float(): NumPy 자료형을 일반 Python 실수형으로 변환
    rows.append([
        name,
        small.shape[1],
        small.shape[0],
        round(float(mse), 4)
    ])

# CSV 파일 생성 및 저장
# encoding="utf-8-sig": 엑셀에서 한글이 깨지는 현상을 줄이는 인코딩
# newline="": CSV 파일에 불필요한 빈 줄이 생기는 것을 방지
with open(
    "outputs/p02_interpolation.csv",
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    # CSV 파일에 데이터를 기록할 객체 생성
    w = csv.writer(f)

    # 첫 번째 행에 열 이름 기록
    w.writerow(["method", "width", "height", "roundtrip_mse"])

    # rows에 저장된 보간법별 결과를 CSV에 기록
    w.writerows(rows)

# 세 가지 보간법의 결과 출력
print(rows)

# 이번 MSE는 이미지 자체의 차이를 비교하는 수치이며
# AI 모델의 예측 성능이나 학습 성능을 의미하지 않음
print("왕복 MSE는 이 합성 이미지의 수치 비교이며 모델 성능이 아닙니다.")
