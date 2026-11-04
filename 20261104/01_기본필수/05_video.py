
"""
[실습 B05: 동영상 프레임을 흑백으로 변환하고 저장하기]

핵심 개념
1. VideoCapture : 동영상 파일을 열고 프레임을 읽는 객체
2. VideoWriter  : 프레임을 새로운 동영상 파일로 저장하는 객체
3. FPS          : 1초에 재생되는 프레임 수
4. Frame        : 동영상을 구성하는 이미지 한 장
5. Grayscale    : 컬러 이미지의 색상 정보를 제거하고 밝기 정보만 남기는 것

전체 실행 흐름
원본 동영상 열기
    ↓
FPS와 해상도 확인
    ↓
새 동영상 저장 객체 준비
    ↓
프레임을 하나씩 읽기
    ↓
각 프레임을 흑백으로 변환
    ↓
새 동영상에 프레임 저장
    ↓
객체 해제 및 저장 결과 확인
"""

from pathlib import Path
import os
import cv2
import numpy as np

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
os.chdir(Path(__file__).resolve().parent)

# --------------------------------------------------
# 1. 원본 동영상 열기
# --------------------------------------------------

# VideoCapture: 동영상 파일을 읽기 위한 객체를 생성합니다.
# sample.avi 파일을 열고 프레임을 읽을 준비를 합니다.
cap = cv2.VideoCapture("sample.avi")

# 원본 동영상의 FPS를 가져옵니다.
# FPS는 1초에 재생되는 프레임 수입니다.
fps = cap.get(cv2.CAP_PROP_FPS)

# 원본 동영상의 가로 너비와 세로 높이를 가져옵니다.
# VideoWriter에 사용할 크기는 정수형이어야 하므로 int()로 변환합니다.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --------------------------------------------------
# 2. 새로운 동영상 저장 객체 준비
# --------------------------------------------------

# VideoWriter: 처리한 프레임을 새로운 동영상 파일로 저장합니다.
#
# 인자 순서
# 1. 저장할 파일명
# 2. 사용할 코덱 (MJPG)
# 3. 출력 동영상의 FPS
# 4. 출력 동영상의 크기 (가로, 세로)
#
# 원본과 동일한 FPS와 해상도로 저장하도록 설정합니다.
writer = cv2.VideoWriter(
    "p05_gray.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (width, height)
)

# 동영상 입력과 출력 객체가 정상적으로 열렸는지 확인합니다.
# isOpened()가 True이면 정상적으로 열렸다는 뜻입니다.
print("열림(입력/출력):", cap.isOpened(), writer.isOpened())

# 원본 동영상의 FPS와 해상도를 출력합니다.
print("입력 FPS/크기:", fps, (width, height))

# 처리한 프레임의 개수를 저장할 변수입니다.
count = 0

# --------------------------------------------------
# 3. 프레임을 하나씩 읽어서 흑백으로 변환
# --------------------------------------------------

# 입력 동영상과 출력 동영상이 모두 정상적으로 열려 있는 동안 반복합니다.
while cap.isOpened() and writer.isOpened():

    # 원본 동영상에서 프레임 한 장을 읽습니다.
    #
    # ok    : 프레임 읽기 성공 여부 (True / False)
    # frame : 읽어온 이미지 데이터 (BGR 컬러 이미지)
    ok, frame = cap.read()

    # 읽어온 프레임이 없으면 반복문을 종료합니다.
    # 일반적으로 동영상의 마지막 프레임까지 읽었을 때 False가 반환됩니다.
    if not ok:
        break

    # >>> TODO B05
    # 현재 frame을 Gray로 변환하여 gray에 넣으세요. VideoCapture는 반복문 밖에서 이미 열었습니다.

    # BGR 컬러 이미지를 Grayscale 흑백 이미지로 변환합니다.
    # frame은 컬러 이미지이고, gray는 밝기 정보만 가진 흑백 이미지입니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # <<< TODO B05

    # VideoWriter는 현재 컬러 3채널 영상으로 설정되어 있습니다.
    # gray는 흑백 1채널이므로 GRAY2BGR을 이용해 3채널 형태로 변환합니다.
    #
    # 실제 색상이 추가되는 것은 아닙니다.
    # 세 채널에 동일한 밝기값을 넣어 흑백 화면을 유지합니다.
    writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # 프레임 한 장을 처리했으므로 개수를 1 증가시킵니다.
    count += 1

# --------------------------------------------------
# 4. 동영상 객체 해제
# --------------------------------------------------

# 원본 동영상 읽기 객체를 해제합니다.
# 파일을 닫고 사용하던 자원을 반환합니다.
cap.release()

# 새 동영상 저장 객체를 해제합니다.
# 저장 작업을 마무리하고 파일을 정상적으로 닫습니다.
writer.release()

# --------------------------------------------------
# 5. 저장한 동영상 다시 열어서 결과 확인
# --------------------------------------------------

# 방금 저장한 흑백 동영상을 다시 엽니다.
check = cv2.VideoCapture("p05_gray.avi")

# 실제 처리한 프레임 수와 저장된 동영상의 프레임 수를 비교합니다.
print("처리/재읽기 Frame:", count, int(check.get(cv2.CAP_PROP_FRAME_COUNT)))

# 저장된 동영상의 FPS를 확인합니다.
print("재읽기 FPS:", check.get(cv2.CAP_PROP_FPS))

# 확인 작업이 끝났으므로 다시 연 동영상 객체도 해제합니다.
check.release()
