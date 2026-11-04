"""
[실습 B07~B09: 이미지 전처리 및 동영상 엣지 검출]

1. 이미지 전처리(Image Preprocessing)
   - 모델이나 영상 처리 작업을 수행하기 전에 데이터를 원하는 형태로 가공하는 과정입니다.
   - 이번 실습에서는 컬러 이미지를 흑백으로 변환하고 크기를 축소합니다.
   - 이미지의 크기와 채널을 통일하면 후속 처리 작업이 편리해집니다.

2. Grayscale(그레이스케일)
   - 컬러 이미지의 색상 정보를 제거하고 밝기 정보만 남긴 흑백 이미지입니다.
   - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)를 사용합니다.
   - BGR 3채널 이미지가 Gray 1채널 이미지로 변환됩니다.
   - 색상보다 밝기와 윤곽선이 중요한 영상 처리 작업에 사용합니다.

3. Resize(이미지 크기 조절)
   - cv2.resize()를 사용하여 이미지의 너비와 높이를 변경합니다.
   - cv2.resize(image, (너비, 높이)) 형식으로 사용합니다.
   - 크기를 직접 지정할 때는 너비가 먼저, 높이가 다음입니다.
   - 이미지 크기를 통일하면 연산량을 줄이고 처리하기 쉬워집니다.

4. Canny Edge Detection(캐니 엣지 검출)
   - 이미지에서 물체의 윤곽선이나 경계선을 검출하는 알고리즘입니다.
   - cv2.Canny(image, threshold1, threshold2)를 사용합니다.
   - threshold1: 낮은 임계값
   - threshold2: 높은 임계값
   - 픽셀의 밝기 변화가 큰 경계 영역을 찾아냅니다.
   - 결과는 일반적으로 검은 배경에 흰색 윤곽선이 표시된 이미지입니다.

5. Path와 파일 이름 처리
   - Path(filename).stem: 파일 이름에서 확장자를 제외한 이름을 가져옵니다.
   - 예: scene.png → scene
   - 파일 이름을 조합하여 여러 입력 이미지의 결과를 구분해서 저장할 수 있습니다.

6. NumPy 배열과 픽셀 개수
   - 이미지도 NumPy 배열로 처리됩니다.
   - np.count_nonzero(edge): 배열에서 0이 아닌 값의 개수를 계산합니다.
   - Canny 결과에서는 흰색으로 검출된 엣지 픽셀의 개수를 확인할 수 있습니다.

7. VideoCapture(동영상 읽기)
   - cv2.VideoCapture()는 동영상 파일을 읽기 위한 객체를 생성합니다.
   - cap.read(): 동영상에서 프레임 하나를 읽습니다.
   - 반환값은 (성공 여부, 프레임 이미지)입니다.
   - 성공 여부가 False이면 읽을 프레임이 없거나 읽기에 실패한 것입니다.
   - FPS(Frame Per Second)는 1초에 재생되는 프레임 수입니다.

8. VideoWriter(동영상 저장)
   - cv2.VideoWriter()는 처리한 프레임을 동영상 파일로 저장합니다.
   - VideoWriter_fourcc(): 동영상 압축 코덱을 지정합니다.
   - MJPG: Motion JPEG 방식의 코덱입니다.
   - FPS: 저장할 영상의 초당 프레임 수입니다.
   - (160, 120): 저장할 영상의 너비와 높이입니다.
   - writer.write(): 처리한 프레임을 영상에 기록합니다.

9. Gray와 BGR 채널 변환
   - Gray 이미지는 1채널, BGR 이미지는 3채널입니다.
   - cv2.COLOR_GRAY2BGR: 흑백 이미지를 3채널 BGR 이미지로 변환합니다.
   - VideoWriter에 프레임을 전달할 때 지정한 형식과 채널을 맞춰야 합니다.

10. CSV 파일 저장
    - CSV(Comma-Separated Values)는 쉼표로 데이터를 구분하여 저장하는 형식입니다.
    - csv.writer(): CSV 데이터를 기록하기 위한 객체를 생성합니다.
    - writerow(): 한 줄의 데이터를 기록합니다.
    - writerows(): 여러 줄의 데이터를 한 번에 기록합니다.
    - 이미지 처리 결과의 파일명, 크기, 엣지 픽셀 수 등을 표 형태로 저장할 수 있습니다.

11. 리소스 해제와 결과 검증
    - cap.release(): 동영상 읽기 객체의 리소스를 해제합니다.
    - writer.release(): 동영상 저장 객체의 리소스를 해제합니다.
    - check.get(): 저장된 동영상의 프레임 수와 FPS 등을 확인합니다.
    - 저장 성공 메시지만 확인하는 것이 아니라 실제 파일을 다시 읽어 결과를 검증합니다.
"""

from pathlib import Path
import os
import cv2
import numpy as np

# 제공 코드: 어느 작업 폴더에서 실행해도 현재 파일 옆의 입력을 읽습니다.
# OpenCV에는 영문 상대 경로를 전달합니다. 원본 파일은 덮어쓰지 않습니다.
# 현재 Python 파일이 위치한 폴더를 작업 디렉터리로 변경합니다.
os.chdir(Path(__file__).resolve().parent)

import csv

# 결과 이름의 p07_ 접두어가 입력 파일 덮어쓰기를 방지합니다.
# 여러 이미지의 처리 결과를 저장할 목록입니다.
# 각 이미지의 파일명, 원본 크기, 축소 크기, 엣지 픽셀 수를 기록합니다.
rows = []

# 처리할 이미지 파일 3개를 순회합니다.
# 반복문을 사용하면 동일한 이미지 처리 작업을 여러 파일에 적용할 수 있습니다.
for filename in ["scene.png", "noisy.png", "objects.png"]:

    # 이미지 파일을 불러옵니다.
    # OpenCV는 이미지를 BGR 순서의 NumPy 배열로 읽어옵니다.
    image = cv2.imread(filename)

    # >>> TODO B07
    # image를 Gray로 만든 뒤 (너비160, 높이120)으로 축소하세요. 변수는 gray, small입니다.

    # 컬러 이미지(BGR)를 흑백 이미지(Gray)로 변환합니다.
    # COLOR_BGR2GRAY는 BGR 3채널을 Gray 1채널로 변환하는 옵션입니다.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 흑백 이미지의 크기를 너비 160px, 높이 120px로 축소합니다.
    # cv2.resize()의 크기 인자는 (너비, 높이) 순서입니다.
    small = cv2.resize(gray, (160, 120))

    # <<< TODO B07

    # >>> TODO B08
    # 축소한 small에 Canny 80,160을 적용하여 edge를 만드세요.

    # 축소한 흑백 이미지에 Canny 엣지 검출을 적용합니다.
    # 80은 낮은 임계값, 160은 높은 임계값입니다.
    # 검출된 윤곽선은 edge 변수에 저장됩니다.
    edge = cv2.Canny(small, 80, 160)

    # <<< TODO B08

    # Path(filename).stem은 확장자를 제외한 파일 이름을 가져옵니다.
    # 예: scene.png → scene
    stem = Path(filename).stem

    # 처리된 흑백 이미지를 PNG 파일로 저장합니다.
    cv2.imwrite(f"p07_{stem}_gray.png", gray)

    # 크기를 축소한 이미지를 PNG 파일로 저장합니다.
    cv2.imwrite(f"p07_{stem}_small.png", small)

    # Canny 엣지 검출 결과를 PNG 파일로 저장합니다.
    cv2.imwrite(f"p07_{stem}_edge.png", edge)

    # 현재 이미지의 처리 결과를 목록에 추가합니다.
    # image.shape[1]: 원본 이미지 너비
    # image.shape[0]: 원본 이미지 높이
    # small.shape[1]: 축소 이미지 너비
    # small.shape[0]: 축소 이미지 높이
    # np.count_nonzero(edge): 검출된 흰색 엣지 픽셀 개수
    rows.append([
        filename,
        image.shape[1],
        image.shape[0],
        small.shape[1],
        small.shape[0],
        int(np.count_nonzero(edge))
    ])


# --------------------------------------------------
# 동영상 처리
# --------------------------------------------------

# sample.avi 동영상 파일을 읽기 위한 객체를 생성합니다.
cap = cv2.VideoCapture("sample.avi")

# 원본 동영상의 FPS를 가져옵니다.
# FPS는 1초 동안 재생되는 프레임 수를 의미합니다.
fps = cap.get(cv2.CAP_PROP_FPS)

# 처리된 프레임을 저장할 동영상 객체를 생성합니다.
#
# 인자 설명:
# 1. "p07_video.avi": 저장할 동영상 파일명
# 2. cv2.VideoWriter_fourcc(*"MJPG"): 사용할 압축 코덱
# 3. fps: 저장할 동영상의 초당 프레임 수
# 4. (160, 120): 저장할 프레임의 너비와 높이
writer = cv2.VideoWriter(
    "p07_video.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (160, 120)
)

# VideoWriter 객체가 정상적으로 생성되었는지 확인합니다.
# True이면 동영상 저장 준비가 완료된 것입니다.
print("Writer:", writer.isOpened())

# 실제로 처리한 동영상 프레임 수를 기록할 변수입니다.
count = 0

# 동영상 읽기 객체와 저장 객체가 모두 정상적으로 열려 있는 동안 반복합니다.
while cap.isOpened() and writer.isOpened():

    # 동영상에서 프레임 하나를 읽어옵니다.
    # ok: 프레임 읽기 성공 여부
    # frame: 읽어온 프레임 이미지
    ok, frame = cap.read()

    # 프레임을 읽지 못했다면 반복문을 종료합니다.
    # 동영상의 마지막 프레임에 도달한 경우에도 False가 반환됩니다.
    if not ok:
        break

    # >>> TODO B09
    # 각 frame에도 같은 Gray→160×120 Resize→Canny 순서를 적용하세요. 마지막 변수는 frame_edge입니다.

    # 현재 동영상 프레임을 흑백 이미지로 변환합니다.
    # 컬러 프레임의 BGR 3채널을 Gray 1채널로 변환합니다.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 흑백 프레임의 크기를 너비 160px, 높이 120px로 축소합니다.
    frame_small = cv2.resize(frame_gray, (160, 120))

    # 축소된 프레임에 Canny 엣지 검출을 적용합니다.
    # 80은 낮은 임계값, 160은 높은 임계값입니다.
    frame_edge = cv2.Canny(frame_small, 80, 160)

    # <<< TODO B09

    # Canny 결과는 Gray 1채널 이미지입니다.
    # VideoWriter에 전달할 3채널 BGR 이미지로 변환합니다.
    # 변환한 프레임을 출력 동영상에 기록합니다.
    writer.write(cv2.cvtColor(frame_edge, cv2.COLOR_GRAY2BGR))

    # 프레임 하나의 처리가 끝났으므로 카운트를 1 증가시킵니다.
    count += 1

# 동영상 읽기 객체의 리소스를 해제합니다.
# 파일을 정상적으로 닫고 사용하던 자원을 반환합니다.
cap.release()

# 동영상 저장 객체의 리소스를 해제합니다.
# 남아 있는 영상 데이터를 파일에 마무리하여 기록합니다.
writer.release()


# --------------------------------------------------
# CSV 결과 저장
# --------------------------------------------------

# CSV 파일을 쓰기 모드로 엽니다.
# encoding="utf-8-sig": Excel 등에서 한글이 깨지는 문제를 줄이기 위한 인코딩입니다.
# newline="": CSV 작성 시 불필요한 빈 줄이 생기는 문제를 방지합니다.
with open("p07_summary.csv", "w", encoding="utf-8-sig", newline="") as f:

    # CSV 파일에 데이터를 기록할 writer 객체를 생성합니다.
    output = csv.writer(f)

    # CSV의 첫 번째 줄에 각 열의 제목을 기록합니다.
    output.writerow([
        "source",
        "source_w",
        "source_h",
        "output_w",
        "output_h",
        "edge_pixels"
    ])

    # rows에 모아둔 각 이미지의 처리 결과를 CSV에 기록합니다.
    output.writerows(rows)


# --------------------------------------------------
# 저장 결과 재검증
# --------------------------------------------------

# 실습08: 파일을 실제로 다시 읽습니다. 성공 메시지만으로 완료 판단하지 않습니다.

# 저장한 p07_video.avi 파일을 다시 읽기 위한 객체를 생성합니다.
check = cv2.VideoCapture("p07_video.avi")

# 처리된 이미지 파일 개수와 동영상 프레임 개수를 출력합니다.
print("이미지 수/Frame 수:", len(rows), count)

# 저장된 흑백 엣지 이미지를 다시 읽어옵니다.
# 두 번째 인자 0은 흑백 모드로 이미지를 읽겠다는 의미입니다.
#
# shape를 통해 실제 저장된 이미지의 높이와 너비를 확인합니다.
print("재읽기 이미지 Shape:", cv2.imread("p07_scene_edge.png", 0).shape)

# 저장된 동영상에서 실제 프레임 개수를 가져옵니다.
# CAP_PROP_FRAME_COUNT: 전체 프레임 수
# CAP_PROP_FPS: 초당 프레임 수
#
# int()는 프레임 개수를 일반 정수로 변환합니다.
print(
    "재읽기 영상 Frame/FPS:",
    int(check.get(cv2.CAP_PROP_FRAME_COUNT)),
    check.get(cv2.CAP_PROP_FPS)
)

# 저장된 동영상의 읽기 객체를 해제합니다.
check.release()

# 최종적으로 생성된 결과 파일의 종류와 개수를 안내합니다.
# 이미지 3개 × 결과 3종류 = PNG 9개
# 동영상 1개 = AVI 1개
# 처리 결과 표 = CSV 1개
print("결과: p07_ PNG 9개 + AVI 1개 + CSV 1개")
