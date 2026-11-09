"""
[실습 01 - A01 기본 개념: 문서 원근 보정과 적응형 이진화]

1. 원근 왜곡과 보정
   - 문서를 비스듬히 촬영하면 직사각형이 사다리꼴처럼 보일 수 있음
   - 문서의 네 모서리를 출력 직사각형의 네 점에 대응시켜 원근을 보정
   - 단순 resize는 크기만 바꾸므로 이러한 기하학적 왜곡을 제거하지 못함
   - 이 실습은 제공된 모서리를 사용하며 모서리를 자동으로 검출하지 않음

2. 모서리 좌표와 순서
   - corners의 순서는 TL(왼쪽 위), TR(오른쪽 위), BR(오른쪽 아래), BL(왼쪽 아래)
   - 원본 점과 목적지 점의 순서가 일치해야 올바른 대응 관계를 얻음
   - 좌표는 (x, y) 순서이며 이미지 배열 접근의 (행, 열)과 구분
   - float32 배열로 변환하여 OpenCV에 전달

3. getPerspectiveTransform()
   - 원본 점 4개와 목적지 점 4개에서 3×3 원근 변환 행렬을 계산
   - 행렬을 계산하는 단계에서는 이미지 픽셀이 아직 변환되지 않음
   - 출력 크기 1000×700에서 끝 픽셀 좌표는 x=999, y=699

4. A01: warpPerspective()
   - 기본 문법: cv2.warpPerspective(원본 이미지, 변환 행렬, 출력 크기)
   - 계산된 행렬에 따라 픽셀을 이동·보간하여 보정 이미지 생성
   - dsize는 (너비, 높이)이므로 size=(1000, 700)을 전달
   - 결과 shape는 색상 영상에서 (높이, 너비, 채널 수)인 (700, 1000, 3)
   - 출력 크기가 같더라도 resize와 원근 변환의 결과는 다름

5. Gray 변환
   - OpenCV 기본 색상 이미지의 채널 순서는 BGR
   - cvtColor(..., COLOR_BGR2GRAY)로 밝기만 가진 2차원 배열 생성
   - 적응형 이진화에 사용할 8비트 단일 채널 입력 준비

6. 적응형 이진화(adaptiveThreshold)
   - 이미지 전체에 하나의 임계값을 사용하는 대신 주변 영역별로 기준을 계산
   - 조명이나 그림자가 위치에 따라 다른 문서를 처리할 때 활용
   - ADAPTIVE_THRESH_GAUSSIAN_C: 주변 밝기를 가우시안 가중 평균한 값에서 C를 뺌
   - blockSize=31: 주변 31×31 영역 사용. 1보다 큰 홀수여야 함
   - C=11: 지역 평균에서 빼는 상수
   - THRESH_BINARY: 지역 임계값보다 밝으면 255, 나머지는 0

7. 설정 읽기와 결과 저장
   - read_text(): 파일 문자열 읽기, json.loads(): JSON 문자열을 Python 객체로 변환
   - lesson.json은 원근 보정에 필요한 모서리 정보를 제공
   - imwrite(): 이미지 저장 후 성공 여부 반환. 이 코드는 반환값을 검사하지 않음
   - outputs에 보정 색상 이미지와 적응형 이진 이미지 저장
   - 원본을 덮어쓰지 않지만 재실행하면 같은 이름의 결과는 덮어쓸 수 있음

8. 해석 범위
   - shape는 배열 크기이며 기하학적 보정 품질이나 OCR 정확도를 뜻하지 않음
   - 모서리 오차와 픽셀 보간으로 글자가 잘리거나 흐려질 수 있어 육안 확인 필요
   - 보정·이진화가 OCR을 개선하는지는 동일한 인식 설정에서 정답과 비교해야 함
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# JSON 설정 문자열을 Python 딕셔너리로 변환
import json
# 정규 표현식 도구이며 현재 파일에서는 사용하지 않음
import re
# 모서리 좌표를 float32 배열로 변환
import numpy as np
# 표 처리 도구이며 현재 파일에서는 사용하지 않음
import pandas as pd
# 이미지 읽기, 원근 변환, 이진화와 저장
import cv2
# 현재 소스 파일의 폴더를 기준 경로로 사용
base = Path(__file__).resolve().parent
# 원본 문서와 설정 파일이 들어 있는 폴더
data = base / 'data'
# 처리 결과를 저장할 폴더
out = base / 'outputs'
# 폴더가 없으면 생성하고 이미 있으면 그대로 사용
out.mkdir(exist_ok=True)
# UTF-8 JSON 파일을 읽어 모서리 설정 등을 담은 딕셔너리로 변환
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

# 비스듬한 원본 문서를 BGR 배열로 읽음. 정상 파일을 전제로 함
image = cv2.imread(str(data / 'perspective.png'))
# 제공된 네 모서리를 왼쪽 위·오른쪽 위·오른쪽 아래·왼쪽 아래 순서로 가져옴
corners = np.float32(lesson['perspective']['corners'])
# OpenCV 출력 크기는 (너비, 높이) 순서
size = (1000, 700)
# 원본 네 점을 출력 직사각형 네 모서리로 보내는 3×3 변환 행렬 계산
matrix = cv2.getPerspectiveTransform(corners, np.float32([[0,0],[999,0],[999,699],[0,699]]))
# TODO A01
# 크기만 바꾸는 resize 대신 위에서 계산한 원근 변환 행렬을 실제 이미지에 적용
rectified = cv2.warpPerspective(image, matrix, size)
# TODO 끝 A01
# 보정된 색상 이미지를 이진화용 Gray 이미지로 변환
gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
# 31×31 주변 영역의 가우시안 가중 평균에서 11을 뺀 지역 임계값으로 이진화
# 결과는 검정 0과 흰색 255를 사용하는 2차원 이미지
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
# 보정된 색상 문서 저장
cv2.imwrite(str(out / 'p01_rectified.png'), rectified)
# 보정 후 적응형 이진화를 적용한 문서 저장
cv2.imwrite(str(out / 'p01_adaptive.png'), adaptive)
# 원본과 보정 이미지의 (높이, 너비, 채널 수) 출력
print('before:', image.shape, 'after:', rectified.shape)
# 이번 작업이 제공 모서리를 이용한 보정임을 명시
print('알려진 모서리로 보정했습니다. 모서리 자동 검출이 아닙니다.')
