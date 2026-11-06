"""실습 01. 픽셀 xyxy와 YOLO 정규화 중심·크기를 왕복합니다."""
"""
[실습 01 - A01 기본 개념: 박스 좌표 정규화와 복원]

1. xyxy 좌표
   - 박스를 [왼쪽 x, 위쪽 y, 오른쪽 x, 아래쪽 y]로 표현
   - 이미지 왼쪽 위가 원점이며 x는 오른쪽, y는 아래로 증가
   - 이 코드는 정답 박스 좌표를 변환하며 모델로 물체를 탐지하지 않음

2. 중심 좌표와 크기
   - 중심 x = (x1 + x2) / 2, 중심 y = (y1 + y2) / 2
   - 폭 = x2 - x1, 높이 = y2 - y1
   - 이 실습은 폭과 높이에 +1을 더하지 않는 좌표 규칙을 사용

3. YOLO 정규화 좌표
   - [중심 x, 중심 y, 폭, 높이]를 전체 이미지 크기에 대한 비율로 표현
   - x와 폭은 이미지 너비로, y와 높이는 이미지 높이로 나눔
   - 이미지 내부의 올바른 박스는 일반적으로 [0, 1] 범위로 표현됨
   - 이미지 밖 좌표를 자동으로 잘라주는 연산은 아님
   - 일반적인 YOLO 라벨에는 클래스 번호도 있지만 이 CSV는 좌표 4개만 저장

4. 배열 슬라이싱과 벡터 연산
   - b는 (박스 수, 4) 배열이며 한 행이 박스 하나
   - b[:, 0]: 모든 행의 x1, b[:, 1]: y1, b[:, 2]: x2, b[:, 3]: y2
   - 배열에 사칙연산을 적용하여 반복문 없이 모든 박스를 계산
   - np.column_stack(): 여러 1차원 배열을 열 방향으로 묶음

5. 역변환과 브로드캐스팅
   - 정규화 값에 [width, height, width, height]를 곱해 픽셀 단위 복원
   - 브로드캐스팅으로 동일한 크기 벡터를 모든 행에 적용
   - x1 = 중심 x - 폭/2, x2 = 중심 x + 폭/2
   - y1 = 중심 y - 높이/2, y2 = 중심 y + 높이/2

6. 최대 왕복 오차
   - 원본 → 정규화 → 복원 좌표가 얼마나 일치하는지 검사
   - np.abs(): 절댓값, np.max(): 전체 원소 중 최대값
   - 0 또는 매우 작은 값이면 해당 데이터의 왕복 변환이 일치함
   - 실수 연산에 따른 작은 오차가 생길 수 있으며 탐지 정확도 지표는 아님

7. pandas와 파일 경로
   - read_csv(): 표 읽기, to_numpy(float): 실수 배열로 변환
   - DataFrame(): 배열을 표로 구성, to_csv(index=False): 행 번호 없이 저장
   - Path.resolve().parent: 현재 소스 폴더, mkdir(exist_ok=True): 출력 폴더 생성
   - width=160, height=128은 코드에서 지정하며 이미지에서 자동으로 읽지 않음
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# 좌표 배열의 수치 연산
import numpy as np
# 정답 CSV 읽기와 결과 표 저장
import pandas as pd
# 현재 소스 파일의 폴더를 기준 경로로 사용
BASE=Path(__file__).resolve().parent
# outputs 폴더를 지정하고 없으면 생성
out=BASE/"outputs";out.mkdir(exist_ok=True)
# 정답 박스 표를 읽음
table=pd.read_csv(BASE/"data/boxes_gt.csv")
# 좌표를 xyxy 순서의 실수 배열로 가져오고 전체 이미지 크기를 지정
b=table[["x1","y1","x2","y2"]].to_numpy(float);width,height=160,128
# >>> A01
# 요구: 중심 x/y와 폭/높이를 계산하고 각 축 크기로 정규화하세요.
# TODO A01: 위 요구사항의 핵심 코드를 작성하세요.
# 모든 박스의 중심 x를 계산한 뒤 전체 이미지 너비로 나눔
cx = ((b[:, 0] + b[:, 2]) / 2) / width
# 모든 박스의 중심 y를 계산한 뒤 전체 이미지 높이로 나눔
cy = ((b[:, 1] + b[:, 3]) / 2) / height
# 박스 폭을 계산하고 전체 이미지 너비로 나눔
w = (b[:, 2] - b[:, 0]) / width
# 박스 높이를 계산하고 전체 이미지 높이로 나눔
h = (b[:, 3] - b[:, 1]) / height

# 각 박스를 [중심 x, 중심 y, 폭, 높이] 순서로 묶음
yolo = np.column_stack((cx, cy, w, h))
# <<< A01
# x 관련 열에는 너비, y 관련 열에는 높이를 곱해 픽셀 단위 중심·크기 복원
pixel=yolo*np.array([width,height,width,height])
# 중심에서 반폭·반높이를 빼거나 더해 [x1, y1, x2, y2] 좌표 복원
restored=np.column_stack((pixel[:,0]-pixel[:,2]/2,pixel[:,1]-pixel[:,3]/2,pixel[:,0]+pixel[:,2]/2,pixel[:,1]+pixel[:,3]/2))
# 원본과 복원 좌표 차이의 절댓값 중 최댓값을 픽셀 단위로 출력
print("최대 왕복 오차(px):", float(np.max(np.abs(b-restored))))
# 정규화 배열에 열 이름을 붙여 CSV 저장. index=False로 행 번호 저장 제외
pd.DataFrame(yolo,columns=["cx","cy","w","h"]).to_csv(out/"yolo_coordinates.csv",index=False)
