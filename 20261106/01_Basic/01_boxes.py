"""
[실습 01 - B01 기본 개념: 정답 Bounding Box 그리기]

1. Bounding Box와 GT
   - 이미지에서 물체의 위치를 사각형으로 표현한 영역
   - GT(Ground Truth): 미리 제공된 실제 정답 정보
   - 이 코드는 모델로 물체를 탐지하지 않고 CSV에 있는 정답 박스를 그림
   - class_id는 물체의 클래스 번호이며 이름과의 대응은 데이터 정의로 확인해야 함

2. xyxy 좌표와 이미지 좌표계
   - (x1, y1): 박스 왼쪽 위 점, (x2, y2): 오른쪽 아래 점
   - 이미지 왼쪽 위가 원점이며 x는 오른쪽, y는 아래쪽으로 증가
   - 점 좌표는 (x, y)이지만 이미지 배열의 픽셀 접근은 image[y, x] 순서

3. cv2.imread()와 이미지 배열
   - 파일을 읽어 NumPy 배열로 반환하는 OpenCV 함수
   - 기본 색상 이미지의 채널 순서는 RGB가 아니라 BGR
   - 일반적인 색상 이미지 shape는 (높이, 너비, 채널 수)
   - 파일을 읽지 못하면 None을 반환하며 이 코드는 정상 입력을 전제로 함

4. copy()와 원본 보존
   - image.copy(): 픽셀 데이터를 별도로 복사한 배열 생성
   - result에 그림을 그려도 원본 image의 픽셀값은 변경되지 않음
   - 단순히 result = image로 대입하면 같은 배열을 가리키므로 원본도 바뀔 수 있음

5. pandas와 itertuples()
   - pd.read_csv(): CSV 파일을 행과 열로 구성된 DataFrame으로 읽음
   - itertuples(): 각 행을 튜플 형태로 순서대로 반환
   - row.x1, row.class_id처럼 열 이름에 대응하는 속성으로 값을 조회
   - 반복문에서 각 정답 박스를 하나씩 처리

6. cv2.rectangle()
   - 기본 문법: cv2.rectangle(이미지, 시작점, 끝점, 색상, thickness=두께)
   - 전달받은 이미지 배열에 직접 사각형을 그림
   - BGR 색상 (0, 255, 0)은 초록색
   - thickness=2는 2픽셀 두께의 테두리, thickness=-1은 내부 채우기
   - 이 실습에서는 result에 각 행의 좌표를 사용하여 두께 2로 그림

7. 정답 박스와 confidence
   - confidence는 일반적으로 모델이 예측에 부여한 점수를 의미
   - 이 파일은 GT를 표시하므로 모델 예측 confidence가 없음
   - N/A는 해당 없음이라는 의미이며 confidence가 0이라는 뜻이 아님

8. Path와 결과 저장
   - Path: 파일·폴더 경로를 관리하는 클래스
   - __file__: 현재 소스 파일 경로, resolve(): 절대 경로 변환
   - parent: 상위 폴더, / 연산자: 하위 경로 연결
   - cv2.imwrite(): 이미지 배열을 지정한 경로에 저장하고 성공 여부를 반환
   - 이 코드의 마지막 print는 저장 성공 여부를 검사하는 코드는 아님
   - 결과는 원본과 다른 이름인 p01_boxes.png로 저장
"""

"""실습 01. 정답 박스 3개를 그립니다. 원본 scene.png는 바꾸지 않습니다."""

# 운영체제에 맞게 파일 및 폴더 경로를 다루는 클래스
from pathlib import Path

# 이미지 읽기, 사각형 그리기, 이미지 저장을 위한 OpenCV
import cv2

# 정답 박스 CSV를 표 형태로 읽는 라이브러리
import pandas as pd

# 실행한 터미널 위치와 관계없이 이 소스 파일의 폴더를 기준 경로로 사용
BASE = Path(__file__).resolve().parent

# data 폴더의 원본 장면 이미지를 BGR 배열로 읽음
image = cv2.imread(str(BASE / "data/scene.png"))

# 클래스 번호와 박스 좌표가 들어 있는 정답 CSV를 읽음
boxes = pd.read_csv(BASE / "data/boxes_gt.csv")

# 그림을 그릴 별도 배열을 복사하여 원본 image를 보존
result = image.copy()

# 정답 표의 각 행을 순서대로 가져와 박스를 하나씩 처리
for row in boxes.itertuples():
    # >>> B01
    # 요구: result에 (x1,y1)~(x2,y2) 박스를 두께 2로 그리세요.
    # cv2.rectangle의 이미지·두 점·색은 필수, thickness는 선택입니다.
    # TODO B01: 위 요구사항의 핵심 코드를 작성하세요.
    # result: 박스를 그릴 이미지 배열
    # (row.x1, row.y1): 현재 행의 왼쪽 위 좌표
    # (row.x2, row.y2): 현재 행의 오른쪽 아래 좌표
    # (0, 255, 0): BGR 순서의 초록색, thickness=2: 테두리 두께 2픽셀
    # result 배열을 직접 수정하므로 반환값을 다시 대입할 필요가 없음
    cv2.rectangle(result, (row.x1, row.y1), (row.x2, row.y2), (0, 255, 0), thickness=2)
    # <<< B01
    # 현재 박스의 클래스 번호와 좌표를 출력. GT이므로 모델 예측 점수는 해당 없음
    print("class:", row.class_id, "xyxy:", row.x1, row.y1, row.x2, row.y2, "GT confidence: N/A")

# 모든 박스를 그린 결과를 현재 소스 폴더에 별도 PNG로 저장
cv2.imwrite(str(BASE / "p01_boxes.png"), result)

# 원본 이미지의 (높이, 너비, 채널 수)와 결과 파일 이름을 출력
print("shape:", image.shape, "저장: p01_boxes.png")
