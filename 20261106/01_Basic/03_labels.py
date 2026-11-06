"""
[실습 03 - B04 기본 개념: 이미지와 YOLO TXT 라벨]

1. 이미지와 라벨 파일의 대응
   - 이미지는 입력 데이터이며 라벨 파일은 정답 물체의 클래스와 위치를 기록
   - 이미지와 같은 기본 이름의 .txt 파일을 라벨로 사용
   - 예: images/train/sample.png ↔ labels/train/sample.txt
   - 한 이미지에 물체가 여러 개 있으면 라벨 파일에 여러 줄로 기록

2. YOLO TXT 라벨 형식
   - 한 줄은 class_id cx cy width height의 다섯 값으로 구성
   - class_id: 클래스 번호, cx·cy: 박스 중심, width·height: 박스 폭·높이
   - 값은 공백으로 구분하며 일반적인 정답 라벨에는 confidence가 없음
   - 이 실습은 라벨을 읽는 작업이며 모델로 예측하는 과정이 아님

3. 정규화된 중심·크기 좌표
   - cx와 width는 픽셀 단위 값을 이미지 전체 너비로 나눈 비율
   - cy와 height는 이미지 전체 높이로 나눈 비율
   - 이미지 내부의 정상 박스는 일반적으로 [0, 1] 범위로 표현됨
   - 중심·크기 형식은 왼쪽 위와 오른쪽 아래를 나타내는 xyxy 형식과 다름

4. 클래스 번호와 이름 매핑
   - 클래스 이름 목록의 위치가 클래스 번호에 대응하며 인덱스는 0부터 시작
   - 이 파일의 names는 square=0, circle=1, triangle=2로 정의됨
   - 이전 날짜 실습의 매핑과 다를 수 있으므로 현재 데이터의 정의를 따라야 함
   - int(cls)로 정수 번호를 얻고 names[번호]로 이름을 조회

5. Path와 파일 검색
   - Path: 파일과 폴더 경로를 관리하는 클래스
   - resolve(): 절대 경로 변환, parent: 상위 폴더
   - glob("*.png"): 폴더에서 PNG 경로 검색
   - name: 확장자를 포함한 파일 이름, stem: 확장자를 제외한 기본 이름
   - sorted(): 경로 정렬, [:6]: 앞의 최대 6개만 선택

6. 텍스트 파일 읽기와 줄 분리
   - Path.read_text(encoding=...): 파일 내용을 하나의 문자열로 읽음
   - encoding="utf-8": 파일 내용을 UTF-8 규칙으로 해석
   - 문자열.splitlines(): 여러 줄 문자열을 줄별 문자열 리스트로 변환
   - 파일 읽기와 줄 나누기는 별개의 작업
   - 라벨이 정상 형식이고 파일이 존재한다는 실습 전제를 사용

7. 한 줄의 값 분리와 숫자 변환
   - 문자열.split(): 공백을 기준으로 한 줄을 여러 문자열로 분리
   - map(float, ...): 각 문자열을 실수로 변환
   - 다섯 변수에 대입하면 순서대로 클래스 번호와 좌표 값을 받음
   - 이 코드에서는 클래스 번호도 먼저 float로 읽고 이름 조회 시 int로 변환

8. 이미지 shape와 박스 개수
   - cv2.imread(): 이미지를 NumPy 배열로 읽으며 기본 색상 순서는 BGR
   - image.shape: 일반적인 색상 이미지의 (높이, 너비, 채널 수)
   - len(lines): 읽어들인 라벨 줄 수
   - 정상 라벨에서 한 줄이 한 박스이므로 줄 수를 박스 개수로 표시
   - 개수와 좌표 출력만으로 라벨이 물체를 정확히 감싸는지까지 판단할 수는 없음
"""

# 파일 및 폴더 경로를 관리하는 클래스
from pathlib import Path

# 이미지 파일을 읽는 OpenCV 라이브러리
import cv2

# 터미널의 실행 위치와 관계없이 현재 소스 파일의 폴더를 기준 경로로 사용
BASE = Path(__file__).resolve().parent

# 목록의 인덱스가 클래스 번호에 대응: square=0, circle=1, triangle=2
names = ["square", "circle", "triangle"]

# train 폴더의 PNG 경로를 정렬하여 앞의 최대 6장만 처리
for image_path in sorted((BASE / "data/images/train").glob("*.png"))[:6]:

    # 현재 이미지를 BGR 배열로 읽음. 이 실습은 정상적으로 읽히는 파일을 전제로 함
    image = cv2.imread(str(image_path))

    # 이미지의 확장자를 제외한 이름(stem)에 .txt를 붙여 대응 라벨 경로 생성
    # 예: sample.png → data/labels/train/sample.txt
    label_path = BASE / "data/labels/train" / (image_path.stem + ".txt")
    # >>> B04
    # 요구: UTF-8 라벨 파일의 문자열을 읽고 줄 목록으로 나누세요.
    # TODO B04: 위 요구사항의 핵심 코드를 작성하세요.
    # label_path는 실제 읽을 파일의 Path 객체. Path 클래스 자체에서 읽는 것이 아님
    # read_text(): 파일 전체를 UTF-8 문자열로 읽음
    # splitlines(): 문자열을 줄별 리스트로 나눔. 정상 라벨에서는 한 줄이 박스 하나
    # 빈 파일이면 []가 되어 아래 박스 처리 반복문은 실행되지 않음
    lines = label_path.read_text(encoding="utf-8").splitlines()
    # <<< B04
    # 파일 이름, (높이, 너비, 채널 수), 정답 박스 개수를 출력
    print(image_path.name, "shape:", image.shape, "boxes:", len(lines))

    # 현재 이미지의 라벨을 한 줄씩 처리하여 모든 정답 박스 정보를 확인
    for line in lines:

        # 공백으로 분리한 다섯 문자열을 실수로 바꾸어 각 변수에 순서대로 대입
        # cls: 클래스 번호, cx·cy: 정규화 중심 좌표, width·height: 정규화 박스 크기
        # 여기의 width·height는 이미지 전체의 픽셀 너비·높이가 아님
        cls, cx, cy, width, height = map(float, line.split())

        # 실수로 읽은 클래스 번호를 정수로 바꾸어 클래스 이름 조회
        # 좌표는 픽셀로 복원하지 않고 라벨에 기록된 정규화 값 그대로 출력
        print(" ", names[int(cls)], "normalized:", cx, cy, width, height)
