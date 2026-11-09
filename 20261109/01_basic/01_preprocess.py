"""실습 01: 정상 문서 3장을 Gray와 Otsu로 변환합니다. 원본을 덮어쓰지 않습니다."""
"""
[실습 01 - B01 기본 개념: 문서 Gray 변환과 Otsu 이진화]

1. 문서 이미지 전처리
   - OCR 등 후속 처리에 사용할 이미지의 표현을 바꾸는 작업
   - 이 파일은 원본 문서 3장을 각각 Gray 이미지와 이진 이미지로 변환
   - 문자 인식이나 OCR 정확도 계산은 수행하지 않음

2. BGR과 Gray
   - cv2.imread()의 기본 색상 채널 순서는 파랑·초록·빨강(BGR)
   - Gray는 색상 대신 밝기를 한 값으로 표현하는 단일 채널 이미지
   - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)로 변환
   - 단순히 채널 하나를 선택하는 것이 아니라 색상별 기여를 반영해 밝기를 계산
   - 색상 배열은 (높이, 너비, 3), 이 코드의 Gray 배열은 (높이, 너비)

3. 이진화와 임계값
   - 밝기를 기준과 비교하여 픽셀을 두 값으로 나누는 처리
   - THRESH_BINARY에서는 임계값보다 크면 maxval, 나머지는 0
   - 이 코드의 maxval=255이므로 검정 0과 흰색 255로 표현
   - Gray는 여러 밝기를 유지하지만 이진 이미지는 두 밝기만 사용

4. Otsu 자동 임계값
   - 영상의 밝기 분포에서 두 집단을 나눌 전역 임계값을 자동 선택하는 방법
   - 두 집단의 클래스 내 분산을 작게 만드는 기준을 사용
   - THRESH_OTSU를 THRESH_BINARY와 함께 지정하여 자동 선택과 이진화 수행
   - thresh=0은 호출에 전달하는 값이며 실제 기준은 Otsu가 계산
   - 조명이 불균일하거나 글자가 흐리면 Otsu만으로 충분하지 않을 수 있음

5. cv2.threshold()의 입력과 반환값
   - 주요 인자: src, thresh, maxval, type
   - 이 코드에서는 src로 8비트 단일 채널 gray를 전달
   - 반환값은 (사용한 임계값, 결과 이미지)의 두 항목
   - threshold 변수는 자동 선택된 숫자, binary 변수는 결과 배열
   - 함수 인자 이름 thresh와 결과 변수 이름 threshold를 구분해야 함

6. 파일 경로와 반복 처리
   - Path.resolve().parent: 현재 소스 파일이 들어 있는 폴더
   - for문으로 doc01, doc02, doc03을 같은 규칙으로 처리
   - f-string으로 문서 이름을 입력·출력 파일명에 삽입
   - 원본과 다른 이름으로 저장하여 원본 파일을 덮어쓰지 않음

7. 저장과 결과 확인
   - cv2.imwrite(): 이미지 저장 후 성공 여부를 반환
   - 현재 코드는 반환값을 검사하지 않으므로 실제 파일 생성 여부도 확인 필요
   - 문서마다 Gray·이진 이미지 2개씩 총 6개를 생성
   - 재실행하면 같은 이름의 결과 파일은 덮어쓸 수 있음
   - shape는 배열 크기이며 threshold는 자동으로 구한 밝기 기준
   - 사람이 읽기 쉬워졌다고 OCR 정확도가 반드시 높아지는 것은 아님
   - OCR 개선 여부는 같은 엔진·설정에서 정답과 비교해 별도로 확인해야 함
"""

# 파일 및 폴더 경로를 관리하는 클래스
from pathlib import Path

# 이미지 읽기, 색상 변환, 이진화와 이미지 저장 도구
import cv2

# 터미널 실행 위치와 관계없이 현재 소스 폴더에서 입력·출력을 찾음
base = Path(__file__).resolve().parent

# 문서 이름 세 개를 순서대로 처리
for name in ['doc01', 'doc02', 'doc03']:
    # doc01.png 등 원본 문서를 BGR 이미지 배열로 읽음
    # 정상 이미지가 존재한다는 실습 전제이며 읽기 실패 시에는 None이 반환됨
    image = cv2.imread(str(base / f'{name}.png'))
    # 아래 두 줄만 완성합니다. 출력은 채널이 없는 (높이, 너비) 배열입니다.
    # TODO B01
    # BGR 3채널 이미지를 밝기만 가진 2차원 Gray 배열로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu가 선택한 임계값은 threshold, 이진 결과 배열은 binary에 저장
    threshold, binary = cv2.threshold(
        # 이진화할 Gray 이미지
        gray,
        # Otsu 모드에서는 자동으로 임계값을 결정하므로 0을 전달
        thresh=0,
        # 임계값보다 밝은 픽셀을 흰색 255로 지정
        maxval=255,
        # 자동 임계값 선택과 일반 이진화 방식을 함께 사용
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # TODO 끝 B01

    # Gray 이미지를 원본과 다른 이름으로 저장
    cv2.imwrite(str(base / f'p01_{name}_gray.png'), gray)

    # 같은 문서의 이진 이미지를 별도 PNG로 저장
    cv2.imwrite(str(base / f'p01_{name}_binary.png'), binary)

    # 문서 이름, Gray 배열의 (높이, 너비), Otsu가 선택한 실제 임계값 출력
    print(name, 'shape:', gray.shape, 'threshold:', threshold)
