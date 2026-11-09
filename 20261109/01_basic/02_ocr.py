"""실습 02: Tesseract를 실제 호출합니다. 엔진과 eng 언어팩이 필요합니다."""
"""
[실습 02 - B02 핵심 개념: 실제 Tesseract OCR과 단어별 결과]

1. OCR(Optical Character Recognition)
   - 이미지에 그려진 문자를 기계가 처리할 수 있는 텍스트로 인식하는 작업
   - 이 파일은 원본 doc01~doc03 이미지로 실제 OCR을 수행
   - 앞 실습에서 만든 Gray·이진화 이미지를 자동으로 사용하는 것은 아님
   - 문자 인식 결과가 올바른 날짜·금액 등 업무 의미까지 보장하는 것은 아님

2. Tesseract와 pytesseract
   - Tesseract는 OCR을 수행하는 별도 실행 프로그램
   - pytesseract는 Python에서 Tesseract를 호출하고 결과를 읽는 연결 도구
   - pip로 pytesseract를 설치하는 것만으로 엔진이 설치되지는 않음
   - lang='eng'를 사용하려면 엔진에 영어 언어 데이터가 준비되어 있어야 함

3. 실행 경로와 환경변수
   - tesseract_cmd는 호출할 엔진 실행 파일의 경로 또는 명령 이름
   - os.environ.get('TESSERACT_CMD', 'tesseract'): 환경변수 값이 없으면 기본 명령 사용
   - 기본 명령을 사용하려면 엔진 설치 폴더가 PATH에서 검색되어야 함
   - 환경변수에 경로를 지정하는 것은 엔진 자체를 설치하는 작업이 아님

4. 여러 이미지의 입력 목록
   - p02_inputs.txt에 이미지 경로를 한 줄씩 기록
   - Tesseract에 목록 파일을 전달하여 여러 페이지를 한 번의 엔진 호출로 처리
   - page_num은 목록의 페이지 번호이며 이 실습에서는 1부터 시작
   - 원본 이미지 파일을 수정하거나 목록에 픽셀 데이터를 저장하는 것은 아님

5. image_to_data()와 DataFrame
   - 이미지 또는 지원되는 입력 파일 경로를 받아 OCR 상세 결과를 반환
   - 단어의 text, conf와 left·top·width·height 등의 좌표 정보를 포함
   - 페이지·블록·문단·줄·단어 등 계층별 행이 포함될 수 있음
   - Output.DATAFRAME은 결과를 행과 열로 구성된 pandas 표로 받는 옵션

6. lang과 PSM
   - lang='eng': 영어 인식용 언어 데이터 사용
   - --psm 6: 페이지를 하나의 균일한 텍스트 블록으로 가정하는 분할 모드
   - 언어·페이지 배치에 맞지 않는 설정은 인식 결과에 영향을 줄 수 있음
   - 언어팩이 여러 개 설치되어 있어도 이 호출은 eng만 지정하여 사용

7. 결과 필터링
   - dropna(subset=['text']): text가 결측값인 행 제거
   - words['conf'] >= 0: 유효한 비음수 confidence가 있는 행만 선택
   - conf=-1은 일반적으로 단어 점수가 없는 계층 행 등에서 나타남
   - 이 필터는 정답 여부를 검사하거나 낮은 양수 점수를 제거하는 기준이 아님

8. confidence와 평균
   - confidence는 엔진의 인식 점수이며 정답 확률이나 정확도가 아님
   - 높은 점수여도 O와 0 혼동, 단어 붙음 등 오류가 있을 수 있음
   - mean()은 남은 각 행 점수의 단순 평균이며 문자 수에 따른 가중 평균이 아님
   - len(rows)는 필터링 후 결과 항목 수이며 정답 기준 실제 단어 수와 다를 수 있음
   - OCR 품질은 정답 텍스트와 대조하여 별도로 확인해야 함

9. 페이지별 텍스트 조합
   - groupby('page_num', sort=True): 페이지 번호별로 묶어 오름차순 처리
   - astype(str): 인식 값을 문자열로 변환
   - ' '.join(...): 항목 사이에 공백을 넣어 한 문자열로 연결
   - 원본의 줄바꿈이나 표 구조를 복원하는 코드는 아님

10. 저장과 표시
    - Path.resolve().parent: 현재 소스 폴더, i:02d: 번호를 두 자리로 표시
    - write_text(): UTF-8 입력 목록 저장, to_csv(index=False): 행 번호 없이 표 저장
    - utf-8-sig: BOM을 포함한 UTF-8로 일부 프로그램의 한글 인식에 도움
    - round(..., 2): 평균 점수를 소수점 둘째 자리까지 반올림
    - 같은 이름의 산출물은 재실행 시 덮어쓰며 엔진 버전에 따라 결과가 달라질 수 있음
"""

# 파일 및 폴더 경로 관리
from pathlib import Path

# 현재 프로세스의 환경변수 조회
import os

# OCR 결과를 표로 처리하고 CSV로 저장
import pandas as pd

# Python에서 별도 Tesseract 엔진을 호출하는 연결 도구
import pytesseract

# 실행한 터미널 위치와 관계없이 현재 소스 폴더를 기준으로 사용
base = Path(__file__).resolve().parent
# Windows에서 PATH 설정이 어려우면 환경변수 TESSERACT_CMD에 exe 경로를 지정합니다.
# 환경변수에 실행 파일 경로가 있으면 사용하고, 없으면 PATH의 tesseract 명령 사용
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT_CMD', 'tesseract')

# range(1, 4)는 1·2·3이며 :02d로 doc01.png~doc03.png 경로 생성
paths = [str(base / f'doc{i:02d}.png') for i in range(1, 4)]

# 여러 이미지를 한 번에 전달할 목록 파일 경로
input_list = base / 'p02_inputs.txt'

# 이미지 경로를 줄바꿈으로 연결하여 목록 파일 저장
input_list.write_text('\n'.join(paths), encoding='utf-8')
# 여러 페이지를 목록으로 전달해 엔진을 한 번 호출합니다.
# TODO B02
words = pytesseract.image_to_data(
    # 원본 이미지 세 장의 경로가 기록된 목록 파일을 전달
    str(input_list),
    # 영어 언어 데이터로 인식
    lang='eng',
    # 각 페이지를 하나의 균일한 텍스트 블록으로 가정
    config='--psm 6',
    # 문자열만이 아니라 단어·위치·점수 등이 포함된 DataFrame 반환
    output_type=pytesseract.Output.DATAFRAME,
)
# TODO 끝 B02
# text가 없는 계층 행 등 결측 텍스트 행 제거
words = words.dropna(subset=['text'])

# 유효한 비음수 점수가 있는 행만 선택. 인식이 맞는지 검증하는 필터는 아님
words = words[words['conf'] >= 0]

# 남은 단어별 상세 정보를 원래 좌표 열 등과 함께 CSV 저장
words.to_csv(base / 'p02_words.csv', index=False, encoding='utf-8-sig')

# 결과를 페이지별로 묶고 페이지 번호 오름차순으로 처리
for page, rows in words.groupby('page_num', sort=True):
    # 각 인식 항목을 문자열로 바꿔 공백으로 연결. 원본 표·줄 배치는 보존하지 않음
    text = ' '.join(rows['text'].astype(str))
    # 필터링 후 항목 수와 해당 행들의 단순 평균 confidence 출력
    print('page:', page, 'words:', len(rows), 'mean confidence:', round(rows['conf'].mean(), 2))
    # 현재 페이지에서 인식한 텍스트 출력
    print(text)

# 실제 엔진 호출 결과임을 명시하고 confidence 해석의 범위를 안내
print('backend: live Tesseract; confidence는 정답 확률이 아닙니다.')
