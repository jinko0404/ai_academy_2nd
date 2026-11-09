"""
[실습 02 - A02 기본 개념: OCR 단어 박스와 PSM 입력 범위]

1. OCR 토큰과 제공 기록
   - 토큰은 OCR 결과에서 하나의 단어 등으로 구분된 항목
   - 이 코드는 lesson.json의 기록을 읽으며 OCR 엔진을 새로 호출하지 않음
   - backend는 기록의 출처를 나타내므로 출력값으로 확인해야 함

2. bbox와 좌표계
   - bbox는 단어 영역을 둘러싼 사각형(Bounding Box)
   - [x1, y1, x2, y2]에서 앞 두 값은 왼쪽 위, 뒤 두 값은 오른쪽 아래 경계
   - 이미지 왼쪽 위가 원점이며 x는 오른쪽, y는 아래로 증가
   - 여기서는 끝 좌표를 제외하는 픽셀 경계 [x1,x2), [y1,y2) 규칙을 사용
   - 마지막 포함 픽셀은 (x2-1, y2-1)이므로 rectangle의 끝점에 이를 반영
   - OCR 좌표가 계산된 이미지와 표시 이미지의 크기·회전·crop 기준이 같아야 함

3. cv2.rectangle()과 BGR
   - 주요 인자: 이미지, 시작점, 끝점, 색상, 두께
   - OpenCV 기본 색상 순서는 BGR이며 파랑은 (255, 0, 0)
   - 전달된 이미지 배열에 직접 그리며 반환값을 다시 대입할 필요가 없음
   - 메모리의 image는 바뀌지만 별도 파일명으로 저장하므로 원본 파일은 유지

4. PSM(Page Segmentation Mode)
   - Tesseract가 입력 이미지의 텍스트 배치를 어떻게 가정할지 정하는 설정
   - PSM 3: 전체 페이지 자동 분할, PSM 6: 하나의 균일한 텍스트 블록
   - PSM 7: 한 줄 텍스트로 가정하므로 적합한 한 줄 crop을 입력
   - 이 코드는 제공된 모드별 기록만 출력하며 모드를 바꿔 실시간 실행하지 않음

5. CER와 공정한 비교
   - CER은 문자 단위 삽입·삭제·치환 오류 수를 정답 문자 수로 나눈 지표
   - 비교할 때는 정답과 예측의 대상 영역·문자열 정규화 규칙을 맞춰야 함
   - 전체 페이지와 한 줄 crop은 범위가 다르므로 수치만으로 모드 순위를 정하면 안 됨
   - 박스가 그려졌다는 사실도 문자 인식의 정확성을 입증하지 않음

6. 설정 읽기와 출력
   - json.loads(): JSON 문자열을 Python 딕셔너리 등으로 변환
   - Path.resolve().parent: 현재 소스 폴더, mkdir(exist_ok=True): 출력 폴더 생성
   - len(tokens): 제공 토큰 개수, [:65]: 텍스트 앞의 최대 65문자 선택
   - repr(): 줄바꿈 등을 이스케이프로 표현하여 문자열을 확인
   - imwrite(): 파일 저장 성공 여부 반환. 이 코드는 반환값을 검사하지 않음
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# 제공된 OCR·PSM 기록을 JSON에서 읽기
import json
# 정규 표현식 도구이며 현재 파일에서는 사용하지 않음
import re
# 수치 배열 도구이며 현재 파일에서는 사용하지 않음
import numpy as np
# 표 처리 도구이며 현재 파일에서는 사용하지 않음
import pandas as pd
# 이미지 읽기와 사각형 그리기·저장
import cv2
# 현재 소스 폴더를 기준 경로로 사용
base = Path(__file__).resolve().parent
# 원본과 설정이 있는 폴더
data = base / 'data'
# 결과 이미지 저장 폴더
out = base / 'outputs'
# 폴더가 없으면 생성
out.mkdir(exist_ok=True)
# UTF-8 설정 파일을 읽고 Python 객체로 변환
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

# 여러 문서 중 doc01의 파일 정보와 OCR 기록 선택
doc = lesson['documents']['doc01']
# 해당 원본 문서를 BGR 배열로 읽음
image = cv2.imread(str(data / doc['file']))
# 제공된 단어 토큰을 하나씩 처리
for word in doc['ocr']['tokens']:
    # 네 경계 좌표를 순서대로 분리
    x1, y1, x2, y2 = word['bbox']
    # TODO A02
    # 끝 경계는 제외되는 좌표이므로 마지막 포함 픽셀까지 파란 테두리로 표시
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), thickness=1)
    # TODO 끝 A02
# 박스를 그린 문서를 별도 PNG로 저장하여 원본 파일 보존
cv2.imwrite(str(out / 'p02_word_boxes.png'), image)
# PSM을 실시간으로 반복 호출하지 않고 모드와 적합한 입력을 설계합니다.
# 토큰 수와 제공 기록의 출처 출력
print('단어:', len(doc['ocr']['tokens']), 'backend:', doc['ocr']['backend'])
# 각 모드에 적합한 입력 범위 안내
print('PSM 3: 전체 문서 / PSM 6: 균일한 글 블록 / PSM 7: 한 줄 crop')
# 제공된 PSM 비교 기록을 하나씩 확인
for record in lesson['psm_records']:
    # 모드·영역 이름과 인식 텍스트 앞부분을 출력하며 줄바꿈 등은 repr로 표시
    print('PSM', record['psm'], record['region'], '->', repr(record['text'][:65]))
# 대상 영역이 다른 기록을 같은 조건의 인식 성능처럼 비교하지 않도록 안내
print('PSM 7은 한 줄 crop입니다. 전체 페이지와 입력 범위가 다르므로 CER을 직접 순위화하지 않습니다.')
