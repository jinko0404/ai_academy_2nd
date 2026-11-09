"""
[실습 03 - A03 기본 개념: OCR 필드 추출과 근거 정보 보존]

1. 근거 기반 필드 추출
   - 값만 저장하는 대신 어떤 OCR 단어에서 추출했는지 원문·좌표·점수를 함께 기록
   - 결과를 사람이 검토하거나 잘못된 추출의 원인을 추적하는 데 활용
   - 이 파일은 lesson.json의 제공 OCR 기록을 읽으며 엔진을 새로 호출하지 않음

2. 토큰과 문서 계층
   - words는 단어별 text, conf, bbox와 문서 계층 번호를 가진 목록
   - page: 페이지, block: 블록, paragraph: 문단, line: 줄
   - 줄 번호만 같아도 다른 문단일 수 있으므로 상위 계층 번호도 함께 비교
   - 이 계층은 OCR이 부여한 결과이며 항상 실제 문서 구조와 정확히 같지는 않음

3. anchor(기준 단어)
   - DATE:, SUPPLIER:, TOTAL:처럼 값의 위치를 찾는 기준 라벨
   - next(조건을 만족하는 항목들)는 첫 번째 일치 항목을 반환
   - 이 코드는 라벨이 정확히 인식되어 존재하고 오른쪽 값도 있다는 정상 자료 전제
   - 라벨이 없으면 StopIteration, 값이 없으면 min·max 등에서 오류가 날 수 있음

4. A03: 같은 줄의 오른쪽 단어 선택
   - anchor와 페이지·블록·문단·줄이 모두 같은 토큰만 선택
   - bbox[0]은 단어 왼쪽 x, anchor bbox[2]는 라벨 오른쪽 경계
   - 단어의 왼쪽이 라벨 오른쪽 경계 이상이면 오른쪽에 있다고 판단
   - sorted(..., key=lambda ...): 선택 단어를 왼쪽 x 기준 오름차순 정렬
   - 이 실습은 왼쪽에서 오른쪽으로 읽는 단순 배치를 전제로 함
   - 같은 줄 뒤에 다른 필드가 이어지는 복잡한 배치는 별도 구역 규칙이 필요

5. raw와 value
   - raw는 선택한 OCR 토큰들을 공백으로 연결한 문자열
   - value는 후속 처리용 값이며 합계만 int로 변환
   - 날짜와 공급자는 문자열로 유지
   - raw도 토큰을 재조합한 값이며 원본 이미지나 전체 OCR 원문 자체는 아님
   - int() 변환 성공이 금액의 업무적 정확성을 검증하는 것은 아님

6. 통합 bbox
   - 각 단어 bbox는 [x1, y1, x2, y2]의 원본 픽셀 경계
   - 왼쪽·위쪽은 최솟값, 오른쪽·아래쪽은 최댓값을 선택
   - 선택한 모든 단어를 둘러싸는 하나의 사각형을 생성
   - 단어 사이 공백도 포함하며 기준 라벨의 박스는 포함하지 않음
   - coordinate_space='original_pixels'로 좌표가 어떤 이미지 기준인지 명시

7. 필드 confidence
   - 선택한 단어 점수 중 최솟값을 필드의 대표 점수로 기록
   - 낮은 점수의 단어를 드러내기 위한 집계 규칙이며 확률 계산이 아님
   - 원문이 맞는지, 라벨과 값의 연결이 맞는지, 업무 의미가 맞는지는 별도 검증 필요

8. 딕셔너리와 JSON
   - fields는 필드 이름을 키로 하여 raw·value·confidence·bbox 등을 보관
   - json.loads(): JSON 문자열 읽기, json.dumps(): Python 객체를 JSON 문자열로 변환
   - ensure_ascii=False: 한글 등 문자 유지, indent=2: 읽기 쉬운 들여쓰기
   - write_text(encoding='utf-8'): 결과 저장. 기존 같은 이름의 파일은 덮어씀

9. 경로와 결과 확인
   - Path.resolve().parent: 현재 소스 폴더, mkdir(exist_ok=True): 출력 폴더 생성
   - 결과 파일은 outputs/p03_evidence.json
   - 마지막 출력은 value만 요약하므로 좌표·점수는 JSON을 열어 확인해야 함
   - 좌표가 저장되었다는 사실만으로 원본과의 시각적 대응 검증이 끝난 것은 아님
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# OCR 기록 읽기와 근거 정보 JSON 저장
import json
# 정규표현식 도구이며 현재 파일에서는 사용하지 않음
import re
# 배열 연산 도구이며 현재 파일에서는 사용하지 않음
import numpy as np
# 표 처리 도구이며 현재 파일에서는 사용하지 않음
import pandas as pd
# 이미지 처리 도구이며 현재 파일에서는 사용하지 않음
import cv2
# 현재 소스 폴더를 입력·출력 경로의 기준으로 사용
base = Path(__file__).resolve().parent
# 제공 데이터와 설정 폴더
data = base / 'data'
# 결과 저장 폴더
out = base / 'outputs'
# 폴더가 없으면 생성
out.mkdir(exist_ok=True)
# UTF-8 JSON을 읽어 문서와 토큰 정보를 담은 Python 객체로 변환
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

# 이번 추출 대상 문서 doc01 선택
doc = lesson['documents']['doc01']
# 단어별 텍스트·계층 번호·좌표·점수 목록
words = doc['ocr']['tokens']
# 필드 이름별 값과 근거를 모을 딕셔너리
fields = {}
# 저장할 필드 이름과 위치를 찾을 기준 라벨을 짝지어 순회
for key, label in [('issued_at','DATE:'), ('supplier','SUPPLIER:'), ('total_amount','TOTAL:')]:
    # 해당 라벨과 text가 정확히 같은 첫 토큰을 기준점으로 선택
    anchor = next(w for w in words if w['text'] == label)
    # 같은 block/paragraph/line이고 라벨 오른쪽에 있는 단어만 선택합니다.
    # TODO A03
    # 목록을 조건으로 추린 뒤, 단어의 왼쪽 x가 작은 순서대로 정렬
    selected = sorted(
        [w for w in words
         # 페이지와 상위 계층까지 같아야 같은 줄로 판단
         if w['page'] == anchor['page']
         and w['block'] == anchor['block']
         and w['paragraph'] == anchor['paragraph']
         and w['line'] == anchor['line']
         # 기준 라벨 자체와 왼쪽 단어를 제외하고 오른쪽 값 토큰만 선택
         and w['bbox'][0] >= anchor['bbox'][2]],
        # 읽는 순서를 왼쪽→오른쪽으로 맞춤
        key=lambda w: w['bbox'][0],
    )
    # TODO 끝 A03

    # 공급자 이름처럼 여러 단어인 값도 공백으로 연결하여 하나의 문자열 생성
    raw = ' '.join(w['text'] for w in selected)

    # 합계는 정수로, 나머지 날짜·공급자는 문자열로 저장
    value = int(raw) if key == 'total_amount' else raw

    # 선택 단어 전체를 둘러싸는 xyxy 박스를 계산: 최소 왼쪽·위, 최대 오른쪽·아래
    box = [min(w['bbox'][0] for w in selected),min(w['bbox'][1] for w in selected),max(w['bbox'][2] for w in selected),max(w['bbox'][3] for w in selected)]

    # 필드 값과 근거 기록. confidence는 단어 점수의 최솟값이며 정답 확률이 아님
    fields[key] = {'raw': raw, 'value': value, 'confidence': min(w['conf'] for w in selected), 'bbox': box, 'coordinate_space':'original_pixels'}

# 필드별 값과 근거를 UTF-8 JSON으로 저장
(out / 'p03_evidence.json').write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding='utf-8')

# items()로 필드 이름과 근거 딕셔너리를 순회하여 value만 뽑은 요약 출력
print({k:v['value'] for k,v in fields.items()})
