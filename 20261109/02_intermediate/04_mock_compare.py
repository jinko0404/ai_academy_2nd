"""
[실습 04 - A04 기본 개념: Mock VLM 응답과 OCR 근거 대조]

1. VLM과 mock 응답
   - VLM은 이미지와 텍스트를 함께 다루는 모델
   - mock은 실제 모델 호출 대신 미리 준비한 응답으로 처리 과정을 연습하는 방식
   - 이 파일은 lesson.json의 응답을 읽으며 실제 VLM이나 OCR 엔진을 호출하지 않음
   - 출력 결과를 실제 VLM 정확도·추론 시간으로 보고하면 안 됨

2. 응답의 필드 구성
   - field: 질문 대상 필드 이름, answer: 제시된 답, evidence: 제시된 근거 문구
   - answer 변수는 답 문자열 하나가 아니라 이 정보를 담은 딕셔너리
   - doc['ocr']['text']는 같은 문서의 제공 OCR 원문

3. A04: 문자열 포함 검사
   - 찾을_문자열 in 전체_문자열은 포함 여부를 True 또는 False로 반환
   - answer['evidence']가 해당 문서의 OCR 원문에 그대로 있는지 검사
   - 기본 문자열 비교는 대소문자·공백·줄바꿈 등의 차이를 자동 보정하지 않음
   - 빈 문자열은 모든 문자열에 포함되므로 이 규칙을 일반화할 때는 별도 고려 필요

4. 근거 발견과 정답의 차이
   - 근거 문구가 존재해도 질문에 맞는 값인지, answer와 일치하는지는 별개
   - OCR 원문 자체가 잘못 인식되었을 수도 있음
   - 반대로 OCR 표기 차이 때문에 의미 있는 근거가 있어도 포함 검사가 실패할 수 있음
   - 이 검사는 1차 점검이며 의미 검증이나 업무 규칙 검증을 대신하지 않음

5. evidence_found와 review
   - evidence_found=True: 근거 문자열이 OCR 원문에 존재
   - review=not evidence_found: 문자열을 찾지 못한 항목을 검토 대상으로 표시
   - review=False도 사람이 확인할 필요가 전혀 없거나 정답이라는 뜻은 아님
   - not은 Boolean 값을 반대로 바꾸는 논리 연산자

6. 반복문과 딕셔너리 펼치기
   - documents.items(): 문서 ID와 문서 정보를 함께 순회
   - 내부 반복문은 문서별 여러 mock 응답을 처리
   - **answer는 응답 딕셔너리의 키·값을 새 결과 딕셔너리에 펼침
   - rows.append(): 응답마다 한 행을 기록하며 이미지나 단어 수와 다른 단위

7. 표 저장과 집계
   - DataFrame(rows): 결과 목록을 표로 변환
   - len(table): 전체 응답 행 수
   - Boolean 열의 sum(): True를 1로 계산하여 근거 발견 행 수 집계
   - 이 발견 비율은 답변 정확도가 아님
   - to_csv(index=False): 행 번호 없이 저장, utf-8-sig: BOM을 포함한 UTF-8

8. 파일 경로와 입력 기록
   - Path.resolve().parent: 현재 소스 폴더, mkdir(exist_ok=True): 결과 폴더 생성
   - read_text(): UTF-8 문자열 읽기, json.loads(): JSON을 Python 객체로 변환
   - outputs/p04_mock_review.csv에 저장하며 같은 이름의 기존 결과는 덮어씀
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# 제공 OCR·mock 응답을 JSON에서 읽음
import json
# 정규표현식 도구이며 현재 파일에서는 사용하지 않음
import re
# 배열 연산 도구이며 현재 파일에서는 사용하지 않음
import numpy as np
# 응답별 결과 표 작성과 CSV 저장
import pandas as pd
# 이미지 처리 도구이며 현재 파일에서는 사용하지 않음
import cv2
# 현재 소스 폴더를 기준 경로로 사용
base = Path(__file__).resolve().parent
# 제공 기록이 들어 있는 폴더
data = base / 'data'
# 결과 저장 폴더
out = base / 'outputs'
# 폴더가 없으면 생성하고 이미 있으면 그대로 사용
out.mkdir(exist_ok=True)
# 설정 파일을 UTF-8로 읽어 문서별 정보를 담은 Python 객체로 변환
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

# 응답별 근거 검사 결과를 모을 리스트
rows = []
# 문서 ID(name)와 해당 문서 정보(doc)를 함께 가져옴
for name, doc in lesson['documents'].items():
    # 현재 문서에 준비된 모의 응답을 하나씩 처리
    for answer in doc['mock_vlm']['responses']:
        # 문자열 포함 검사는 1차 점검일 뿐 의미상 올바른 답임을 증명하지 않습니다.
        # TODO A04
        # 응답이 제시한 evidence 문구가 같은 문서의 OCR 전체 문자열에 있는지 검사
        # 답 문자열 answer['answer']의 정답 여부를 직접 검사하는 것은 아님
        evidence_found = answer['evidence'] in doc['ocr']['text']
        # TODO 끝 A04
        # 원래 응답의 field·answer·evidence를 **answer로 펼쳐 함께 저장
        # 근거가 없으면 not False가 True가 되어 review에 검토 필요 표시
        rows.append({'document_id':name, 'backend':'mock', **answer, 'evidence_found':evidence_found, 'review':not evidence_found})
# 모든 문서의 응답별 결과를 하나의 표로 구성
table = pd.DataFrame(rows)
# 근거 발견 여부와 검토 표시를 행 번호 없이 CSV 저장
table.to_csv(out / 'p04_mock_review.csv', index=False, encoding='utf-8-sig')
# 전체 응답 수와 근거 문자열 발견 수 출력. 발견 수는 정확한 답의 수와 다름
print('mock 필드:', len(table), '근거 문자열 발견:', int(table['evidence_found'].sum()))
# 실제 모델 평가와 구분하도록 데이터 출처와 해석 범위 명시
print('모의 응답의 평가 연습이며 실제 VLM 정확도가 아닙니다.')
