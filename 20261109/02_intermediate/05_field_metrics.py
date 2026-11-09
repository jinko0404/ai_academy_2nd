"""
[실습 05 - A05 기본 개념: mock 답변의 값 일치와 문자열 근거 평가]

1. 코드의 목적
   - 제공된 mock 응답을 정답과 비교하고 OCR 원문에서 근거 문자열도 확인
   - 값이 맞는 응답 수와 값·근거가 동시에 확인되는 응답 수를 구분
   - 실제 VLM이나 OCR 엔진을 호출하지 않으며 지연시간을 측정하지 않음

2. 정답은 평가에만 사용
   - item은 이미 준비된 응답이며 field·answer·evidence를 포함
   - doc['truth']는 문서의 평가용 정답 딕셔너리
   - item['field']로 어떤 정답 필드와 비교할지 결정
   - 정답을 이용하여 응답을 생성하거나 고치지 않고 비교 단계에서만 참조

3. A05: 값 완전 일치(exact)
   - item['answer'] == doc['truth'][item['field']]로 같은 필드 값 비교
   - ==는 True 또는 False를 반환하며 문자열의 대소문자·공백을 자동 보정하지 않음
   - 제공 자료는 합계가 정수, 날짜·공급자가 문자열로 준비되어 있음
   - 일반 데이터에서는 비교할 자료형과 정규화 규칙을 먼저 정의해야 함
   - 필드 값 일치는 문서의 모든 문자가 정확하다는 의미가 아님

4. 문자열 근거(evidence_found)
   - evidence 문구가 해당 OCR 원문에 그대로 포함되는지 in으로 검사
   - 근거가 발견되어도 그 문구가 답을 의미상 뒷받침한다는 보장은 없음
   - OCR 오류·공백 차이 때문에 실제 근거가 있어도 검사가 실패할 수 있음
   - 빈 근거 문자열은 항상 포함되므로 일반화할 때는 별도 처리가 필요

5. 값과 근거의 공동 조건
   - exact와 evidence_found는 서로 다른 평가 항목
   - 두 Boolean 열의 &는 같은 행에서 두 조건이 모두 True인지 계산
   - sum()은 True를 1로 계산하여 조건을 만족하는 행 수 집계
   - 이 공동 조건도 의미 검증을 완료한 정답률로 단정할 수 없음

6. review의 의미와 한계
   - review=not grounded로 근거를 못 찾은 응답을 검토 대상으로 표시
   - 현재 정책은 exact를 검토 표시 계산에 사용하지 않음
   - 따라서 값이 틀려도 근거 문구가 있으면 review=False일 수 있음
   - review=False는 정답 확정이나 자동 승인이라는 뜻이 아님

7. 출력과 해석 범위
   - 각 행은 문서·필드 응답 하나이며 문서 수와 전체 행 수는 다름
   - 출력은 일치 수 / 전체 응답 수이며 백분율을 직접 출력하지 않음
   - mock의 기록 평가 결과를 실제 VLM 정확도로 보고하면 안 됨
   - 추론 지연시간 N/A는 미측정·해당 없음을 뜻하며 0ms와 다름

8. 경로와 표 저장
   - Path.resolve().parent: 현재 소스 폴더, mkdir(exist_ok=True): 결과 폴더 생성
   - json.loads(): JSON 문자열을 Python 객체로 변환
   - DataFrame(rows): 응답별 딕셔너리 목록을 표로 구성
   - to_csv(index=False): 행 번호 없이 저장
   - utf-8-sig: BOM을 포함한 UTF-8로 일부 프로그램의 한글 인식에 도움
   - outputs/p05_scores.csv에 저장하며 기존 결과는 덮어씀
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# 문서·응답·정답 기록을 JSON에서 읽음
import json
# 정규표현식 도구이며 현재 파일에서는 사용하지 않음
import re
# 수치 배열 도구이며 현재 파일에서는 사용하지 않음
import numpy as np
# 응답별 평가 결과 표 생성과 CSV 저장
import pandas as pd
# 이미지 처리 도구이며 현재 파일에서는 사용하지 않음
import cv2
# 현재 소스 폴더를 입력·출력 기준으로 사용
base = Path(__file__).resolve().parent
# 제공 기록 폴더
data = base / 'data'
# 평가 결과 저장 폴더
out = base / 'outputs'
# 결과 폴더가 없으면 생성
out.mkdir(exist_ok=True)
# UTF-8 JSON을 읽어 문서별 OCR·mock 응답·정답을 담은 객체로 변환
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

# 문서·필드 응답별 평가 결과를 모을 리스트
rows = []
# 문서 ID와 해당 문서 정보를 함께 순회
for name, doc in lesson['documents'].items():
    # 현재 문서의 미리 작성된 응답을 하나씩 평가
    for item in doc['mock_vlm']['responses']:
        # TODO A05
        # 응답의 field로 정답 키를 선택하고 답과 직접 비교. 응답 생성에는 정답을 사용하지 않음
        exact = item['answer'] == doc['truth'][item['field']]
        # TODO 끝 A05

        # 답의 값과 별개로 제시된 근거 문자열이 같은 문서의 OCR에 있는지 확인
        grounded = item['evidence'] in doc['ocr']['text']

        # 값 일치와 근거 발견을 각각 기록. review는 근거 발견 여부만 반전한 값
        rows.append({'document_id':name, 'field':item['field'], 'backend':'mock', 'exact':exact, 'evidence_found':grounded, 'review':not grounded})
# 평가 결과를 표로 구성하여 행별 두 조건을 비교할 수 있도록 준비
table = pd.DataFrame(rows)
# 상세 평가 결과를 행 번호 없이 CSV 저장
table.to_csv(out / 'p05_scores.csv', index=False, encoding='utf-8-sig')
# 값이 정답과 일치한 응답 수 집계
print('값 일치:', int(table['exact'].sum()), '/', len(table))
# 같은 행에서 값 일치와 문자열 근거 발견이 모두 True인 경우만 집계
print('값과 문자열 근거:', int((table['exact'] & table['evidence_found']).sum()), '/', len(table))
# 실제 모델 실행이 없어 추론 시간을 측정하지 않았음을 명시
print('추론 지연시간: N/A (mock)')
