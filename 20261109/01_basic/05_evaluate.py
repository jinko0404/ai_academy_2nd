"""실습 05: 5문서×3필드의 일치율을 구합니다. 정답은 평가할 때만 읽습니다."""
"""
[실습 05 - B06 기본 개념: OCR 추출 필드의 완전 일치 평가]

1. 평가 대상과 입력 출처
   - lesson.json에 제공된 문서별 OCR 기록을 읽어 필드를 추출하고 평가
   - 실제 OCR 엔진이나 VLM을 새로 호출하지 않음
   - 앞 실습에서 저장한 결과 파일을 자동으로 읽는 것도 아님
   - 문서 5개에서 날짜·공급자·합계 3개씩 총 15개 필드를 평가

2. 추출과 정답의 분리
   - predicted는 OCR 원문에서 정규표현식으로 얻은 값
   - truth는 정답이며 추출값과 비교하는 평가 단계에서만 사용
   - 정답을 추출에 사용하면 실제 추출 능력을 평가할 수 없음
   - lesson 전체를 메모리에 읽는 것과 정답을 추출 로직에 이용하는 것은 구분

3. 정규표현식과 캡처 그룹
   - re.search(): 문자열에서 패턴과 처음 일치하는 부분을 찾음
   - group(1): 첫 번째 괄호 안에 해당하는 문자열 추출
   - 날짜 패턴은 숫자 4개-2개-2개 형식을 찾음
   - SUPPLIER 뒤의 문자를 공급자로, TOTAL 뒤의 숫자를 합계로 선택
   - int(): 합계 문자열을 정수로 변환
   - 정상 자료에 패턴이 존재한다는 전제이며 누락·다양한 금액 형식 처리는 별도 필요

4. 딕셔너리와 필드별 조회
   - issued_at: 날짜, supplier: 공급자, total_amount: 합계
   - predicted[field]와 doc['truth'][field]로 같은 의미의 값을 선택
   - 문자열은 문자열끼리, 합계는 정수끼리 비교하도록 자료형을 맞춤
   - 날짜 모양이 맞아도 실제 달력에 존재하는 날짜인지는 이 코드에서 검사하지 않음

5. B06: 완전 일치 비교
   - == 연산자로 추출값과 정답값의 동일 여부를 판단
   - 같으면 True, 다르면 False를 correct에 저장
   - 문자열은 대소문자·띄어쓰기 차이를 자동으로 보정하지 않음
   - 오타가 조금만 있어도 해당 필드는 불일치로 계산

6. 필드 일치율과 다른 지표의 차이
   - 필드 일치율 = 일치한 필드 수 / 전체 평가 필드 수
   - 문서별 정답률이나 모든 문자의 OCR 정확도와는 다른 지표
   - 날짜·공급자·합계가 맞아도 다른 본문에 인식 오류가 있을 수 있음
   - 합성 영어 문서 5개의 결과를 일반 문서·한국어 성능으로 일반화하면 안 됨

7. 결과 표와 Boolean 집계
   - rows에는 문서·필드 조합마다 한 행씩 저장
   - DataFrame(rows): 결과 목록을 표로 변환
   - correct 열의 sum()은 True를 1로 계산하여 일치한 필드 수 집계
   - len(table)은 전체 평가 필드 수이며 마지막 출력은 분자·분모를 표시
   - 이 코드는 백분율을 별도로 계산해 출력하지 않음

8. 경로와 CSV 저장
   - Path.resolve().parent: 현재 소스 폴더를 기준 경로로 사용
   - json.loads(): JSON 문자열을 Python 객체로 변환
   - to_csv(index=False): DataFrame 행 번호 없이 저장
   - utf-8-sig: BOM을 포함한 UTF-8이며 일부 프로그램의 한글 인식에 도움
   - 결과는 p05_field_scores.csv에 저장하며 재실행하면 덮어씀
"""

# 파일 및 폴더 경로 관리
from pathlib import Path
# 제공 OCR 기록과 평가용 정답을 JSON에서 읽음
import json
# 날짜·공급자·합계를 원문에서 패턴으로 추출
import re
# 필드별 평가 결과를 표로 구성하여 CSV 저장
import pandas as pd

# 터미널 실행 위치와 관계없이 현재 소스 폴더를 기준으로 사용
base = Path(__file__).resolve().parent

# UTF-8 설정 파일을 읽어 문서별 OCR 기록과 정답을 포함한 Python 객체로 변환
lesson = json.loads((base / 'lesson.json').read_text(encoding='utf-8'))
# 문서·필드별 평가 결과를 모을 리스트
rows = []
# 문서 ID(name)와 해당 문서 정보(doc)를 함께 순회
for name, doc in lesson['documents'].items():
    # 추출 입력은 정답이 아니라 기록된 OCR 원문
    text = doc['ocr']['text']
    # DATE의 날짜, SUPPLIER의 이름, TOTAL의 숫자를 각각 캡처
    # 합계만 int()로 바꾸고 날짜·공급자는 문자열로 유지
    predicted = {'issued_at': re.search(r'DATE: (\d{4}-\d{2}-\d{2})', text).group(1), 'supplier': re.search(r'SUPPLIER: (.+)', text).group(1), 'total_amount': int(re.search(r'TOTAL: (\d+)', text).group(1))}
    # 현재 문서에서 평가할 세 필드를 순서대로 처리
    for field in ['issued_at', 'supplier', 'total_amount']:
        # TODO B06
        # 같은 필드의 추출값과 정답을 비교하여 True 또는 False로 기록
        correct = predicted[field] == doc['truth'][field]
        # TODO 끝 B06
        # 어떤 문서의 어떤 필드가 맞거나 틀렸는지 추적할 수 있도록 행 단위 저장
        rows.append({'document_id': name, 'field': field, 'predicted': predicted[field], 'correct': correct})
# 문서 5개 × 필드 3개인 평가 결과 15행을 표로 구성
table = pd.DataFrame(rows)
# 추출값과 일치 여부를 행 번호 없이 CSV 저장
table.to_csv(base / 'p05_field_scores.csv', index=False, encoding='utf-8-sig')
# 평가한 문서 수와 필드 수를 구분하여 출력
print('문서 수:', len(lesson['documents']), '필드 수:', len(table))
# True 개수를 합산하여 일치 필드 수 / 전체 필드 수 출력
print('필드 일치:', int(table['correct'].sum()), '/', len(table))
# 실제 평가한 데이터 범위를 명시하여 일반 OCR 성능과 구분
print('합성 영어 문서의 기록 비교이며 일반 OCR·한국어 성능이 아닙니다.')
