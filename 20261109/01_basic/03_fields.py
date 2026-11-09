"""실습 03: 실제 OCR 기록에서 필요한 3개 필드를 추출합니다. 정상 자료만 사용합니다."""
"""
[실습 03 - B03·B04 기본 개념: OCR 원문에서 필드 추출과 JSON 저장]

1. OCR과 필드 추출
   - OCR은 이미지에서 문자를 인식하고, 필드 추출은 그 문자열에서 필요한 값을 선택
   - 이 파일은 lesson.json에 제공된 OCR 기록을 사용하며 엔진을 새로 호출하지 않음
   - 앞 실습의 p02_words.csv를 자동으로 읽는 코드도 아님
   - 날짜·공급자·합계를 이름 있는 필드로 정리하여 후속 처리에 사용

2. 정규표현식과 re.search()
   - 정규표현식은 문자열에서 특정 형식을 찾기 위한 패턴
   - re.search(패턴, 문자열, 옵션): 처음 일치한 부분의 Match 객체 반환
   - 일치하는 내용이 없으면 None을 반환
   - 이 실습은 정상 자료에 해당 줄이 존재한다는 전제이므로 예외 처리를 추가하지 않음

3. 줄 시작·끝과 MULTILINE
   - ^는 시작, $는 끝을 나타내며 re.MULTILINE에서는 각 줄의 시작·끝에 적용
   - DATE:, SUPPLIER:, TOTAL:로 시작하는 줄을 각각 구분하여 검색
   - 문서 전체에서 첫 숫자를 가져오면 문서 번호·날짜 등을 합계로 오인할 수 있음

4. 캡처 그룹과 group(1)
   - 괄호 (...)는 추출할 부분을 캡처 그룹으로 지정
   - group(0)은 패턴 전체의 일치 내용, group(1)은 첫 번째 괄호 안의 내용
   - (.+)는 줄바꿈을 제외한 문자가 하나 이상 이어지는 부분을 의미
   - DATE: 뒤의 내용을 추출하는 것만으로 실제 달력 날짜가 유효한지 검증하지는 않음

5. 숫자 패턴과 정수 변환
   - \\d+는 숫자 하나 이상, \\s*는 공백 문자 0개 이상을 의미
   - TOTAL: 줄의 숫자 부분만 group(1)로 추출한 뒤 int()로 정수 변환
   - 이 패턴은 제공된 정수 합계 형식용이며 쉼표·소수점·통화 기호 형식은 별도 고려 필요
   - r'...'는 역슬래시를 정규표현식에 전달하기 편리한 raw 문자열 표기

6. 구조화된 레코드와 원문 보존
   - 딕셔너리의 키로 document_id, issued_at 등 필드 의미를 명시
   - records.append(): 문서별 딕셔너리를 결과 리스트에 추가
   - raw_text로 OCR 원문을 함께 저장하여 추출 근거와 오류를 추적
   - currency='KRW'와 ocr_source는 코드에 지정한 값이며 원문에서 자동 추출한 것이 아님

7. JSON 읽기와 저장
   - json.loads(): JSON 문자열을 Python 객체로 변환
   - json.dumps(): Python 객체를 JSON 문자열로 변환
   - ensure_ascii=False: 한글 등을 유니코드 이스케이프 대신 그대로 표현
   - indent=2: 들여쓰기 2칸으로 읽기 쉽게 표시
   - write_text(encoding='utf-8'): UTF-8 파일로 저장하며 기존 결과는 덮어씀

8. 해석 범위
   - 필드가 추출되었다고 OCR 원문이나 업무 의미까지 정확하다는 뜻은 아님
   - 원본 이미지와 대조하고 날짜 형식·항목 합계·통화 등 업무 규칙을 별도 검증해야 함
   - Path.resolve().parent로 현재 소스 폴더를 입력·출력 기준으로 사용
"""

# 파일 및 폴더 경로 관리
from pathlib import Path

# 제공 OCR 기록 읽기와 결과 JSON 문자열 생성
import json

# 줄별 패턴에서 필요한 값을 찾는 정규표현식 모듈
import re

# 터미널 위치와 관계없이 현재 소스 폴더를 기준 경로로 사용
base = Path(__file__).resolve().parent
# 설정 파일을 UTF-8로 읽고 문서별 정보가 담긴 documents 항목 선택
docs = json.loads((base / 'lesson.json').read_text(encoding='utf-8'))['documents']

# 문서마다 추출한 필드를 모을 리스트
records = []

# 제공된 문서 3개의 기록을 순서대로 처리
for name in ['doc01', 'doc02', 'doc03']:
    # 현재 문서의 기록된 OCR 전체 문자열 조회
    text = docs[name]['ocr']['text']
    # DATE 줄 뒤의 날짜 부분을 추출합니다. OCR 원문도 함께 보존합니다.
    # TODO B03
    # ^DATE: 로 시작하는 줄의 내용을 찾아 첫 캡처 그룹인 날짜 문자열 추출
    # MULTILINE이 있어 문서 첫 줄이 아니어도 DATE 줄을 찾을 수 있음
    issued_at = re.search(r'^DATE: (.+)$', text, re.MULTILINE).group(1)
    # TODO 끝 B03

    # SUPPLIER: 뒤의 공급자 이름을 추출. 이름에 포함된 공백도 함께 유지
    supplier = re.search(r'^SUPPLIER: (.+)$', text, re.MULTILINE).group(1)
    # TODO B04
    # TOTAL:로 시작하는 줄에서 숫자만 캡처하고 끝의 공백은 허용
    # group(1)은 숫자 문자열이므로 int()로 변환하여 합계를 수치로 저장
    total_amount = int(re.search(r'^TOTAL: (\d+)\s*$', text, re.MULTILINE).group(1))
    # TODO 끝 B04

    # 추출 필드와 고정 메타데이터, 근거 OCR 원문을 문서별 딕셔너리로 기록
    records.append({'document_id': name, 'issued_at': issued_at, 'supplier': supplier, 'total_amount': total_amount, 'currency': 'KRW', 'ocr_source': 'recorded Tesseract', 'raw_text': text})

    # 현재 문서의 ID, 날짜, 공급자, 합계를 터미널에 출력
    print(name, issued_at, supplier, total_amount)

# 전체 결과를 한글이 보존되는 들여쓰기 JSON으로 변환해 별도 파일에 저장
(base / 'p03_fields.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
