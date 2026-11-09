"""실습 04: 이미지+질문을 구성하고 모의 응답을 읽습니다. 실제 VLM을 호출하지 않습니다."""
"""
[실습 04 - B05 기본 개념: 이미지·질문 메시지와 모의 응답 기록]

1. VLM과 멀티모달 입력
   - VLM은 이미지와 텍스트를 함께 다루는 모델
   - 문서 이미지와 질문을 함께 입력하여 날짜·공급자·합계 등을 물을 수 있음
   - 이 파일은 입력 구조만 구성하며 이미지 내용을 실제 모델로 분석하지 않음

2. messages와 role
   - messages는 대화 메시지 딕셔너리를 담는 리스트
   - role은 메시지의 발화자 역할이며 user는 사용자 입력을 의미
   - 이번 실습에서는 질문마다 user 메시지 하나를 새로 구성
   - 이전 질문을 계속 누적하는 대화 기록은 아님

3. content의 이미지·텍스트 항목
   - content는 한 메시지 안의 여러 입력 항목을 담는 리스트
   - type='image', path=...: 문서 이미지 파일의 위치를 기록
   - type='text', text=...: 해당 문서에 물어볼 질문 문자열을 기록
   - 여기의 형식은 실습용 구조이며 실제 서비스·라이브러리마다 입력 규격이 다를 수 있음
   - 경로 문자열 저장만으로 이미지가 읽히거나 업로드되지는 않음

4. field와 질문 선택
   - field는 issued_at, supplier, total_amount 같은 추출 대상 이름
   - answer['field']를 키로 questions에서 해당 질문을 조회
   - answer 변수는 응답 문자열이 아니라 field·answer·evidence 등이 담긴 딕셔너리
   - answer['answer']가 미리 준비된 답 문자열

5. mock과 실제 추론의 구분
   - mock은 실제 모델 대신 교사가 준비한 응답으로 흐름을 연습하는 방식
   - 이 파일은 generate나 외부 API를 호출하지 않음
   - 출력한 답은 messages에서 새로 생성한 결과가 아니라 lesson.json의 제공 기록
   - 따라서 이 결과로 실제 VLM 정확도·처리 속도를 평가할 수 없음

6. 메타데이터와 None
   - backend='mock'으로 모의 응답이라는 출처를 명시
   - model_id=None: 실제 호출한 모델 ID가 없음
   - inference_ms=None: 실제 추론 시간을 측정하지 않음
   - None은 0과 다르며 JSON 저장 시 null로 표현
   - 실제 추론하지 않았는데 시간 0ms를 측정했다고 기록하면 안 됨

7. 리스트·딕셔너리와 결과 저장
   - rows.append(): 질문별 메시지·응답·메타데이터를 한 레코드로 추가
   - json.loads(): JSON 문자열을 Python 객체로 변환
   - json.dumps(): Python 객체를 JSON 문자열로 변환
   - ensure_ascii=False: 한글 등 문자 유지, indent=2: 들여쓰기 2칸
   - write_text(encoding='utf-8'): 문자열을 파일로 저장하며 기존 결과는 덮어씀

8. 경로와 해석 범위
   - Path.resolve().parent: 현재 소스 파일이 들어 있는 폴더
   - str(base / doc['file']): 문서 이미지 경로를 문자열로 표현
   - 결과 파일은 p04_mock_answers.json이며 원본 이미지·설정은 변경하지 않음
   - 파일 생성 성공과 실제 이미지 해석·답변 검증 성공은 서로 다른 단계
"""

# 파일 및 폴더 경로 관리
from pathlib import Path

# 제공 설정 읽기와 결과 JSON 문자열 생성
import json

# 터미널 실행 위치와 관계없이 현재 소스 폴더를 기준으로 경로 생성
base = Path(__file__).resolve().parent

# 문서 정보·질문·모의 응답이 들어 있는 UTF-8 JSON을 읽음
lesson = json.loads((base / 'lesson.json').read_text(encoding='utf-8'))
# 메시지에 연결할 문서 doc01의 정보 선택
doc = lesson['documents']['doc01']

# 질문별 입력 구조와 모의 응답을 모을 리스트
rows = []

# doc01에 제공된 모의 응답을 하나씩 처리
for answer in doc['mock_vlm']['responses']:
    # 응답의 field에 대응하는 질문을 질문 사전에서 조회
    question = lesson['questions'][answer['field']]
    # TODO B05
    # 질문 하나마다 이미지·텍스트가 포함된 새 user 메시지 목록 생성
    messages = [
        {
            # 이 메시지는 사용자 입력임을 표시
            'role': 'user',
            # 동일 메시지에 이미지 위치와 질문 문자열을 함께 기록
            'content': [
                # 이미지 파일의 경로만 기록하며 이 줄에서 이미지를 읽지는 않음
                {'type': 'image', 'path': str(base / doc['file'])},
                # 현재 필드에 대해 물어볼 질문
                {'type': 'text', 'text': question},
            ],
        }
    ]
    # TODO 끝 B05

    # 입력 메시지와 제공 응답을 함께 보관. 실제 모델·실제 추론 시간은 없으므로 None
    rows.append({'backend': 'mock', 'messages': messages, 'response': answer, 'model_id': None, 'inference_ms': None})

    # 질문과 미리 작성된 답을 출력. 모델이 방금 생성한 답이 아님
    print(question, '->', answer['answer'])

# 전체 레코드를 읽기 쉬운 UTF-8 JSON으로 저장. Python None은 JSON null로 변환
(base / 'p04_mock_answers.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

# 결과 출처를 명시하여 실제 VLM 실행으로 오해하지 않도록 안내
print('MOCK: 교사가 작성한 응답 예시입니다. 실제 모델 출력이 아닙니다.')
