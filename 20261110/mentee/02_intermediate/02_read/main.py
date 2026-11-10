"""
[02_intermediate / 02_read / main.py 기본 개념]

1. FastAPI와 ASGI 서버
   - FastAPI 앱에 요청 경로와 함수를 등록하고 Uvicorn으로 서버를 실행함
   - 이 폴더에서 python -m uvicorn main:app --reload 실행
   - main은 파일 이름, app은 앱 객체, reload는 개발 중 변경 감지 옵션
2. HTTP와 JSON
   - GET은 조회, POST는 새 항목 생성에 사용
   - dict와 list 반환값은 FastAPI가 JSON 응답으로 변환
   - 200은 정상 조회, 201은 생성 성공, 404는 항목 없음, 422는 요청 검증 실패
   - /docs는 자동 API 문서이며 /health는 경로 응답 확인일 뿐 DB·AI 상태 검증이 아님
3. 경로·쿼리·Body
   - /{knowledge_id}는 경로 변수이며 int 타입으로 검증
   - name과 category는 쿼리 인자, 기본값이 있으면 생략 가능
   - KnowledgeWrite는 POST JSON Body의 형식·문자열 길이 제약을 정의
   - 아래 코드에 등록된 경로만 해당 실습에서 사용할 수 있음
4. 조회와 필터
   - entries는 사용자에게 반환할 목록, next_id는 내부 ID 발급 상태
   - category는 정확히 일치하는 항목만 선택하며 결과가 없으면 200과 []
   - 상세 조회에서 정수 ID가 없으면 404, 정수가 아닌 경로는 타입 검증으로 422
5. 생성과 저장
   - model_dump()로 검증된 입력을 dict로 바꾸고 서버가 ID를 부여
   - 목록 append와 next_id 증가 후 전체 data를 저장해야 재실행에도 보존
   - response_model은 응답 구조를 검증하며 파일 저장을 대신하지 않음
   - Field의 길이 제약은 공백 문자열의 업무적 유효성까지 판단하지 않음
6. 파일 잠금의 범위
   - LOCK으로 읽기부터 ID 발급·쓰기까지 묶어 같은 프로세스의 동시 접근을 보호
   - 프로세스 간 잠금이나 DB 트랜잭션은 아니며 여러 worker 운용에는 별도 설계 필요
   - 이 실습은 JSON 원본 지식 관리이며 인증·DB·LLM 호출을 수행하지 않음
"""

"""JSON 목록·상세 조회. 실행: python -m uvicorn main:app --reload"""
# 앱 생성과 HTTP 오류 도구를 가져옴. Response 등 일부 도구는 후속 교시용으로 남아 있을 수 있음
from fastapi import FastAPI, HTTPException, Response
# 현재 교시의 파일 잠금과 읽기·쓰기 함수를 가져옴
from storage import LOCK, read_data, write_data

# 앱 객체 생성. title과 description은 자동 API 문서에 표시
app = FastAPI(title="Knowledge API", description="AI 플랫폼의 원본 지식 관리 연습")

# GET 경로와 바로 아래 함수를 연결하는 데코레이터
@app.get("/health")
# 요청 처리 또는 보조 함수 정의. 타입 표기는 입력·출력 계약을 표현
def health():
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    # 요청 처리 함수가 반환한 dict를 JSON 응답으로 전달
    return {"status": "ok"}

# GET 경로와 바로 아래 함수를 연결하는 데코레이터
@app.get("/hello")
# 요청 처리 또는 보조 함수 정의. 타입 표기는 입력·출력 계약을 표현
def hello(name: str = "수강생"):
    # 쿼리 name을 포함하는 인사말을 JSON으로 반환
    return {"message": f"안녕하세요, {name}"}

# GET 경로와 바로 아래 함수를 연결하는 데코레이터
@app.get("/api/knowledge")
# 요청 처리 또는 보조 함수 정의. 타입 표기는 입력·출력 계약을 표현
def list_knowledge(category: str | None = None):
    # 같은 프로세스에서 공유하는 잠금을 획득하고 블록 종료 시 해제
    with LOCK:
        # 현재 JSON 상태를 읽음. 저장된 entries와 next_id를 함께 가져옴
        data = read_data()
    # 필터링할 사용자 항목 목록 선택
    result = data["entries"]
    # 쿼리 인자가 전달된 경우만 필터 적용. 생략과 빈 문자열은 다름
    if category is not None:
        # BEGIN A02-01
        # TODO A02-01: category와 정확히 일치하는 항목만 result에 남기세요. 일치 항목이 없으면 빈 목록이어야 합니다.
        # 힌트: for 반복문으로 새 목록을 만들어도 됩니다.
        # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
        # category가 정확히 같은 항목으로 새 목록 생성
        result = [entry for entry in result if entry["category"] == category]
        # END A02-01
    # 조건에 맞는 지식이 없으면 정상 조회 200과 빈 배열 []입니다.
    # 조회 결과 반환. 일치 항목이 없으면 빈 JSON 배열
    return result

# GET 경로와 바로 아래 함수를 연결하는 데코레이터
@app.get("/api/knowledge/{knowledge_id}")
# 요청 처리 또는 보조 함수 정의. 타입 표기는 입력·출력 계약을 표현
def get_knowledge(knowledge_id: int):
    # 같은 프로세스에서 공유하는 잠금을 획득하고 블록 종료 시 해제
    with LOCK:
        # 현재 JSON 상태를 읽음. 저장된 entries와 next_id를 함께 가져옴
        data = read_data()
    # BEGIN A02-02
    # TODO A02-02: entries를 순회하여 id가 knowledge_id인 지식을 반환하세요. 찾지 못하면 반복문 다음으로 진행하세요.
    # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
    # 저장된 항목을 순서대로 확인
    for entry in data["entries"]:
        # 현재 항목이 요청한 정수 ID와 같은지 검사
        if entry["id"] == knowledge_id:
            # 찾거나 생성한 항목을 응답. 저장 동작과 반환 동작은 별개
            return entry
    # END A02-02
    # 입력은 정수이지만 저장된 항목이 없으면 404입니다. foo 같은 Path는 먼저 422입니다.
    # BEGIN A02-03
    # TODO A02-03: 반복문에서 찾지 못한 경우 HTTPException으로 404와 Knowledge not found를 전달하세요.
    # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
    # 탐색 후 항목이 없으면 404와 detail 메시지를 응답
    raise HTTPException(404, "Knowledge not found")
    # END A02-03
