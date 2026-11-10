"""
[실습 04 기본 개념: PUT 수정과 기존 ID 유지]

1. HTTP API와 FastAPI
   - 데코레이터로 HTTP 메서드·경로와 처리 함수를 연결
   - GET은 조회, POST는 생성, PUT은 지정한 항목의 입력 필드 전체 교체
   - dict·list 반환값은 JSON 응답으로 전달
   - 이 폴더에서 python -m uvicorn main:app --reload로 실행

2. 경로·쿼리·Body
   - knowledge_id는 URL 경로의 정수 ID이며 목록 위치와 다름
   - payload는 JSON Body를 검증한 KnowledgeWrite 객체
   - 응용의 name·category는 쿼리 인자이며 기본필수에는 해당 경로·필터가 없음
   - 아래 실제 등록된 경로와 함수 인자를 기준으로 요청해야 함

3. 입력·응답 모델
   - BaseModel과 Field로 필수 문자열과 길이 제약을 정의
   - 수정에도 title·content·category 세 필드가 모두 필요
   - 공백만 있는 문자열의 업무적 유효성까지 검사하는 것은 아님
   - Knowledge는 입력 필드를 상속하고 정수 id를 추가한 응답 모델
   - response_model은 응답 계약이며 파일 저장을 대신하지 않음

4. data, entries, entry, i 구분
   - data는 파일 전체 dict이며 entries 목록과 next_id를 포함
   - data['entries']는 저장된 지식들의 리스트
   - entry는 그 목록에서 꺼낸 개별 dict로 id·title·content·category를 포함
   - enumerate는 목록 위치 i와 해당 항목 entry를 함께 반환
   - i=0은 첫 번째 위치이며 항목의 id가 반드시 0이라는 뜻은 아님

5. 수정 대상 찾기와 교체
   - entry['id'] == knowledge_id로 현재 항목이 수정 대상인지 비교
   - payload.model_dump()로 새 입력을 dict로 만들고 기존 ID를 붙임
   - data['entries'][i] = updated로 원래 위치의 항목을 교체
   - i = updated는 위치 변수만 바꾸며 목록을 교체하지 못함
   - entry에는 entries 키가 없으므로 entry['entries']로 접근하지 않음

6. 생성과 수정의 차이
   - POST는 append로 항목을 추가하고 next_id를 증가시킴
   - 이 실습의 PUT은 항목 수와 ID·next_id를 유지하고 기존 항목만 교체
   - 없는 ID를 새로 만들지 않으며 404를 반환
   - 같은 내용을 반복 PUT해도 항목을 추가하지 않는 멱등적 동작

7. 잠금과 영속화
   - with LOCK으로 읽기→대상 탐색→교체→파일 저장을 하나의 잠금 구간으로 처리
   - write_data(data)는 메모리의 수정 결과를 실제 JSON 파일에 저장
   - write_data(data=data)도 같은 의미의 키워드 인자 호출
   - threading.Lock은 같은 프로세스의 스레드를 보호하며 DB 트랜잭션·프로세스 간 잠금은 아님

8. 상태코드와 실패 처리
   - GET·PUT 정상 처리 200, POST 생성 성공 201
   - 정수 ID가 없으면 HTTPException으로 404와 detail 메시지 반환
   - 정수가 아닌 경로 또는 누락·길이 위반 Body는 요청 검증에서 422
   - return updated는 저장 성공 뒤 수행하며 응답만 반환한다고 저장되지는 않음

9. 실습 범위
   - 교시별 storage.py가 이 폴더의 JSON 파일을 읽고 저장
   - /health는 경로 응답 확인이며 DB·AI 모델의 전체 준비 상태 검사가 아님
   - 이 실습에는 인증·실제 DB·LLM 호출이 포함되지 않음
"""

"""같은 지식의 본문 수정. 실행: python -m uvicorn main:app --reload"""
# 앱과 HTTP 오류 응답 도구를 가져옴. Response는 현재 직접 사용하지 않음
from fastapi import FastAPI, HTTPException, Response
# 현재 교시의 파일 읽기·쓰기와 공유 잠금 도구
from storage import LOCK, read_data, write_data

# 요청·응답 모델 및 문자열 길이 검증 도구
from pydantic import BaseModel, Field

# 입력 모델은 HTTP Body의 규칙입니다. DB 테이블이나 저장 코드가 아닙니다.
# POST와 PUT에서 공통으로 받는 필수 입력 구조 정의
class KnowledgeWrite(BaseModel):
    # 필수 문자열과 길이 제한. 요청 검증 실패 시 처리 함수 실행 전에 422
    title: str = Field(min_length=1, max_length=80)
    # 필수 문자열과 길이 제한. 요청 검증 실패 시 처리 함수 실행 전에 422
    content: str = Field(min_length=1, max_length=2000)
    # 필수 문자열과 길이 제한. 요청 검증 실패 시 처리 함수 실행 전에 422
    category: str = Field(min_length=1, max_length=30)


# 입력 필드를 상속하고 응답용 ID를 추가
class Knowledge(KnowledgeWrite):
    # 서버가 관리하는 정수 ID이며 클라이언트가 새 입력으로 정하는 값이 아님
    id: int


# 웹 앱 객체 생성. 제목과 설명은 /docs의 자동 문서에도 표시
app = FastAPI(title="Knowledge API", description="AI 플랫폼의 원본 지식 관리 연습")

# 조회 메서드 GET과 아래 처리 함수를 해당 URL에 등록
@app.get("/health")
# 요청 처리 함수 정의. 경로·쿼리·Body의 구분은 함수 인자와 모델 타입에 따라 결정
def health():
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    # 이 경로가 응답 가능한지 JSON으로 표시
    return {"status": "ok"}

# 조회 메서드 GET과 아래 처리 함수를 해당 URL에 등록
@app.get("/hello")
# 요청 처리 함수 정의. 경로·쿼리·Body의 구분은 함수 인자와 모델 타입에 따라 결정
def hello(name: str = "수강생"):
    # 쿼리 name을 포함한 인사말 반환
    return {"message": f"안녕하세요, {name}"}

# 조회 메서드 GET과 아래 처리 함수를 해당 URL에 등록
@app.get("/api/knowledge")
# 요청 처리 함수 정의. 경로·쿼리·Body의 구분은 함수 인자와 모델 타입에 따라 결정
def list_knowledge(category: str | None = None):
    # 공유 잠금을 획득하고 블록 종료 시 해제. 수정은 읽기부터 저장까지 같은 구간에서 수행
    with LOCK:
        # entries 목록과 next_id를 포함하는 전체 JSON 상태를 읽음
        data = read_data()
    # 응답과 필터링에 사용할 항목 목록 선택
    result = data["entries"]
    # category를 생략한 경우와 전달한 경우를 구분
    if category is not None:
        # 분류 문자열이 정확히 같은 항목만 새 목록에 포함
        result = [entry for entry in result if entry["category"] == category]
    # 조건에 맞는 지식이 없으면 정상 조회 200과 빈 배열 []입니다.
    # 일치한 목록 반환. 결과가 없으면 정상 200과 빈 배열
    return result

# 조회 메서드 GET과 아래 처리 함수를 해당 URL에 등록
@app.get("/api/knowledge/{knowledge_id}")
# 요청 처리 함수 정의. 경로·쿼리·Body의 구분은 함수 인자와 모델 타입에 따라 결정
def get_knowledge(knowledge_id: int):
    # 공유 잠금을 획득하고 블록 종료 시 해제. 수정은 읽기부터 저장까지 같은 구간에서 수행
    with LOCK:
        # entries 목록과 next_id를 포함하는 전체 JSON 상태를 읽음
        data = read_data()
    # 상세 조회 대상 ID를 찾기 위해 개별 항목 순회
    for entry in data["entries"]:
        # 개별 항목의 ID와 요청 ID를 비교하여 True·False 생성
        same_id = entry["id"] == knowledge_id
        # ID가 같은 대상에 대해서만 반환 또는 수정 수행
        if same_id:
            # 조회하거나 생성한 항목을 JSON 응답으로 반환
            return entry
    # 입력은 정수이지만 저장된 항목이 없으면 404입니다. foo 같은 Path는 먼저 422입니다.
    # 반복문을 끝까지 돌아도 대상을 못 찾으면 404. 데이터 생성·저장은 하지 않음
    raise HTTPException(404, "Knowledge not found")

# 새 지식 등록 경로. 성공 시 201, 응답은 Knowledge로 검증
@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
# 요청 처리 함수 정의. 경로·쿼리·Body의 구분은 함수 인자와 모델 타입에 따라 결정
def create_knowledge(payload: KnowledgeWrite):
    # 하나의 잠금으로 현재 데이터 읽기 → ID 발급 → 저장을 함께 보호합니다.
    # 공유 잠금을 획득하고 블록 종료 시 해제. 수정은 읽기부터 저장까지 같은 구간에서 수행
    with LOCK:
        # entries 목록과 next_id를 포함하는 전체 JSON 상태를 읽음
        data = read_data()
        # 검증된 생성 입력을 수정 가능한 dict로 변환
        entry = payload.model_dump()
        # 생성할 항목에 현재 발급 가능한 새 ID 부여
        entry["id"] = data["next_id"]
        # POST 생성에서만 새 항목을 목록 끝에 추가
        data["entries"].append(entry)
        # 다음 생성에 쓸 ID 증가. PUT 수정에서는 실행하지 않음
        data["next_id"] += 1
        # 수정된 전체 상태를 파일로 저장
        write_data(data)
    # 201을 선언하는 것만으로 저장되지는 않습니다. 실제 저장 성공 뒤 응답합니다.
    # 조회하거나 생성한 항목을 JSON 응답으로 반환
    return entry

# 기존 지식 전체 수정 경로. 경로 ID로 대상을 지정하고 응답은 Knowledge로 검증
@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
# 요청 처리 함수 정의. 경로·쿼리·Body의 구분은 함수 인자와 모델 타입에 따라 결정
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    # 공유 잠금을 획득하고 블록 종료 시 해제. 수정은 읽기부터 저장까지 같은 구간에서 수행
    with LOCK:
        # entries 목록과 next_id를 포함하는 전체 JSON 상태를 읽음
        data = read_data()
        # i는 교체할 목록 위치, entry는 ID를 확인할 개별 항목
        for i, entry in enumerate(data["entries"]):
            # BEGIN A04-01
            # TODO A04-01: 현재 항목이 수정 대상인지 비교 결과를 same_id에 넣으세요.
            # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
            # 개별 항목의 ID와 요청 ID를 비교하여 True·False 생성
            same_id = entry["id"] == knowledge_id
            # END A04-01
            # ID가 같은 대상에 대해서만 반환 또는 수정 수행
            if same_id:
                # BEGIN A04-02
                # TODO A04-02: 새 입력을 dict로 만들고 기존 knowledge_id를 유지하여 updated를 구성하세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # 검증된 PUT 입력 세 필드를 새 dict로 변환
                updated = payload.model_dump()
                # 새 ID 발급 없이 수정 대상의 기존 ID 유지
                updated["id"] = knowledge_id
                # END A04-02
                # BEGIN A04-03
                # TODO A04-03: 대상 위치 i를 updated로 교체하세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # 전체 데이터 안의 목록에서 i번째 항목만 교체. append로 중복 생성하지 않음
                data["entries"][i] = updated
                # END A04-03
                # BEGIN A04-04
                # TODO A04-04: 변경 내용을 저장하세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # 같은 잠금 안에서 변경된 전체 data를 저장. 위치 인자 write_data(data)와 동일
                write_data(data=data)
                # END A04-04
                # 파일 저장 성공 뒤 교체한 항목을 반환. PUT의 기본 성공 상태는 200
                return updated
    # 반복문을 끝까지 돌아도 대상을 못 찾으면 404. 데이터 생성·저장은 하지 않음
    raise HTTPException(404, "Knowledge not found")
