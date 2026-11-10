"""
[Day17 - A07 기본 개념: 지식 CRUD API 통합]

1. FastAPI와 Route
   - @app.get/post/put/delete는 HTTP 메서드와 URL을 처리 함수에 연결함
   - 실행: python -m uvicorn main:app --reload
   - Uvicorn이 main.py의 app을 실행하며 --reload는 코드 변경을 감지함

2. Pydantic 모델
   - KnowledgeWrite는 요청 Body의 필수 필드·자료형·문자열 길이를 검증함
   - 검증 실패 시 422이며 Route 본문의 저장 단계는 실행하지 않음
   - Knowledge는 입력 필드에 id를 더한 응답 구조임
   - 형식 검증은 지식 내용의 사실성을 확인하는 작업과 다름

3. data, entries, entry, next_id
   - data는 JSON 전체 상태, entries는 지식 목록, entry는 지식 한 개의 dict임
   - next_id는 다음 등록에 사용할 ID이며 삭제해도 감소시키지 않음
   - 리스트 위치 i와 지식의 고유 ID는 서로 다른 값임

4. CRUD와 상태 코드
   - 조회 200, 등록 201, 수정 200, 삭제 204이며 없는 ID는 404임
   - PUT은 세 입력 필드를 모두 받아 기존 ID를 유지하면서 항목을 교체함
   - 204는 본문 없이 Response(status_code=204)로 반환함

5. 응용 실습의 조회 분리
   - ?category=api처럼 전달하는 선택 쿼리 인자로 목록을 필터링함
   - queries.py는 정확히 일치하는 category를 고르며 생략하면 전체를 반환함
   - 목록에 일치 항목이 없으면 정상 200과 [], 상세 ID가 없으면 404임
   - 조회 계산은 queries.py, HTTP 응답은 main.py, 파일 입출력은 storage.py가 담당함

6. 저장과 잠금
   - dict를 변경한 뒤 write_data(data)를 호출해야 실제 파일에 반영됨
   - 같은 프로세스의 LOCK으로 읽기·ID 발급·변경·저장을 한 구간으로 보호함
   - 여러 worker 사이의 잠금이나 DB 트랜잭션을 제공하는 구조는 아님

7. 실습 확인
   - 등록 응답 ID로 조회 → 수정 → 삭제하고 상태 코드와 JSON 파일을 함께 확인함
   - category 생략·일치·불일치, 잘못된 Body의 422, 재시작 후 저장 유지도 확인함
   - 이 API는 지식을 관리하며 AI 답변을 생성하지 않음
"""
# HTTP 경로 등록, 오류 응답, 본문 없는 응답을 위한 도구
from fastapi import FastAPI, HTTPException, Response
# 같은 프로세스에서 공유하는 잠금과 JSON 입출력 함수
from storage import LOCK, read_data, write_data
# HTTP나 파일을 다루지 않고 조회 결과만 계산하는 함수
from queries import select_category, find_entry

from pydantic import BaseModel, Field

# 입력 모델은 HTTP Body의 규칙입니다. DB 테이블이나 저장 코드가 아닙니다.
class KnowledgeWrite(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)


# 입력 필드를 상속하고 서버가 발급한 정수 ID를 추가
class Knowledge(KnowledgeWrite):
    id: int


# /docs의 자동 문서에도 표시할 제목과 설명
app = FastAPI(title="Knowledge API", description="AI 플랫폼의 원본 지식 관리 연습")

@app.get("/health")
def health():
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    return {"status": "ok"}

@app.get("/hello")
def hello(name: str = "수강생"):
    # ?name=민준을 보내면 해당 이름을, 생략하면 기본값을 사용
    return {"message": f"안녕하세요, {name}"}

@app.get("/api/knowledge")
def list_knowledge(category: str | None = None):
    with LOCK:
        data = read_data()
    # BEGIN A07-01
    # TODO A07-01: 읽은 entries와 category를 select_category에 넘겨 결과를 응답하세요.
    # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
    # 내부용 next_id를 제외한 목록을 전달하고 필터 결과만 반환
    return select_category(data["entries"], category)
    # END A07-01

@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    with LOCK:
        data = read_data()
    # 경로 ID와 일치하는 dict 또는 None을 조회 함수에서 받음
    entry = find_entry(data["entries"], knowledge_id)
    if entry is None:
        raise HTTPException(404, "Knowledge not found")
    return entry

@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    # 하나의 잠금으로 현재 데이터 읽기 → ID 발급 → 저장을 함께 보호합니다.
    with LOCK:
        data = read_data()
        # 검증된 Pydantic 객체를 수정 가능한 dict로 변환
        entry = payload.model_dump()
        # 서버가 관리하는 다음 ID를 발급
        entry["id"] = data["next_id"]
        # BEGIN A07-02
        # TODO A07-02: 생성할 entry를 한 번 추가하고 next_id를 한 번 증가시키세요.
        # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
        # 새 항목을 목록 끝에 한 번 추가
        data["entries"].append(entry)
        # 다음 생성 요청에서 사용할 번호를 준비
        data["next_id"] += 1
        # END A07-02
        write_data(data)
    # 201을 선언하는 것만으로 저장되지는 않습니다. 실제 저장 성공 뒤 응답합니다.
    return entry

@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    with LOCK:
        data = read_data()
        # i는 리스트 위치이며 entry는 해당 위치의 지식 dict
        for i, entry in enumerate(data["entries"]):
            same_id = entry["id"] == knowledge_id
            if same_id:
                updated = payload.model_dump()
                updated["id"] = knowledge_id
                # BEGIN A07-03
                # TODO A07-03: 수정 대상 위치를 updated로 교체하세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # i 변수 자체가 아니라 목록의 i번째 항목을 교체
                data["entries"][i] = updated
                # END A07-03
                write_data(data)
                return updated
    # 없는 ID를 새로 생성하지 않습니다. PUT에서도 세 입력 필드가 모두 필수입니다.
    raise HTTPException(404, "Knowledge not found")

@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                # 일치하는 항목 하나만 삭제하며 다른 항목과 next_id는 유지
                data["entries"].pop(i)
                write_data(data)
                # BEGIN A07-04
                # TODO A07-04: 삭제 성공에서 빈 204 응답을 반환하세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # 첫 위치 인자는 content이므로 상태 코드는 키워드로 지정
                return Response(status_code=204)
                # END A07-04
    raise HTTPException(404, "Knowledge not found")
