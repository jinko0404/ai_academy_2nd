"""삭제와 실패 후 데이터 보존. 실행: python -m uvicorn main:app --reload"""
"""
[응용 실습 05 - A05 기본 개념: 삭제·실패 보존과 ID 발급]

1. HTTP와 CRUD
   - GET은 조회, POST는 생성, PUT은 기존 항목 교체, DELETE는 삭제
   - FastAPI 데코레이터로 메서드·경로와 함수를 연결
   - 이 폴더에서 python -m uvicorn main:app --reload로 서버 실행

2. 데이터 구조와 인덱스
   - data는 파일 전체 dict이며 entries 목록과 next_id를 포함
   - entry는 지식 한 개, i는 목록의 0부터 시작하는 위치
   - 항목 ID와 목록 위치는 다르므로 ID를 비교한 뒤 실제 위치 i를 사용

3. A05-01: 삭제와 저장
   - entry['id'] == knowledge_id인 항목만 삭제
   - del data['entries'][i]는 메모리 목록의 항목 하나를 제거하며 파일 자체를 삭제하지 않음
   - write_data로 전체 상태를 저장해야 파일에 반영됨
   - 저장 후 즉시 return하므로 목록 삭제 뒤 순회를 계속하지 않음
   - 다른 항목과 next_id는 유지

4. A05-02: 없는 대상과 오류
   - 탐색을 끝까지 해도 항목이 없으면 404와 Knowledge not found 응답
   - 없는 ID 처리에서는 삭제·저장을 호출하지 않음
   - 정수가 아닌 경로 ID는 FastAPI 검증에서 422
   - 성공은 본문 없는 204이므로 Postman의 빈 Body는 정상

5. A05-03: 삭제 뒤 새 ID
   - 생성할 때 목록 길이가 아니라 저장된 next_id에서 ID를 가져옴
   - 예: ID 1·2·3 중 2를 지워도 다음 생성 ID는 기존 next_id인 4를 사용
   - 생성 이후 next_id를 1 증가시키고 함께 저장
   - 이 규칙은 next_id가 올바르게 유지된 상태를 전제로 하며 삭제 ID 재사용을 피함

6. 입력 검증과 응답
   - KnowledgeWrite는 title·content·category의 필수 문자열과 길이를 검증
   - Knowledge는 같은 구조에 정수 id를 추가한 응답 모델
   - model_dump()는 검증된 입력을 dict로 변환
   - 입력 검증과 실제 파일 저장은 별개의 단계

7. 잠금과 데이터 보존
   - 읽기→변경→저장을 같은 LOCK으로 묶어 같은 프로세스의 동시 접근 보호
   - 여러 프로세스의 파일 잠금이나 DB 트랜잭션은 아님
   - 잘못된 Body의 422, 없는 항목의 404에서 파일이 바뀌지 않는지 확인 필요

8. 조회·수정과 실습 범위
   - category 필터는 정확한 문자열 일치이며 결과가 없으면 200과 빈 목록
   - PUT은 기존 ID와 항목 수를 유지하고 해당 위치를 교체
   - /health는 경로 응답 확인이며 실제 DB·AI 준비 상태 검사가 아님
   - 저장은 교시별 JSON 파일을 사용하며 인증·DB·LLM 호출은 포함하지 않음
"""

# 앱, HTTP 오류, 본문 없는 응답 도구
from fastapi import FastAPI, HTTPException, Response
# 교시별 JSON 읽기·쓰기와 공유 잠금
from storage import LOCK, read_data, write_data

# 요청·응답 모델과 필드 길이 제약 정의
from pydantic import BaseModel, Field

# 입력 모델은 HTTP Body의 규칙입니다. DB 테이블이나 저장 코드가 아닙니다.
# POST·PUT에서 공통으로 사용하는 필수 입력 구조
class KnowledgeWrite(BaseModel):
    # 문자열 길이 제약. 위반하면 처리 함수 실행 전에 422 응답
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)


# 입력 필드를 상속하고 응답에 서버 관리 ID 추가
class Knowledge(KnowledgeWrite):
    id: int


# API 앱 생성. 제목과 설명은 자동 문서 /docs에도 표시
app = FastAPI(title="Knowledge API", description="AI 플랫폼의 원본 지식 관리 연습")

# 상태 확인 GET 경로
@app.get("/health")
def health():
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    return {"status": "ok"}

# name 쿼리로 인사말을 구성하는 GET 경로
@app.get("/hello")
def hello(name: str = "수강생"):
    return {"message": f"안녕하세요, {name}"}

# 전체 목록 또는 category 필터 결과 조회
@app.get("/api/knowledge")
def list_knowledge(category: str | None = None):
    with LOCK:
        data = read_data()
    # 내부용 next_id를 제외하고 항목 목록만 선택
    result = data["entries"]
    if category is not None:
        # 전달된 category와 정확히 같은 항목만 포함
        result = [entry for entry in result if entry["category"] == category]
    # 조건에 맞는 지식이 없으면 정상 조회 200과 빈 배열 []입니다.
    return result

# 경로의 정수 ID로 항목 하나 조회
@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        same_id = entry["id"] == knowledge_id
        if same_id:
            return entry
    # 입력은 정수이지만 저장된 항목이 없으면 404입니다. foo 같은 Path는 먼저 422입니다.
    raise HTTPException(404, "Knowledge not found")

# 생성 성공 201과 Knowledge 응답 구조 지정
@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    # 하나의 잠금으로 현재 데이터 읽기 → ID 발급 → 저장을 함께 보호합니다.
    with LOCK:
        data = read_data()
        # 검증된 생성 Body를 수정 가능한 dict로 변환
        entry = payload.model_dump()
        # BEGIN A05-03
        # TODO A05-03: 삭제 뒤 다시 생성해도 중복되지 않도록 next_id에서 ID를 발급하세요.
        # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
        # 삭제로 목록 길이가 줄어도 기존 next_id를 사용하여 새 ID 발급
        entry["id"] = data["next_id"]
        # END A05-03
        # 새 항목을 추가하고 다음 ID를 증가시킨 후 두 변경 사항을 함께 저장
        data["entries"].append(entry)
        data["next_id"] += 1
        write_data(data)
    # 201을 선언하는 것만으로 저장되지는 않습니다. 실제 저장 성공 뒤 응답합니다.
    return entry

# 기존 지식의 입력 필드를 전체 교체하는 PUT 경로
@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            same_id = entry["id"] == knowledge_id
            if same_id:
                # 새 입력을 dict로 변환하고 기존 ID 유지
                updated = payload.model_dump()
                updated["id"] = knowledge_id
                # 대상 위치 교체 후 파일 저장. append나 next_id 증가를 하지 않음
                data["entries"][i] = updated
                write_data(data)
                return updated
    # 없는 ID를 새로 생성하지 않습니다. PUT에서도 세 입력 필드가 모두 필수입니다.
    raise HTTPException(404, "Knowledge not found")

# 삭제 성공 시 본문 없이 204를 반환할 경로
@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    # 읽기·대상 탐색·제거·저장 전체를 동일 잠금으로 보호
    with LOCK:
        # 파일 전체 상태 조회
        data = read_data()
        # i는 삭제 위치, entry는 현재 지식 하나
        for i, entry in enumerate(data["entries"]):
            # BEGIN A05-01
            # TODO A05-01: 현재 entry가 요청 ID와 같을 때만 위치 i의 항목을 제거·저장하고 본문 없는 204를 반환하세요.
            # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
            # 요청 ID와 같은 항목만 처리
            if entry["id"] == knowledge_id:
                # 해당 항목 하나만 메모리 목록에서 제거
                del data["entries"][i]
                # 변경된 전체 상태를 실제 파일로 저장
                write_data(data=data)
                # 저장 성공 후 즉시 함수 종료. 응답 본문은 없음
                return Response(status_code=204)
            # END A05-01
    # BEGIN A05-02
    # TODO A05-02: 대상을 찾지 못하면 404를 raise하세요. next_id나 파일을 변경하지 마세요.
    # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
    # 대상을 찾지 못하면 파일을 변경하지 않고 404 반환
    raise HTTPException(404, "Knowledge not found")
    # END A05-02
