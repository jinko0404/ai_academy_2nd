"""삭제와 실패 후 데이터 보존. 실행: python -m uvicorn main:app --reload"""
"""
[실습 05 - B05 기본 개념: 지식 삭제와 실패 시 데이터 보존]

1. CRUD와 HTTP 메서드
   - CRUD는 생성(Create), 조회(Read), 수정(Update), 삭제(Delete)를 의미
   - 이 파일은 POST 생성, GET 조회, PUT 전체 입력 필드 교체, DELETE 삭제를 제공
   - 데코레이터는 요청 메서드·경로와 처리 함수를 연결
   - 이 폴더에서 python -m uvicorn main:app --reload로 서버 실행

2. data, entries, entry, i
   - data: JSON 파일 전체를 읽은 dict로 entries 목록과 next_id를 포함
   - data['entries']: 저장된 지식 목록, entry: 목록 안의 항목 하나
   - enumerate(): 목록 위치 i와 해당 항목 entry를 함께 반환
   - i는 0부터 시작하는 위치이며 항목의 id와 같은 값이라고 가정하면 안 됨

3. B05-01: del로 항목 제거
   - del data['entries'][i]는 목록의 i번째 항목 한 개를 제거하는 Python 문법
   - 파일 자체를 삭제하거나 전체 목록을 비우는 작업이 아님
   - 삭제 이후 뒤의 항목 위치는 당겨지지만 각 항목의 id는 변경하지 않음
   - 목록 순회 중 삭제 후 곧바로 return하므로 다음 항목으로 순회를 이어가지 않음

4. ID와 next_id 보존
   - knowledge_id는 삭제할 항목을 식별하는 정수 경로 인자
   - entry['id']와 비교하여 실제 목록 위치를 찾음
   - 삭제해도 next_id를 줄이지 않으므로 지운 ID를 다음 생성에서 재사용하지 않음
   - 다른 지식의 내용과 ID도 그대로 유지

5. B05-02: 파일 저장
   - del은 메모리의 목록만 바꾸므로 write_data(data)로 파일에도 반영해야 함
   - write_data(data=data)는 같은 함수에 키워드 인자로 전달하는 동일한 호출
   - 저장에 실패하면 성공 204를 반환하는 줄까지 진행되지 않음
   - 실제 파일 처리 방식은 제공된 storage.py에서 담당

6. 잠금과 동시 접근
   - with LOCK으로 읽기·탐색·삭제·저장을 한 구간에서 보호
   - 같은 프로세스의 스레드 간 잠금이며 다중 프로세스 잠금이나 DB 트랜잭션은 아님
   - 서버를 여러 worker로 운용할 경우 별도의 저장·동시성 설계가 필요

7. 204, 404, 422 응답
   - 삭제 성공은 Response(status_code=204)로 본문 없는 응답을 반환
   - Postman의 Body가 비어 있어도 상태가 204이면 정상 삭제 성공
   - 없는 정수 ID는 404이며 파일 저장을 호출하지 않음
   - 정수가 아닌 경로는 FastAPI 입력 검증 단계에서 422
   - 같은 항목을 다시 DELETE하면 이 구현에서는 404지만 삭제된 상태는 유지

8. Pydantic과 등록·수정
   - KnowledgeWrite는 title·content·category의 필수 문자열과 길이를 검증
   - Knowledge는 입력 구조에 응답용 정수 id를 추가
   - POST는 새 항목 추가와 next_id 증가, PUT은 기존 위치 교체와 ID 유지
   - DELETE는 이 함수에서 Body를 받지 않고 경로 ID로 대상을 선택

9. 확인 방법과 실습 범위
   - 삭제 전후 GET 목록에서 항목 수·대상 ID를 비교하고 JSON 파일도 확인
   - 없는 ID 삭제 후 파일이 바뀌지 않는지도 확인해야 함
   - /health는 경로 응답 확인일 뿐 DB·AI 모델의 전체 상태 검사가 아님
   - 이 파일에는 인증·실제 DB·LLM 호출이 포함되지 않음
"""

# 웹 앱 생성, HTTP 오류, 본문 없는 204 응답 도구
from fastapi import FastAPI, HTTPException, Response
# 현재 교시의 공유 잠금과 JSON 파일 읽기·쓰기 함수
from storage import LOCK, read_data, write_data

# 요청·응답 모델과 필드 길이 검증 도구
from pydantic import BaseModel, Field

# 입력 모델은 HTTP Body의 규칙입니다. DB 테이블이나 저장 코드가 아닙니다.
class KnowledgeWrite(BaseModel):
    # 생성·수정 시 필요한 제목, 본문, 분류의 최소·최대 문자열 길이
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)


# 입력 구조를 상속하고 응답에 서버 관리 ID를 추가
class Knowledge(KnowledgeWrite):
    id: int


# 앱 객체 생성. title·description은 자동 문서 /docs에도 표시
app = FastAPI(title="Knowledge API", description="AI 플랫폼의 원본 지식 관리 연습")

# 상태 확인 GET 경로 등록
@app.get("/health")
def health():
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    return {"status": "ok"}

# 전체 지식 목록 조회 경로 등록
@app.get("/api/knowledge")
def list_knowledge():
    # 잠금 안에서 현재 파일 상태를 읽고 나서 목록 반환
    with LOCK:
        data = read_data()
    # 파일 전체에는 내부용 next_id도 있으므로 entries만 응답합니다.
    return data["entries"]

# 경로의 정수 ID로 지식 하나를 조회
@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        # 현재 항목의 ID와 요청 ID를 비교하며 목록 위치와 혼동하지 않음
        same_id = entry["id"] == knowledge_id
        if same_id:
            return entry
    # 입력은 정수이지만 저장된 항목이 없으면 404입니다. foo 같은 Path는 먼저 422입니다.
    raise HTTPException(404, "Knowledge not found")

# 새 지식 생성 경로. 성공 시 201과 Knowledge 형태의 응답
@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    # 하나의 잠금으로 현재 데이터 읽기 → ID 발급 → 저장을 함께 보호합니다.
    with LOCK:
        data = read_data()
        # 검증된 입력을 dict로 변환하고 서버의 next_id로 새 ID 부여
        entry = payload.model_dump()
        entry["id"] = data["next_id"]
        # 생성에서는 목록에 추가하고 다음 발급 ID를 증가시킨 뒤 파일 저장
        data["entries"].append(entry)
        data["next_id"] += 1
        write_data(data)
    # 201을 선언하는 것만으로 저장되지는 않습니다. 실제 저장 성공 뒤 응답합니다.
    return entry

# 기존 지식을 입력 필드 전체로 교체하는 경로
@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    # 수정의 읽기·탐색·교체·저장을 같은 잠금으로 보호
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            same_id = entry["id"] == knowledge_id
            if same_id:
                # 새 입력을 dict로 만들고 기존 ID를 유지
                updated = payload.model_dump()
                updated["id"] = knowledge_id
                # 대상 위치를 교체하고 전체 상태를 저장. 새 항목을 추가하는 것이 아님
                data["entries"][i] = updated
                write_data(data)
                return updated
    # 없는 ID를 새로 생성하지 않습니다. PUT에서도 세 입력 필드가 모두 필수입니다.
    raise HTTPException(404, "Knowledge not found")

# 삭제 경로 등록. 성공하면 JSON 본문 없이 204 응답
@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    # 읽기부터 삭제 결과 저장까지 잠금을 유지
    with LOCK:
        # 파일 전체 상태를 메모리로 읽음
        data = read_data()
        # i는 삭제할 위치, entry는 대상 ID를 확인할 항목
        for i, entry in enumerate(data["entries"]):
            # 요청한 ID와 같은 항목에 대해서만 삭제 수행
            if entry["id"] == knowledge_id:
                # BEGIN B05-01
                # TODO B05-01: 찾은 위치 i의 지식 한 개만 entries에서 제거하세요. 파일 전체를 삭제하지 마세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # 목록의 해당 위치 한 개를 메모리에서 제거. 파일 전체와 next_id는 유지
                del data["entries"][i]
                # END B05-01
                # BEGIN B05-02
                # TODO B05-02: 제거한 결과를 파일에 저장하세요. 다른 항목과 next_id는 유지하세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # 수정된 전체 상태를 파일에 저장. 키워드 호출이며 write_data(data)와 동일
                write_data(data=data)
                # END B05-02
                # 저장 성공 뒤 본문 없는 응답 반환. 즉시 끝나므로 삭제 후 순회를 계속하지 않음
                return Response(status_code=204)
    # 끝까지 대상이 없으면 저장하지 않고 404 반환
    raise HTTPException(404, "Knowledge not found")
