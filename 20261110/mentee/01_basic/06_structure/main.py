"""
[01_basic 실습 06 / main.py 기본 개념]

1. 역할 분리
   - main.py는 HTTP 경로 등록, 요청 검증 모델, 오류·응답 처리를 담당
   - storage.py는 JSON 파일 읽기·쓰기를 담당하며 데이터 파일 자체와 구분
   - 응용은 queries.py로 목록 필터·ID 탐색 계산을 분리하고 기본필수는 내부에서 처리
   - 함수 분리는 새 기능 추가가 아니라 책임을 나누면서 기존 API 동작을 유지하는 작업

2. 입력 모델과 경로
   - KnowledgeWrite는 title·content·category의 필수 문자열·길이 제약을 검증
   - Knowledge는 응답용 정수 id를 추가하며 저장 함수나 DB 테이블이 아님
   - GET 조회, POST 생성, PUT 전체 필드 교체, DELETE 삭제를 데코레이터로 등록
   - 경로 knowledge_id는 정수, 응용의 name·category는 쿼리 인자

3. 데이터와 ID
   - data는 entries와 next_id를 담은 전체 dict, entry는 항목 하나, i는 목록 위치
   - POST는 next_id로 ID 발급 후 append·next_id 증가·파일 저장
   - PUT은 기존 ID와 항목 수를 유지하며 목록 위치를 교체
   - DELETE는 항목 하나를 제거하며 next_id와 다른 항목은 유지

4. 저장과 잠금
   - read_data로 읽은 메모리 값만 바꿔서는 영속화되지 않으므로 write_data 호출 필요
   - 잠금 안에서 읽기·ID 발급·수정·저장을 묶어 같은 프로세스의 동시 접근 보호
   - threading.Lock은 프로세스 간 잠금이나 DB 트랜잭션이 아님

5. 조회 계산과 HTTP 오류
   - 필터 결과가 없으면 200과 빈 목록, 상세 ID가 없으면 404
   - 응용의 find_entry는 None만 반환하고 main.py가 404로 변환
   - 잘못된 정수 경로나 누락·길이 위반 입력은 요청 검증 단계에서 422
   - DELETE 성공은 본문 없는 204이며 POST는 201, 정상 조회·수정은 기본 200

6. 실행과 범위
   - 이 교시 폴더에서 python -m uvicorn main:app --reload 실행
   - /docs에서 요청 구조를 확인하고 실제 응답과 파일 변화를 함께 검증
   - /health는 경로 응답 확인이며 AI·DB 전체 상태 확인이 아님
   - 인증·DB·LLM 호출은 하지 않으며 각 교시는 독립된 JSON 파일 사용
"""

"""파일 역할 이해·분리. 실행: python -m uvicorn main:app --reload"""
# 앱, HTTP 오류, 본문 없는 응답 도구
from fastapi import FastAPI, HTTPException, Response
# 교시별 JSON 읽기·쓰기와 공유 잠금
from storage import LOCK, read_data, write_data

# 요청·응답 모델과 필드 길이 제약 정의
from pydantic import BaseModel, Field

# 입력 모델은 HTTP Body의 규칙입니다. DB 테이블이나 저장 코드가 아닙니다.
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
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    return {"status": "ok"}

# 전체 목록 또는 category 필터 결과 조회
@app.get("/api/knowledge")
def list_knowledge():
    # 읽기·대상 탐색·제거·저장 전체를 동일 잠금으로 보호
    with LOCK:
        # 파일 전체 상태 조회
        data = read_data()
    # 파일 전체에는 내부용 next_id도 있으므로 entries만 응답합니다.
    # 파일 전체에는 내부용 next_id도 있으므로 entries만 응답합니다.
    return data["entries"]

# 경로의 정수 ID로 항목 하나 조회
@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    # 읽기·대상 탐색·제거·저장 전체를 동일 잠금으로 보호
    with LOCK:
        # 파일 전체 상태 조회
        data = read_data()
    for entry in data["entries"]:
        # 현재 항목의 ID와 요청 ID를 비교하며 목록 위치와 혼동하지 않음
        same_id = entry["id"] == knowledge_id
        if same_id:
            # 201을 선언하는 것만으로 저장되지는 않습니다. 실제 저장 성공 뒤 응답합니다.
            return entry
    # 입력은 정수이지만 저장된 항목이 없으면 404입니다. foo 같은 Path는 먼저 422입니다.
    # 대상을 찾지 못하면 파일을 변경하지 않고 404 반환
    raise HTTPException(404, "Knowledge not found")

# 생성 성공 201과 Knowledge 응답 구조 지정
@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    # 하나의 잠금으로 현재 데이터 읽기 → ID 발급 → 저장을 함께 보호합니다.
    # 읽기·대상 탐색·제거·저장 전체를 동일 잠금으로 보호
    with LOCK:
        # 파일 전체 상태 조회
        data = read_data()
        # 검증된 생성 Body를 수정 가능한 dict로 변환
        entry = payload.model_dump()
        # 삭제로 목록 길이가 줄어도 기존 next_id를 사용하여 새 ID 발급
        entry["id"] = data["next_id"]
        # 새 항목을 추가하고 다음 ID를 증가시킨 후 두 변경 사항을 함께 저장
        data["entries"].append(entry)
        data["next_id"] += 1
        write_data(data)
    # 201을 선언하는 것만으로 저장되지는 않습니다. 실제 저장 성공 뒤 응답합니다.
    # 201을 선언하는 것만으로 저장되지는 않습니다. 실제 저장 성공 뒤 응답합니다.
    return entry

# 기존 지식의 입력 필드를 전체 교체하는 PUT 경로
@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    # 읽기·대상 탐색·제거·저장 전체를 동일 잠금으로 보호
    with LOCK:
        # 파일 전체 상태 조회
        data = read_data()
        # i는 삭제 위치, entry는 현재 지식 하나
        for i, entry in enumerate(data["entries"]):
            # 현재 항목의 ID와 요청 ID를 비교하며 목록 위치와 혼동하지 않음
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
    # 대상을 찾지 못하면 파일을 변경하지 않고 404 반환
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
            # 요청 ID와 같은 항목만 처리
            if entry["id"] == knowledge_id:
                # 찾은 위치의 항목 하나 제거. pop의 반환 항목은 여기서는 사용하지 않음
                data["entries"].pop(i)
                write_data(data)
                # 저장 성공 후 즉시 함수 종료. 응답 본문은 없음
                return Response(status_code=204)
    # 대상을 찾지 못하면 파일을 변경하지 않고 404 반환
    raise HTTPException(404, "Knowledge not found")
