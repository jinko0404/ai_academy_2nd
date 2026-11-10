"""함께 통합하고 프로젝트 준비. 실행: python -m uvicorn main:app --reload"""
"""
[기본필수 실습 07 - B07 기본 개념: 지식 관리 CRUD 통합]

1. 프로젝트 목적
   - 제목·본문·분류를 가진 지식을 JSON 파일에 생성·조회·수정·삭제
   - AI 질문에 답을 생성하는 서비스가 아니라 원본 지식을 관리하는 API
   - main.py는 HTTP 처리, storage.py는 파일 접근, knowledge.json은 실제 데이터

2. 라우트와 실행
   - @app.get/post/put/delete는 메서드·URL과 바로 아래 함수를 연결
   - 이 폴더에서 python -m uvicorn main:app --reload로 서버 실행
   - main:app은 main.py의 app 객체, reload는 개발 중 변경 감지 옵션
   - /docs는 자동 API 문서이며 /health는 해당 경로의 응답 확인

3. 요청 검증과 응답 모델
   - KnowledgeWrite는 Body의 필수 title·content·category와 문자열 길이 검증
   - Knowledge는 입력 필드를 상속하고 서버가 부여하는 정수 id를 추가
   - 잘못된 입력은 422이며 response_model은 응답 구조를 검증
   - 검증 모델과 응답 상태코드를 정의하는 것만으로 파일이 저장되지는 않음

4. data·entry·i의 차이
   - data는 entries 목록과 next_id를 가진 파일 전체 dict
   - entry는 개별 항목, i는 enumerate로 얻은 목록 위치
   - 항목 id는 식별자이고 i는 위치이므로 서로 같은 값이라고 가정하면 안 됨

5. B07-01: 목록 조회
   - data['entries']만 반환하여 내부 ID 발급 상태 next_id는 응답에서 제외
   - dict·list 반환값은 FastAPI가 JSON 응답으로 변환
   - 상세 조회는 ID가 없으면 404, 정수가 아닌 경로는 422

6. B07-02: 생성과 영속화
   - model_dump()로 검증된 Body를 dict로 변환
   - next_id로 새 ID 발급 후 목록에 추가하고 next_id를 1 증가
   - 두 변경이 담긴 전체 data를 write_data로 저장한 후 201 응답
   - 응답만 성공한 것처럼 보지 말고 GET 재조회와 파일 상태도 확인

7. B07-03: 수정
   - 새 입력에 기존 ID를 붙이고 data['entries'][i]를 교체
   - PUT에도 세 입력 필드가 모두 필요하며 부분 수정은 구현하지 않음
   - 목록 수·ID·next_id는 유지하고 없는 ID를 새로 생성하지 않음

8. B07-04: 삭제와 204
   - pop(i)는 목록의 항목 하나를 제거하고 반환하지만 여기서는 반환값을 사용하지 않음
   - 다른 항목과 next_id는 유지하고 변경 결과를 파일에 저장
   - return Response(status_code=204)는 본문 없는 성공 응답
   - Response(204)는 첫 인자에 본문을 넣는 호출이므로 같은 의미가 아님
   - Response는 예외가 아니므로 raise가 아니라 return 사용

9. 잠금과 실패 보존
   - LOCK으로 읽기부터 ID 발급·수정·저장까지 같은 프로세스의 동시 접근 보호
   - 다중 프로세스 잠금이나 DB 트랜잭션은 아님
   - 없는 ID의 404와 잘못된 Body의 422에서 파일이 바뀌지 않는지 확인 필요

10. 재시작 확인
    - 파일로 저장한 데이터는 같은 폴더의 서버를 재시작한 뒤에도 조회되어야 함
    - /docs 새로고침은 서버 프로세스 재시작이 아님
    - 지속성 확인 중 reset_data.py를 실행하면 초기화되므로 실행하지 않음
    - 인증·DB·LLM 호출은 이 실습에 포함되지 않음
"""

# 웹 앱, HTTP 오류와 본문 없는 응답 생성 도구
from fastapi import FastAPI, HTTPException, Response
# 현재 교시의 JSON 파일 읽기·쓰기와 공유 잠금
from storage import LOCK, read_data, write_data

# 요청·응답 모델과 필드 길이 검증 도구
from pydantic import BaseModel, Field

# 입력 모델은 HTTP Body의 규칙입니다. DB 테이블이나 저장 코드가 아닙니다.
class KnowledgeWrite(BaseModel):
    # 생성·수정 시 필요한 필수 문자열과 길이 제약
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)


# 입력 구조를 상속하고 응답에 ID 추가
class Knowledge(KnowledgeWrite):
    id: int


# 앱 객체 생성. title·description은 /docs에도 표시
app = FastAPI(title="Knowledge API", description="AI 플랫폼의 원본 지식 관리 연습")

# 상태 확인 요청을 health 함수에 연결
@app.get("/health")
def health():
    # 이 경로의 응답 확인입니다. 파일·DB·AI 모델의 전체 상태 점검은 아닙니다.
    return {"status": "ok"}

# 전체 지식 목록 조회 경로 등록
@app.get("/api/knowledge")
def list_knowledge():
    # 파일 접근 중 공유 잠금을 사용
    with LOCK:
        data = read_data()
    # 파일 전체에는 내부용 next_id도 있으므로 entries만 응답합니다.
    # BEGIN B07-01
    # TODO B07-01: 통합 목록 Route에서 entries만 응답하도록 연결하세요.
    # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
    # 사용자 지식 배열만 반환. 목록이 비면 []가 됨
    return data["entries"]
    # END B07-01

# URL의 정수 ID로 지식 하나 조회
@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        # 현재 항목 ID와 경로 ID 비교
        same_id = entry["id"] == knowledge_id
        if same_id:
            return entry
    # 입력은 정수이지만 저장된 항목이 없으면 404입니다. foo 같은 Path는 먼저 422입니다.
    raise HTTPException(404, "Knowledge not found")

# 생성 경로 등록. 정상 생성은 201, 반환 구조는 Knowledge
@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    # 하나의 잠금으로 현재 데이터 읽기 → ID 발급 → 저장을 함께 보호합니다.
    with LOCK:
        data = read_data()
        # 검증된 Body를 수정 가능한 dict로 변환
        entry = payload.model_dump()
        # 클라이언트 입력 대신 서버의 next_id에서 식별자 부여
        entry["id"] = data["next_id"]
        # 새 항목은 목록 끝에 추가
        data["entries"].append(entry)
        # 다음 생성에 사용할 번호 증가
        data["next_id"] += 1
        # BEGIN B07-02
        # TODO B07-02: 생성된 지식과 증가한 next_id를 파일에 함께 저장하세요.
        # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
        # 새 항목과 next_id가 함께 들어 있는 전체 상태 저장
        write_data(data=data)
        # END B07-02
    # 201을 선언하는 것만으로 저장되지는 않습니다. 실제 저장 성공 뒤 응답합니다.
    return entry

# 기존 항목의 입력 필드를 전체 교체하는 PUT 등록
@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    # 탐색·교체·저장까지 한 잠금 구간으로 처리
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            same_id = entry["id"] == knowledge_id
            if same_id:
                # 새 입력으로 dict를 만들고 대상의 기존 ID 유지
                updated = payload.model_dump()
                updated["id"] = knowledge_id
                # BEGIN B07-03
                # TODO B07-03: 같은 ID의 위치 i를 updated로 교체하세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # 새 항목을 추가하지 않고 기존 위치 한 개를 교체
                data["entries"][i] = updated
                # END B07-03
                # 메모리에서 교체한 결과를 파일에 반영
                write_data(data)
                return updated
    # 없는 ID를 새로 생성하지 않습니다. PUT에서도 세 입력 필드가 모두 필수입니다.
    raise HTTPException(404, "Knowledge not found")

# 삭제 경로 등록. 성공 응답에 본문을 넣지 않음
@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    # 읽기→대상 확인→삭제→저장 순서 전체를 잠금으로 보호
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                # 해당 위치의 항목 한 개만 제거. next_id는 감소시키지 않음
                data["entries"].pop(i)
                # 삭제한 전체 상태를 실제 파일에 저장
                write_data(data)
                # BEGIN B07-04
                # TODO B07-04: 삭제 저장 후 본문 없는 204 응답을 반환하세요. dict 또는 JSON 문자열을 넣지 마세요.
                # 아래 미완성 표시를 지우고 요구한 코드를 작성하세요.
                # status_code를 이름으로 지정하여 본문 없는 204 반환 후 함수 종료
                return Response(status_code= 204)
                # END B07-04
    # 삭제 대상을 못 찾았으면 파일을 변경하지 않고 404 응답
    raise HTTPException(404, "Knowledge not found")
