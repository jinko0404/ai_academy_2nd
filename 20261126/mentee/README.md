# Day29 · 2026-11-26 · AI Agent Fundamentals & Tool Calling (수강생용)

## 실제 11/20 DB 재사용

기존 **`report_agent_lab`**을 그대로 조회합니다. 새 DB/테이블을 만들거나 데이터 리셋·재시드를 수행하지 않습니다.

**실제 DB 스키마 계약:** `reports(report_id,category_id,overview,index_status,revision)`, `report_embeddings(report_id,embedding vector(384),model_name,overview_hash,indexed_revision)`, `report_read_view(report_id,category_code,title,content,overview,...)`. 6종 Category는 30건씩 총 초기 180건이며, 실습 후 실제 row 수는 다를 수 있습니다.

강의의 `task_id`/`owner_id`는 Tool 개념 예시입니다. 이 DB에는 그 필드가 없습니다. 실습의 `lookup_status`는 **과제 제출 상태가 아니라 보고서 인덱싱 상태**를 조회합니다. `author`도 인증된 user_id로 취급하지 않습니다.

## 설치·실행

1. Windows 11/VS Code의 기존 Python 가상환경 활성화.
2. 기존 CUDA PyTorch 빌드를 유지합니다. 필요 패키지만 `pip install -r requirements.txt`로 설치(다른 PyTorch 설치가 일어나는지 확인).
3. `.env.example`을 `.env`로 복사하고 **기존 report_agent_lab** 접속 URL, 활성 LLM_MODEL, API Key를 입력.
4. `python tools/check_environment.py`로 실제 PostgreSQL·CUDA 확인. 서버 실행 전에 `sql/check_existing_db_readonly.sql`의 SELECT 확인.
5. `python 01_basic/01_agent_vs_rag/main.py`로 시작. 각 수준은 `01_basic`, `02_intermediate`, `03_advanced`.
6. `python -m pytest -q`는 **오프라인 단위 계약 테스트**입니다. 실제 DB·GPU·LLM 테스트가 아닙니다.
7. 마지막: `python 03_advanced/08_end_to_end/main.py --serve` 후 `POST http://127.0.0.1:8008/api/agent/ask`에 JSON `{"question":"RAG 근거 인용 정확도 향상과 관련된 보고서의 내용을 설명하고, 보고서 63번의 인덱싱 상태도 알려줘."}` 전달.

**GPU**: E5 `cuda:0`, CrossEncoder `cuda:0` 필수. CPU로 자동 폴백하지 않습니다. 11/20 DB에 벡터가 없다면 **이전 11/20 프로젝트의** `scripts/index_overviews.py`를 이용해 기존 DB에 인덱싱했는지 강사와 확인하고, 새 seed를 실행하지 마세요.

**실제 성공 판정**: DB preflight 연결, CUDA 장치 확인, 인덱싱/해시/Revision 맞는 벡터 검색, LLM API 인증 및 Tool Calling, 2개의 실제 Tool 결과, final AIMessage, 출처·상태 확인을 각각 수행해야 합니다. 모델이 모든 요청에서 두 Tool을 반드시 선택하는 것은 아닙니다.

## 강의 Chapter ↔ 실습

- 01. [AI Agent·RAG·Workflow와 실제 Report DB](01_basic/01_agent_vs_rag/README.md)
- 02. [Tool Call ↔ ToolMessage와 실제 조회](01_basic/02_tool_message/README.md)
- 03. [@tool·Schema·Pydantic](01_basic/03_tool_schema/README.md)
- 04. [실제 CUDA E5·pgvector·GPU Reranker RAG Tool](01_basic/04_rag_tool/README.md)
- 05. [11/20 보고서 인덱싱 상태 Read-only Tool](01_basic/05_db_status_tool/README.md)
- 06. [create_agent와 두 Tool 통합](01_basic/06_create_agent/README.md)
- 07. [Agent 오류·권한·호출 제한](01_basic/07_reliability/README.md)
- 08. [FastAPI·Agent 통합·개인 프로젝트 연결](01_basic/08_end_to_end/README.md)

## 세 단계 목표

- Basic: TODO 2개씩, 최소 파일로 원리와 실제 DB 조회를 이해.
- Intermediate: TODO 3개씩, 데이터 계약과 검사 범위를 약간 확대.
- Advanced: TODO 4개씩, 오류·권한 정책·실행 로그·API 통합 검증에 집중.

**유의:** `ALLOWED_CATEGORIES`는 애플리케이션의 정적 범위 제한이며, 사용자별 로그인과 RBAC는 구현돼 있지 않습니다. 외부 서비스 배포는 금지합니다. 실제 Agent Tool의 출처 ID 일치 검사는 Claim Faithfulness 자동 보증이 아닙니다.

**인용 검증:** API 예시는 모델이 검증되지 않은 `RPT-번호`를 인용했으면 502 오류로 중단합니다. 출처 ID 검사는 답변 내용의 Faithfulness와 별도로 진행해야 합니다.
