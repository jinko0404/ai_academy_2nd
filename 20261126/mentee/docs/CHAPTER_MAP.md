# Day29 강의 ↔ 실제 실습 대응

| Chapter | 핵심 개념 | 실제 실습 연결 |
|---|---|---|
| 01 | Agent vs RAG/Workflow | 기존 DB를 조회하며 데이터 소스·계획 분리 |
| 02 | Tool Call·ToolMessage | 실제 DB SELECT 결과를 call_id에 대응 |
| 03 | @tool·Schema | LangChain Tool과 report_id 유효성 |
| 04 | RAG Tool | 실제 GPU E5 + pgvector + GPU Reranker + LLM API |
| 05 | Read-only DB Tool | reports.index_status/revision, vector freshness |
| 06 | create_agent | RAG·DB 두 Tool 등록, ToolMessage/최종 AIMessage |
| 07 | Reliability | invalid ID, 호출 한도, 실패/보안 경계 |
| 08 | End-to-End | 실제 Agent를 로컬 FastAPI POST에 연결 |

비교할 것: 강의의 교육용 과제 제출 상태 DB 예시는 실제 11/20 Schema에 존재하지 않음. 이 실습은 보고서 인덱싱 상태로 기능을 대응합니다.
