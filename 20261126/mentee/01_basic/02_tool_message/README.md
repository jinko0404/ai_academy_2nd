# Day29 · 02 · AIMessage.tool_calls / ToolMessage · 기본필수

**연결된 강의자료:** Day29 Chapter 02 실습 안내. 11/27 개인 프로젝트 준비.

**목표:** 모델의 호출 요청과 Python DB 조회 완료 메시지를 명확히 구분합니다.

**중요한 사실:** 11/20 기존 PostgreSQL `report_agent_lab`을 재사용합니다. `CREATE DATABASE`, `DROP`, `schema.sql`, `seed.sql` 실행 금지. **실제 DB를 SELECT로 조회합니다.**

**실습 전제:** AIMessage.tool_calls는 요청, ToolMessage는 실제 도구 실행 결과입니다. 이 실습은 실제 DB 조회를 사용해 구조를 설명합니다.

## 진행 순서

1. `../../.env`를 설정합니다. 이미 11/20 실습에서 쓰던 DB를 지정해야 합니다.
2. 다음 TODO 2개를 `tasks.py`에서 해결합니다: `D29-02-B-01` make_request, `D29-02-B-02` match_result.
3. **실제 실행:** `python 01_basic/02_tool_message/main.py` (역할 루트에서 실행; 기존 DB 사용).
4. 구조 점검만 필요하면 `python 01_basic/02_tool_message/main.py --check`. 실제 DB/GPU 실행 성공을 뜻하지 않습니다.
5. 최종 제출: 수정한 `tasks.py`, 실제 실행 출력(비밀값 제거), 3~5줄 학습 기록.

## 실습 확인 기준

- Tool 입력에 `task_id` 또는 존재하지 않는 `owner_id`를 넣지 않습니다. **기존 DB의 `report_id`와 `index_status`**만 조회합니다.
- Tool Call의 요청과 실제 ToolMessage/DB 응답은 다른 단계입니다.
- 실제 Embedding과 Reranker는 CUDA GPU를 사용하며 CPU fallback을 허용하지 않습니다.
- 외부 LLM API가 필요한 실습에는 실제 API 호출 비용이 발생합니다.
- 검색 결과를 읽을 때 `report_read_view` 원문과 `reports` 인덱싱 상태를 구분합니다.
- 고정 예시를 실제 E5/DB 결과라고 부르지 않고, DB 변경 없이 오류를 점검합니다.

## 용어·예시

**예시 요청:** RAG 근거 인용 정확도 향상과 관련된 보고서의 내용을 설명하고, 보고서 63번의 인덱싱 상태도 알려줘.

보고서 63은 기존 `research_plans`의 교육용 레코드입니다. `index_status`가 ready인지 여부는 학생 PC의 현재 DB 상태를 조회해야만 알 수 있습니다.
