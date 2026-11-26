# 11/20 SQL 분석·Day29 Tool 매핑

- DB 이름 `report_agent_lab`, PostgreSQL에 직접 설치한 pgvector `vector(384)`.
- `reports`: report_id / category_id / overview / index_status(pending/ready/failed) / revision.
- `report_embeddings`: report_id PK/FK, embedding(384), model_name, overview_hash, indexed_revision.
- `report_read_view`: 6개 상세 테이블의 공통 title/content/author를 UNION ALL로 묶은 VIEW.
- 처음 6개 카테고리 × 각 30건 = 180건, prompt_versions 60건. 실제 학생 PC 수량은 변할 수 있습니다.
- 검색용 **passage**는 이미 인덱싱된 `overview`입니다. 질문은 같은 모델의 `query: `로 CUDA 추론합니다.
- 인덱스 쿼리는 `index_status='ready'`, 동일 E5 모델, `indexed_revision=revision`, overview SHA256을 조건으로 합니다.
- CrossEncoder는 실제 보고서 원문을 일부 읽어 CUDA에서 재정렬합니다.
- Agent Tool 1 `rag_search`: 실제 검색 → 컨텍스트 선택 → LLM API 답변 → **제공된 근거 출처** 반환. Faithfulness는 별도 평가.
- Agent Tool 2 `lookup_status`: `reports` + `report_categories` + `report_embeddings` SELECT로 report_id의 인덱싱 상태 조회.
- **없음**: tasks, task_id, submission status, owner_id, users, AuthZ ACL. 임의로 DB에 추가하거나 있다고 주장하지 않습니다.
- 새 DB·새 테이블·데이터 재적재 금지. 실습 SQL 파일은 SELECT만 제공합니다. 모델 사용에 따른 API 비용 주의.

## 강의 사례와 실제 데이터의 차이

이 스키마에서 `report_embeddings`는 보고서 당 **1개 overview Vector**이며, `report_read_view`는 상세 원문을 제공합니다. Day26~28의 문서별 Chunk·citation_id 저장 구조와 동일하지 않습니다. Day29 실습의 인용은 보고서 단위 `RPT-<report_id>`이며, 문단별 위치까지 식별하는 기능은 추가 인덱싱 Schema가 필요하지만 이번에는 DB를 수정하지 않습니다.

`RPT-63`은 서버에서 실제 조회한 Report 레코드를 지칭합니다. 단순히 근거가 검색됐다는 사실만으로 LLM 답변의 Claim이 참임을 증명하지 않습니다.

**ToolRuntime 관련 유의:** 강의의 사용자별 `owner_id` 기반 Authorization은 개념 예시입니다. 실습 DB에 해당 테이블이 없으므로 이번 실습에서 사용자별 인증·소유권 검사 성공을 주장해서는 안 됩니다. 앱의 ALLOWED_CATEGORIES는 서버 설정에 따른 정적 허용범위입니다.
