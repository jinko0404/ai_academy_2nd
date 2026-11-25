# 강의 HTML ↔ 실습·코드·출력 맵

**기준 HTML**: 11월 25일 Day28 최종 검수본 68페이지.

| 실습 | HTML 페이지 | Basic | Intermediate | Advanced |
|---|---:|---|---|---|
| 01 검색 실패 위치 진단 | 9 | `01_basic/01_failure_diagnosis/main.py` | `02_intermediate/01_failure_diagnosis/steps.py`, `checks.py` | `03_advanced/01_failure_diagnosis/app/` 아래 목적별 모듈 |
| 02 Gold Dataset 설계 | 18 | `01_basic/02_gold_dataset/main.py` | `02_intermediate/02_gold_dataset/steps.py`, `checks.py` | `03_advanced/02_gold_dataset/app/` 아래 목적별 모듈 |
| 03 Recall · Precision · Hit | 25 | `01_basic/03_retrieval_metrics/main.py` | `02_intermediate/03_retrieval_metrics/steps.py`, `checks.py` | `03_advanced/03_retrieval_metrics/app/` 아래 목적별 모듈 |
| 04 RR · MRR · nDCG | 33 | `01_basic/04_ranking_metrics/main.py` | `02_intermediate/04_ranking_metrics/steps.py`, `checks.py` | `03_advanced/04_ranking_metrics/app/` 아래 목적별 모듈 |
| 05 Faithfulness · Relevance · Correctness | 42 | `01_basic/05_generation_quality/main.py` | `02_intermediate/05_generation_quality/steps.py`, `checks.py` | `03_advanced/05_generation_quality/app/` 아래 목적별 모듈 |
| 06 LangChain Evaluation Runner | 51 | `01_basic/06_langchain_runner/main.py` | `02_intermediate/06_langchain_runner/steps.py`, `checks.py` | `03_advanced/06_langchain_runner/app/` 아래 목적별 모듈 |
| 07 A/B 비교 · 실패 원인 | 60 | `01_basic/07_debug_comparison/main.py` | `02_intermediate/07_debug_comparison/steps.py`, `checks.py` | `03_advanced/07_debug_comparison/app/` 아래 목적별 모듈 |
| 08 통합 평가 · Agent Quality Gate | 68 | `01_basic/08_agent_quality_report/main.py` | `02_intermediate/08_agent_quality_report/steps.py`, `checks.py` | `03_advanced/08_agent_quality_report/app/` 아래 목적별 모듈 |

## 실습 구조와 코드 데이터 흐름

`data/questions.json` → Target 입력 정제 (`question`,`category`,`history`) → `RunnableLambda(...).invoke()` → 검색·생성 `trace` → **별도** `gold_labels.json` → Evaluator → `outputs/result.json` / CSV / Markdown Report → Agent Tool 1차 Gate.

**같은 source code 파일 ≠ 같은 실행 노드:** `Target Runner`는 기존 RAG를 호출하고, `Evaluator`는 실행 후 Gold를 읽습니다. `Report`는 결과를 종합하며 DB 검색을 수행하지 않습니다.

### Basic / Intermediate / Advanced

- **Basic:** 한 파일에서 함수를 읽고 실행 순서와 반환값을 확인합니다.
- **Intermediate:** 기능과 검사 로직을 분리하고 Python Import와 단위 테스트로 연결합니다.
- **Advanced:** Metrics, Datasets, Target, Evaluator, Reporting, Policy로 목적을 분리합니다. 실습 01처럼 규모가 작은 과제는 필요한 폴더만 생성합니다.

### 평가 개념에서 흔히 틀리는 부분

- **Hit@k:** 관련 문서가 1개라도 있으면 1 (문서 개수를 뜻하지 않음)
- **Recall@k:** Top-k에서 찾은 Gold 문서 수 / 전체 Gold 문서 수
- **Precision@k:** Top-k에서 찾은 Gold 문서 수 / k (이 수업의 고정 분모 정책)
- **MRR@k:** 질문마다 첫 Gold 문서 순위의 역수(RR)를 구한 후 **평가 가능한 질문끼리** 평균
- **nDCG@k:** 등급이 높은 문서가 앞에 있는지 정규화하여 평가. 이 실습은 비관련 문서도 0으로 사전 판정한 등급표를 사용합니다.
- **Faithfulness:** 주장과 실제 Context 일치. Claim 레이블은 교육용 사람 판정이며 자동 추론이 아닙니다.
- **No-answer:** Gold가 0개인 질문의 Recall은 `None`으로 처리하고, 올바르게 답변을 거절했는지 별도로 봅니다.
