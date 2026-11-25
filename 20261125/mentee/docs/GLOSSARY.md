# Day28 비전공자를 위한 핵심 용어 사전

| 용어 | 무엇인가 | 실무에서 어디에 쓰는가 |
|---|---|---|
| Dataset | 평가할 질문·정답 기준·문서 버전 모음 | RAG 기능 배포 전 회귀 테스트 입력 |
| Target Runner | RAG가 **정답을 모르는 상태**에서 질문을 실행하는 코드 | `Runnable.invoke()`·FastAPI 요청 |
| Evaluator | 실행된 결과를 Gold와 비교해 점수를 계산·판정하는 코드 | Recall 계산기, LLM Judge, 사람 평가표 |
| Report | 질문별 성능·실패·환경·개선책을 정리한 문서 | 배포 전 품질회의·A/B 비교 |
| Gold Label | 질문과 실제 관련 있는 문서/청크 ID | Retrieval 정답 판정 |
| Graded Relevance | 각 문서 관련성의 0·1·2·3 등급 | DCG·nDCG 계산 |
| Unjudged | 아직 관련성 판정을 하지 않은 문서 | **0등급으로 가정하지 않고** 검토 필요 |
| Recall@k | 전체 관련 문서 중 검색 상위 k개가 찾은 비율 | 검색 누락 진단 |
| Precision@k | 상위 k개 검색 결과 중 실제 관련 문서의 비율 | 불필요한 Context 후보 점검 |
| Hit@k | 관련 문서가 상위 k개에 1개라도 있는지(0/1) | 빠른 검색 결과 포함 여부 |
| RR·MRR | 첫 관련 문서 순위의 역수·질문별 평균 | 답 문서가 앞에 있는지 평가 |
| DCG | 관련성 등급과 순위 할인으로 계산한 점수 | 여러 관련 문서의 순위 평가 |
| IDCG | 같은 문서로 가장 이상적인 순서를 가정한 DCG | DCG를 정규화하는 기준 |
| nDCG | DCG/IDCG | 등급 있는 검색 순위 비교 |
| Faithfulness | 답변의 주장이 실제 Context로 지지되는 정도 | 환각·Grounding 검수 |
| Answer Relevance | 질문에 적절히 답하는 정도 | 다른 질문으로 새는 답변 확인 |
| Answer Correctness | 별도로 정한 Reference Answer와 사실이 일치하는 정도 | 사실 정답 검토 |
| RunnableLambda | Python 함수를 LangChain Runnable로 감싼 실행 단위 | 기존 RAG/평가 함수를 파이프라인에 연결 |
| `invoke()` | Runnable에 데이터를 넣어 **실제 실행**하는 메서드 | 질문 1건 실행·Trace 수집 |
| Trace | 단계별 입력·출력·시간·실패 위치 기록 | 모델/DB/Reranker 오류 진단 |
| Offline A/B | 동일 평가셋으로 두 코드 버전을 비교 | Rewrite·Rerank 개선 여부 비교 |
| Quality Gate | 정해 놓은 최소 통과 조건 | Agent Tool 연결 전 1차 검토 |
| No-answer | 현재 근거에 답이 없는 질문 | 부정확한 확답 대신 거절 품질 평가 |
| LangSmith | LangChain 생태계의 평가·추적 도구 | **선택적** 실험 추적/비교 (필수 설치 아님) |
