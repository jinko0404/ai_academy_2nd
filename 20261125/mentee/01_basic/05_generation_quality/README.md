# Day28 실습 05 · Faithfulness · Relevance · Correctness · 01_basic

**강의 연결:** 2026-11-25, Day28 최종 HTML **42페이지** · Faithfulness · Relevance · Correctness

## 무엇을 학습하나요?

Claim의 근거 일치와 질문 적합성·정답성을 구분합니다.

Faithfulness는 실제 Context와 주장 일치, Correctness는 참조 정답과의 사실 관계입니다.

## 준비사항

- Python 3.12, Windows 11, VS Code, Git Bash, 프로젝트 역할 루트의 `.venv`
- 강의 HTML 68페이지에서 해당 실습 안내를 먼저 확인
- 공통 데이터는 역할 폴더의 `data/`를 사용. 이전 날짜 과제 완료 여부와 무관하게 시작 가능

## 사용할 파일

- `main.py`: 실행 진입점
- `main.py`: 실행 진입점

## TODO 요구사항 — 수강생 체크리스트

- **D28-05-B01 · `claim_faithfulness()`**: Claim 목록에서 근거가 지지하는 Claim 비율을 계산하세요. Claim이 없으면 None.  \n  **힌트:** 모든 Claim의 supported는 교육용 사람 레이블입니다.
- **D28-05-B02 · `describe_claim_result()`**: 비지원 Claim을 이름과 함께 기록하세요. “모든 Claim 지원”과 “Claim 없음”은 구별합니다.  \n  **힌트:** Faithfulness는 Reference Answer와의 비교가 아닙니다.

## 실행 명령 (VS Code Git Bash, `mentee/` 기준)

```bash
python 01_basic/05_generation_quality/main.py --offline
python -m pytest -q 01_basic/05_generation_quality/tests
```

실습 06~08은 **TODO 완성 후**, CUDA·PostgreSQL·LLM API 환경 준비 시 아래 실제 경로도 실행할 수 있습니다.

```bash
python 01_basic/05_generation_quality/main.py --live
```

- **`--offline`(기본):** 실제 모델/DB를 호출하지 않는 고정 교육용 사례입니다. 결과가 좋아도 모델 성능이라 주장하지 마세요.
- **`--live`:** 06~08에서만 지원하며 실제 LangChain·E5(CUDA)·pgvector·Cross-Encoder(CUDA)·ChatOpenAI를 사용합니다. 외부 호출 비용 및 사전 DB 준비가 필요합니다.
- 실패 시 어느 TODO인지 확인하고 먼저 작은 입력으로 테스트하세요. 수강생 버전은 미완성 TODO를 실행하면 오류가 나도록 설계되어 있습니다.

## 결과물과 완료기준

- `outputs/result.json` : 수행 결과. `mode`, `question_id`, 평가값과 No-answer 처리 여부 확인
- `expected/EXPECTED.md` : 결과 필드와 해석 방법 확인
- `reports/review.md` : 본인 출력·오류·해석·다음 날 Agent와의 연결 기록
- **완료 기준:** Claim의 근거 일치와 질문 적합성·정답성을 구분합니다. Gold/정답 레이블이 실행 대상 Target으로 누출되지 않고, 평가 결과를 구체적인 질문·문서 ID로 설명할 수 있어야 합니다.

## 현업에서 중요한 점

- `question_id`와 데이터·코드 버전으로 실행을 재현할 수 있도록 합니다.
- Gold 없는 No-answer는 일반 Retrieval 지표 평균과 분리합니다.
- Retrieval 지표와 Generation의 Claim/답변 검토는 서로 다른 판단입니다.
- Day29 Agent Tool로 연결하기 전에는 출처·권한·오류도 검증해야 합니다.
