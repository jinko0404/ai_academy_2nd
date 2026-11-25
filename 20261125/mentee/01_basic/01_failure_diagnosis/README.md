# Day28 실습 01 · 검색 실패 위치 진단 · 01_basic

**강의 연결:** 2026-11-25, Day28 최종 HTML **9페이지** · Definition · 검색 vs 생성 오류

## 무엇을 학습하나요?

질문의 실패 단계와 조치 근거를 기록합니다.

검색 후보 ID와 실제 Context ID는 다릅니다.

## 준비사항

- Python 3.12, Windows 11, VS Code, Git Bash, 프로젝트 역할 루트의 `.venv`
- 강의 HTML 68페이지에서 해당 실습 안내를 먼저 확인
- 공통 데이터는 역할 폴더의 `data/`를 사용. 이전 날짜 과제 완료 여부와 무관하게 시작 가능

## 사용할 파일

- `main.py`: 실행 진입점
- `main.py`: 실행 진입점

## TODO 요구사항 — 수강생 체크리스트

- **D28-01-B01 · `classify_case()`**: case의 system_error·candidate_has_gold·context_has_gold·answer_grounded를 차례로 확인하고 stage 문자열을 반환하세요. 시스템 장애와 검색 실패를 구분하세요.  \n  **힌트:** 먼저 System → Candidate → Context → Generation 순서로 판단합니다.
- **D28-01-B02 · `describe_evidence()`**: 분류된 stage를 설명할 수 있는 구체적인 한 줄의 근거를 반환하세요. 빈 문자열은 안 됩니다.  \n  **힌트:** 후보 누락과 Context 누락을 구별하는 문장으로 작성합니다.

## 실행 명령 (VS Code Git Bash, `mentee/` 기준)

```bash
python 01_basic/01_failure_diagnosis/main.py --offline
python -m pytest -q 01_basic/01_failure_diagnosis/tests
```

실습 06~08은 **TODO 완성 후**, CUDA·PostgreSQL·LLM API 환경 준비 시 아래 실제 경로도 실행할 수 있습니다.

```bash
python 01_basic/01_failure_diagnosis/main.py --live
```

- **`--offline`(기본):** 실제 모델/DB를 호출하지 않는 고정 교육용 사례입니다. 결과가 좋아도 모델 성능이라 주장하지 마세요.
- **`--live`:** 06~08에서만 지원하며 실제 LangChain·E5(CUDA)·pgvector·Cross-Encoder(CUDA)·ChatOpenAI를 사용합니다. 외부 호출 비용 및 사전 DB 준비가 필요합니다.
- 실패 시 어느 TODO인지 확인하고 먼저 작은 입력으로 테스트하세요. 수강생 버전은 미완성 TODO를 실행하면 오류가 나도록 설계되어 있습니다.

## 결과물과 완료기준

- `outputs/result.json` : 수행 결과. `mode`, `question_id`, 평가값과 No-answer 처리 여부 확인
- `expected/EXPECTED.md` : 결과 필드와 해석 방법 확인
- `reports/review.md` : 본인 출력·오류·해석·다음 날 Agent와의 연결 기록
- **완료 기준:** 질문의 실패 단계와 조치 근거를 기록합니다. Gold/정답 레이블이 실행 대상 Target으로 누출되지 않고, 평가 결과를 구체적인 질문·문서 ID로 설명할 수 있어야 합니다.

## 현업에서 중요한 점

- `question_id`와 데이터·코드 버전으로 실행을 재현할 수 있도록 합니다.
- Gold 없는 No-answer는 일반 Retrieval 지표 평균과 분리합니다.
- Retrieval 지표와 Generation의 Claim/답변 검토는 서로 다른 판단입니다.
- Day29 Agent Tool로 연결하기 전에는 출처·권한·오류도 검증해야 합니다.
