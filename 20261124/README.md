# Day27 · LangChain RAG & Reranking — 초보자용 기본 개념·용어 정리

> **수업 날짜:** 2026-11-24 · **주제:** Query Optimization — 검색 전 질문 개선과 검색 후 후보 재정렬  
> **기준 자료:** 사용자가 제공한 `Day27_LangChainRAG_Reranking_정리.md`  
> **학습 목표:** 사용자의 질문이 들어온 순간부터 출처가 달린 답변이 돌아갈 때까지 **어떤 함수·모델·DB가 어떤 순서로 움직이는지** 설명하고, 새 용어의 의미를 구분할 수 있다.

**먼저 기억할 한 문장:** **Query Rewrite는 검색하기 *전에 질문*을 다듬고, Reranking은 검색한 *다음 문서 후보의 순서*를 다듬는다.**

이 문서는 원본 강의자료의 내용을 쉬운 말로 재구성한 학습 노트다. 실습 사례의 **C1~C5, 순위, 관련성 점수는 수업에서 미리 만든 설명용 예시**로서 실제 E5·DB·Cross-Encoder·LLM을 실행해 얻은 결과가 아니다. 함수명과 파일 경로는 수업 프로젝트 예시이므로 모든 프로젝트에 공통으로 적용되는 이름은 아니다.

---

## PART A. 이번 Day27 수업의 위치부터 이해하기

### 1. 어제 만든 RAG에 무엇을 추가했을까?

**Day26(기존):**

```text
사용자 질문
  ↓
Retriever: E5 + PostgreSQL/pgvector로 문서 후보 검색
  ↓
Context Builder: 답변에 사용할 자료 선택
  ↓
Prompt: 규칙 + 질문 + 근거
  ↓
Chat Model: 답변 생성
  ↓
Validation: 형식·출처 확인
  ↓
FastAPI: 답변과 출처 반환
```

**Day27(개선):**

```text
사용자 질문 + 대화 History
  ↓
Query Rewrite                 ← 새로 추가: 검색 전에 질문 명확히 하기
  ↓
(필요하면 Expansion·Multi-query) ← 선택적 확장: 다양한 표현으로 찾아보기
  ↓
Retriever: 기존 E5 + pgvector ← 기존 검색 기능 재사용
  ↓
Candidate Documents          ← 1차 검색에서 발견한 문서 후보
  ↓
Cross-Encoder Reranking      ← 새로 추가: 후보 문서의 관련성 재평가
  ↓
Context Builder              ← 최종 문서 선택
  ↓
Prompt → Chat Model → Validation
  ↓
FastAPI: 검증된 답변과 출처 반환
```

**왜 추가했을까?** 기존 Retriever가 문서를 찾더라도 사용자가 질문을 짧게 하거나, 정확한 문서가 검색 후보 뒤쪽에 있다면 최종 답변의 근거로 사용되지 못할 수 있기 때문이다.

### 2. 검색 전에 하는 일과 검색 후에 하는 일을 나누자

| 시점 | 이름 | 쉽게 말해 | 수정하는 대상 |
|---|---|---|---|
| Retriever **이전** | Query Rewrite | 생략된 말을 채워 검색 가능한 질문으로 변경 | 질문 |
| Retriever **이전** | Query Expansion | 같은 뜻의 검색 표현을 보강 | 검색 표현 |
| Retriever **이전/검색 반복** | Multi-query | 서로 다른 검색어로 여러 번 검색하고 합침 | 검색 횟수·후보 집합 |
| **1차 검색** | Candidate Retrieval | DB에서 관련 있어 보이는 자료를 빠르게 가져옴 | 후보 문서 집합 |
| Retriever **이후** | Reranking | 이미 찾은 문서끼리 관련성 순위 재평가 | 후보의 순서 |
| Reranking **이후** | Context Builder | 답변 생성 모델이 실제 읽을 근거만 선택 | 최종 근거 |

**주의:** Rewrite·Reranking은 검색 품질을 높일 *가능성*이 있는 기술이지, 넣기만 하면 항상 정답률이 오르는 기술이 아니다.

### 3. 문서는 이미 DB에 준비되어 있다

Day27에서는 Day24~26에서 구축한 문서 Chunk와 E5 벡터를 **재사용**한다.

```text
[이전 수업에서 미리 수행]
문서 → Parsing → Cleaning → Chunking → E5 Embedding → PostgreSQL + pgvector 저장·공개

[이번 수업에서 질문마다 수행]
질문 → Rewrite → E5 질문 벡터 → pgvector 후보 검색 → Cross-Encoder Rerank → 답변
```

**Cross-Encoder를 추가했다고 저장된 문서를 다시 Embedding하거나 재색인하는 것은 아니다.** 수업의 Day27 실습은 E5 로컬 추론을 기존 CPU 설정에서 CUDA GPU로 실행하도록 바꾸지만, E5 모델과 저장 벡터의 생성 규칙 자체는 그대로 유지한다.

---

## PART B. 사용자 질문부터 최종 답변까지 절차적으로 설명

하나의 상황을 끝까지 사용하자.

- 이전 사용자 질문: **"회의실 예약 방법은?"**
- 이번 사용자 질문: **"그럼 취소는?"**
- 실제 찾아야 할 정보: **회의실 예약 취소 절차**

각 단계마다 **담당 → 입력 → 처리 → 출력 → 이유**를 구분한다.

### STEP 1. 사용자 → FastAPI: 질문을 접수한다

**담당:** FastAPI의 API 요청 처리 함수. 요청 데이터 형식은 Pydantic 모델이 검사한다.

**입력 예시:**

```python
request = {
    "question": "그럼 취소는?",
    "history": [{"role": "user", "content": "회의실 예약 방법은?"}],
    "category": "시설",
    "candidate_k": 10,
    "context_k": 3,
    "use_rewrite": True,
    "use_rerank": True,
}
```

**하는 일:** FastAPI는 HTTP 요청을 접수하고 해당 Python 함수로 연결한다. Pydantic은 `question`·`category` 등의 필드 형식과 설정 범위를 검사한다.

**출력:** 검증된 요청 데이터가 서비스 처리 코드로 전달된다.

**왜?** 잘못된 데이터로 검색·모델 호출을 시작하지 않기 위해서다.

수업 예시 설정:

| 입력 항목 | 의미 | 수업 예시 |
|---|---|---|
| `question` | 사용자 원 질문 | 1~1000자 |
| `history` | 이전 대화의 일부 | 최대 6개 항목 |
| `category` | 조회할 문서 분류 | `학사` / `시설` / `지원` |
| `candidate_k` | 1차 검색 후보 최대 개수 | 기본 10, 1~20 |
| `context_k` | Context에 넣을 자료 최대 개수 | 기본 3, 1~3 |
| `use_rewrite` | 질문 재작성 기능 사용 여부 | 기본 True |
| `use_rerank` | 후보 재정렬 기능 사용 여부 | 기본 True |

*주의: 위 수치는 Day27 강의의 예시 API 계약이며, 다른 서비스의 고정 규칙은 아니다.*

### STEP 2. Query Rewrite: 생략된 질문을 명확하게 만든다

**담당:** 서비스 코드가 만든 Rewrite Chain. 내부적으로 `ChatPromptTemplate`과 Chat Model을 연결하여 사용할 수 있다.

**입력:** 사용자 원 질문 + 관련된 이전 대화.

```text
History: 회의실 예약 방법은?
현재 질문: 그럼 취소는?
```

**하는 일:** 이전 대화에서 "취소"의 대상이 회의실 예약임을 확인하고, **검색용 독립 질문**을 만든다.

**출력:**

```python
question = "그럼 취소는?"                   # 원래 사용자 질문: 보존
search_query = "회의실 예약 취소 방법은?"   # 새로 만든 검색용 질문
```

**왜?** 원 질문만 검색하면 "무엇을 취소하는지" 알기 어려워 관련 없는 문서를 찾을 수 있기 때문이다.

**절대 하면 안 되는 일:** 대화에 없던 "무료 취소", "3일 이내", "승인 없이 취소" 같은 조건을 상상해서 추가하는 것. 이를 **Semantic Drift(의미 이탈)**이라고 한다.

**이전 대화로 대상을 확정할 수 없다면?** `needs_clarification` 상태로 멈추고 사용자에게 대상을 확인한다. 이 경우 검색을 진행하지 않는다.

**중요한 구분:** `search_query`는 Retriever와 Reranker용이다. **최종 Prompt는 원래 사용자 질문 `question`을 보존해 사용**한다.

### STEP 3. (선택) Query Expansion / Multi-query: 검색 표현을 넓힌다

**담당:** 검색어 생성 및 검색 결과 통합 Python 함수. 수업 예시에서는 `prepare_queries()`·`search_queries_step()` 등을 사용한다.

**입력:** STEP 2에서 만든 독립 질문.

**하는 일:** 질문 의도를 유지하는 다른 표현을 추가할 수 있다.

```text
Q1: 회의실 예약 취소 방법은?
Q2: 회의실 예약 철회 절차는?
Q3: 회의실 예약 취소 가능 기간은?
```

각 질문으로 기존 Retriever 검색을 반복한 다음 후보를 합친다. 예를 들어:

```text
Q1 검색 → C1, C2
Q2 검색 → C2, C3
합친 결과 → C1, C2, C3
              (C2는 중복이므로 한 번만)
```

**출력:** 중복을 제거한 후보 목록. 수업에서는 `chunk_id`를 기준으로 중복을 제거한다.

**왜?** 문서가 "취소" 대신 "철회"라는 단어를 사용하면 단일 표현으로 관련 문서를 놓칠 수 있기 때문이다.

**주의:** 이 기능은 선택적이다. 명확한 질문은 그대로 검색할 수도 있다. 모든 질의에서 **같은 Category·문서 버전·권한 필터**를 유지해야 하며, 검색어 수가 많다고 정확도가 반드시 높아지는 것은 아니다.

### STEP 4. Retriever: 기존 E5 + pgvector로 1차 후보를 검색한다

**담당:** Day26의 Python 검색 함수와 `RunnableLambda`로 감싼 Retriever 실행 객체. 실제 벡터 변환은 E5 모델이, DB 검색은 PostgreSQL과 pgvector가 수행한다.

**입력:** `search_query`, `category`, `candidate_k`.

**실행 순서:**

1. Python 검색 함수가 검색용 질문을 E5 Embedding 모델에 전달한다.
2. E5가 질문을 **384차원 벡터**로 바꾼다. 문서에는 미리 저장해 둔 E5 벡터를 사용한다.
3. Python 코드가 PostgreSQL에 검색 요청을 보낸다.
4. PostgreSQL은 pgvector 거리 연산과 SQL 조건을 사용해 가까운 문서를 조회한다.
5. 검색 함수가 현재 공개 버전·Category·권한 등 허용된 범위에서 **최대 `candidate_k`개의 후보**를 반환한다.

**출력 예시:**

```text
Candidate-k = 10 (상한)
실제로 검색된 예시 후보: C1, C2, C3
C1: 회의실 예약 신청 방법
C2: 시설 위치 안내
C3: 회의실 예약 취소 절차
```

**왜?** 전체 문서를 Cross-Encoder로 하나하나 비교하면 계산 비용이 크므로, 먼저 빠른 E5 검색으로 후보 수를 줄이기 위해서다.

**여기서 알아둘 것:** `candidate_k=10`은 **최대 10개**이지, 반드시 10개를 반환한다는 뜻이 아니다. 또한 질문이 잘못돼 관련 문서가 후보에 포함되지 않았다면 다음 Reranker는 그 문서를 새로 가져올 수 없다.

### STEP 5. Hit → Document: 후보의 본문과 출처를 함께 보관한다

**담당:** Python의 검색 결과 변환 코드(`hits_to_documents()` / 어댑터).

**입력:** DB 검색 결과인 Hit 목록.

**하는 일:** 검색 결과의 본문을 LangChain `Document.page_content`에, 출처 ID·원문 위치 등을 `Document.metadata`에 옮긴다.

```text
DB 검색 결과 Hit
  ├─ body       → Document.page_content
  ├─ chunk_id   → Document.metadata["chunk_id"]
  ├─ citation_id→ Document.metadata["citation_id"]
  └─ 원문 출처   → Document.metadata["source_locator"]
```

**출력:** `candidate_docs: list[Document]`.

**왜?** 뒤쪽 Reranker와 Context Builder가 본문을 읽으면서도 어떤 문서에서 왔는지 잃어버리지 않도록 하기 위해서다.

**중요:** `Document`는 자료를 담는 **데이터 객체**이지, 스스로 검색·재정렬·LLM 호출을 실행하는 기능이 아니다.

### STEP 6. Reranker: 검색한 후보의 순서를 다시 매긴다

**담당:** `rerank_step()` 같은 Python 함수 + `BAAI/bge-reranker-v2-m3` Cross-Encoder 모델.

**입력:** `search_query` + `candidate_docs`.

**하는 일:** Cross-Encoder가 질문과 후보 문서를 **한 쌍으로 함께 읽고** 관련성 점수를 계산한다. 후보 수만큼 질문–문서 쌍을 만들어 비교한다.

```text
질문: 회의실 예약 취소 방법은?

(질문, C1 예약 신청 문서) → 관련성 점수
(질문, C2 시설 위치 문서) → 관련성 점수
(질문, C3 예약 취소 문서) → 관련성 점수
                                ↓
                           점수순으로 재정렬
```

**변경 전 / 변경 후 예시:**

| 1차 검색 순위 | Reranking 이후 순위 |
|---|---|
| 1위 C1 (예약 신청) | **1위 C3 (예약 취소)** |
| 2위 C2 (시설 위치) | 2위 C1 (예약 신청) |
| 3위 C3 (예약 취소) | 3위 C2 (시설 위치) |

**출력:** `ranked_docs: list[Document]` — 같은 후보 문서들을 **관련성 순서대로 재배치**한 목록.

**왜?** 처음 검색의 거리 순위가 사용자 질문에 대한 정답의 우선순위와 반드시 일치하지는 않기 때문이다.

**세 가지 주의:**

- Reranker는 **기존 Candidate 안의 순위만 바꾼다.** 후보에 없는 문서를 찾아오지 않는다.
- Reranker 점수는 **정답 확률이 아니다.** 후보가 모두 틀려도 상대적 1위는 존재한다.
- 원래의 문서 ID와 Metadata는 계속 보존한다. 문서의 텍스트나 ID를 새로 만들지 않는다.

### STEP 7. Context Builder: LLM에게 실제 보여 줄 자료를 고른다

**담당:** Day26에서 사용한 `build_context()` 등 Python 코드.

**입력:** Reranking을 거친 `ranked_docs`, `context_k`, 길이 예산.

**하는 일:** 순위가 높은 자료 중에서 중복·개수·길이 조건에 맞는 Chunk를 선택하고 본문을 문자열로 묶는다.

```text
Rerank 결과: C3 → C1 → C2
Context-k = 2 (예시)
실제 포함: C3, C1
제외: C2
```

**출력:**

- `context`: LLM에게 실제 제공할 근거 텍스트
- `included_docs` 또는 `included_ids`: **실제로 Context에 넣은 문서/출처 ID 목록**

**왜?** LLM이 모든 후보를 읽을 수는 없고, 선택한 근거만 읽기 때문이다.

**출처 관련 중요 사항:** 수업 데모의 **C1/C2/C3는 가상 후보 라벨**이다. 실제 답변의 `source_ids`에는 이를 그대로 넣는 것이 아니라 **DB에서 받은 양의 정수 `citation_id`**를 사용한다. Context에 포함한 자료의 정수 ID만 인용할 수 있다.

### STEP 8. Prompt: 답변 생성 요청을 만든다

**담당:** `ChatPromptTemplate`.

**입력:** 원 사용자 질문 `question` + Context + 서비스 답변 규칙.

```text
System: 제공된 근거만으로 답하고 없는 사실은 추측하지 마세요.
Human: 그럼 취소는?
Context: [실제 출처 ID] 회의실 예약을 취소하려면 운영실에 요청합니다.
```

**출력:** `ChatPromptValue` — 모델이 읽을 메시지 묶음.

**왜?** LLM이 질문과 근거, 답변 규칙을 한 번의 모델 입력으로 받을 수 있도록 하기 위해서다.

**주의:** `prompt.invoke(...)`는 **메시지를 만드는 작업**이다. 그 자체로 외부 LLM API를 호출하지 않는다.

### STEP 9. Chat Model: 답변을 생성한다

**담당:** `ChatOpenAI` 같은 Chat Model 연결 객체가 호출한 외부 LLM.

**입력:** Prompt 메시지.

**하는 일:** `generation_chain.invoke(...)` 또는 `chat_model.invoke(...)` 실행 시 실제 모델 API를 호출해 답변을 받는다.

**출력:** 생성 답변(`AIMessage` 또는 Structured Output 결과).

**왜?** 검색된 규정 내용을 참고하여 사용자가 이해할 수 있는 자연어 답변을 만들기 위해서다.

### STEP 10. Validation과 출처 연결: 답변을 확인한다

**담당:** 애플리케이션의 검증 함수와 서버 측 출처 연결 코드.

**입력:** 모델 응답 + 실제 Context에 포함한 문서 ID와 Metadata.

**하는 일:**

1. 요청한 출력 형식과 상태 값이 올바른지 검사한다.
2. 답변이 인용한 `source_ids`가 **실제로 Context에 포함한 정수 `citation_id`**인지 확인한다.
3. 서버가 보관한 Metadata에서 원문 위치(`source_locator`)를 가져와 출처를 연결한다.
4. 답변의 실제 주장·숫자·조건·예외가 원문과 맞는지도 별도로 대조한다(Grounding).

**출력:** 검증된 답변과 출처 목록, 또는 정보 부족·오류 상태.

**중요:** `source_ids`가 맞는다고 답변의 내용까지 참인 것은 아니다. 또한 원문 경로는 LLM이 만들어 내는 것이 아니라 **서버가 저장해 둔 Metadata에서 연결**해야 한다.

### STEP 11. FastAPI: 사용자에게 최종 결과를 돌려준다

**담당:** FastAPI의 API 요청 처리 함수.

**입력:** 검증된 답변과 출처, 또는 상태·오류.

**출력:** HTTP 응답.

- 정상 처리: 답변과 출처 반환
- `needs_clarification`: 질문의 대상이 불명확하므로 확인 요청
- `insufficient_context`: 검색은 정상이나 필요한 근거를 찾지 못함
- `rewrite_failed`, `retrieval_failed`, `rerank_failed`, `generation_failed`: 해당 단계 실행 오류로 구분

**검색 결과가 0개인 경우**는 정상적인 정보 부족일 수 있지만, **DB 연결이 실패해 검색을 못 한 경우**는 정보 부족이 아니라 오류다.

---

## PART C. 핵심 개념을 더 쉽게 이해하기

### 1. Rewrite와 Reranking의 차이

**Rewrite = 질문을 고친다.**  
"그럼 취소는?" → "회의실 예약 취소 방법은?"

**Reranking = 검색 결과의 순서를 고친다.**  
C1(신청) → C2(위치) → C3(취소) 를 **C3 → C1 → C2**로 재정렬

| 질문 | 답 |
|---|---|
| 질문의 대상이 빠졌다면? | 검색 전 Rewrite 검토 |
| 문서는 찾았지만 정답이 뒤에 있다면? | 검색 후 Reranking 검토 |
| 정답 문서가 아예 후보에 없다면? | 검색 조건·Indexing·Candidate-k·질의 확장부터 확인 |
| 과거 버전 문서가 반환됐다면? | Metadata/Version/권한 필터를 확인 |

### 2. Query Rewrite · Expansion · Multi-query는 같은 말이 아니다

| 구분 | 하는 일 | 예시 |
|---|---|---|
| **Rewrite** | 생략된 대상을 복원해 **독립적인 질문 하나**로 정리 | "그럼 취소는?" → "회의실 예약 취소는?" |
| **Expansion** | 의미를 유지하는 관련 표현을 보강 | "예약 취소" + "예약 철회" |
| **Multi-query** | 표현이 다른 **질문 여러 개**로 각각 검색한 뒤 결과 합침 | 취소 방법/철회 절차 질문으로 반복 검색 |

Rewrite나 Expansion이 원래 사용자의 의도를 바꾸면 안 된다. 필요하지 않다면 생략할 수도 있다.

### 3. Bi-Encoder와 Cross-Encoder의 차이

| 구분 | Bi-Encoder (E5) | Cross-Encoder (BGE Reranker) |
|---|---|---|
| 언제 쓰나? | 1차 후보 검색 | 2차 후보 재정렬 |
| 무엇을 받나? | 질문과 문서를 **각각** 입력 | 질문과 문서를 **한 쌍으로** 입력 |
| 무엇을 내놓나? | 질문·문서 각각의 숫자 벡터 | 질문–문서 쌍의 관련성 점수 |
| 미리 저장할 수 있나? | 문서 벡터를 DB에 미리 저장 | 질문이 바뀔 때마다 후보와 다시 비교 |
| 특징 | 전체 문서에서 빠르게 후보를 좁힘 | 좁힌 후보에 대해 더 정밀한 비교 시도 |
| 주의 | 비슷한 거리 ≠ 정답 보장 | 높은 점수 ≠ 정답 확률 |

**기억하기:** **E5는 먼저 넓게 찾고, Cross-Encoder는 찾은 것들만 다시 꼼꼼히 비교한다.** Cross-Encoder가 항상 더 정확하다는 보장은 없고 실제 문서로 평가해야 한다.

### 4. Candidate-k와 Context-k의 차이

**`candidate_k=10`**은 Retriever가 검색할 후보를 **최대 10개** 가져오라는 뜻이다.  
**`context_k=3`**은 최종 Context에 문서를 **최대 3개** 넣으라는 뜻이다.

```text
첫 검색: 최대 10개 후보
        ↓
Reranker: 후보 10개를 점수화하고 재정렬
        ↓
Context Builder: 그중 최대 3개 선택
```

후보를 많이 가져오면 정답 문서를 포함할 가능성이 높아질 수 있지만, Cross-Encoder 계산량과 응답 시간도 증가한다. 반대로 후보가 적으면 정답을 놓칠 수 있다. 길이 예산 때문에 Context에 들어가는 문서 수는 `context_k`보다 적을 수도 있다.

### 5. Recall Ceiling: Reranker의 한계

```text
후보: C1, C2, C3    / 정답: C3 → C3를 1위로 끌어올릴 수 있음
후보: C1, C2, C4    / 정답: C3 → C3가 없으므로 절대로 끌어올릴 수 없음
```

**초기 검색에서 정답 후보를 놓치면 Reranking으로 고칠 수 없다.** 이를 **Recall Ceiling**이라고 한다. 원인이 검색 범위, Category·Version 필터, Chunking/Indexing, 검색어 등 어디에 있는지 먼저 확인해야 한다.

### 6. 원 질문과 검색용 질문은 왜 따로 보관하나?

사용자가 **"그럼 취소는?"** 이라고 했고, 검색용으로 **"회의실 예약 취소 방법은?"** 을 만들었다면:

| 변수 | 보관하는 값 | 사용 위치 |
|---|---|---|
| `question` | 사용자가 실제 작성한 문장 | 최종 Prompt·답변 대상·기록 |
| `search_query` | History로 복원한 독립 질의 | Retriever·Reranker |
| `search_queries` | Expansion/Multi-query를 위해 준비한 여러 질의 | 반복 검색 |
| `candidate_docs` | 1차 검색 후보 | Reranking 입력 |
| `ranked_docs` | 재정렬한 후보 | Context Builder 입력 |
| `context` | 실제 선택한 참고 본문 | LLM Prompt |
| `included_ids` | 실제 Context에 포함한 출처 정수 ID | 최종 Citation 검사 |

원 질문을 지우거나 모든 단계를 `question`이라는 하나의 변수로 덮어쓰면 어느 단계에서 의미가 바뀌었는지 추적하기 어렵다.

### 7. LangChain의 Runnable, `|`, `invoke()`

```python
# 일반 Python 함수는 실행하는 일을 정의한다.
def clean_question(state):
    return {**state, "search_query": state["question"].strip()}

# RunnableLambda는 함수를 LangChain 실행 단위로 감싼다.
step = RunnableLambda(clean_question)

# | 는 실행 순서를 연결한다. 아직 실행된 것이 아니다.
chain = step | another_step

# invoke()가 호출될 때 연결된 함수들이 실제 실행된다.
result = chain.invoke({"question": " 회의실 예약 취소는? "})
```

- `Document`는 **데이터 객체**라서 `Document.invoke()`로 일을 시키는 대상이 아니다.
- `RunnableLambda`는 Python 함수를 **실행 단위**로 감싼 객체다.
- `prompt.invoke(...)`는 **메시지 생성**이며 외부 LLM 호출이 아니다.
- `chat_model.invoke(...)`는 **외부 모델 API 호출**이다.
- `|`로 연결한 순간에 DB나 LLM이 실행되는 것은 아니다.

위 코드는 작동 관계를 보여 주는 학습용 발췌이며 실제 실행하려면 `RunnableLambda`를 import하고 `another_step`을 정의해야 한다.

### 8. GPU·CUDA·Batch Size와 실행 비용

Day27 강의에서는 E5와 BGE Cross-Encoder의 로컬 추론에 **NVIDIA CUDA GPU를 기본 실행 환경**으로 사용한다.

- `torch.cuda.is_available()`로 GPU 사용 가능 여부 확인
- `device="cuda:0"`로 첫 번째 CUDA GPU 선택
- `CrossEncoder(..., device="cuda:0")`는 모델 **준비**
- `model.predict(pairs, batch_size=4)`는 질문–문서 쌍의 관련성 **점수 계산 실행**
- **Batch Size**는 모델이 한 번에 처리할 쌍 개수
- **VRAM**은 GPU 메모리
- **OOM**은 메모리가 부족해 실행할 수 없는 상황

GPU가 없거나 오류가 발생하면 수업의 기본 정책은 **조용히 CPU로 전환하지 않고 오류를 표시**하는 것이다. OOM이라면 배치 크기, 후보 수, 입력 길이를 줄이는 방법을 검토한다. 실제 GPU 성능과 처리 시간은 직접 실행해야 알 수 있다.

### 9. Fallback을 숨기면 안 되는 이유

Fallback은 어떤 기능이 실패했을 때 미리 정한 **대체 처리 경로**를 사용하는 것이다.

예를 들어 Reranker가 실패해서 기존 Retriever 순위로 답변을 진행하도록 설계할 수는 있다. 그러나 이 경우 반드시 **"Reranking을 사용하지 않았다"**는 기록이 필요하다. 조용히 기존 순위를 사용하면 성능이 개선된 것처럼 잘못 비교할 수 있다.

수업에서는 **오류 종료** 또는 **기록된 Fallback**을 정책으로 정할 수 있다고 설명한다. 모든 서비스가 무조건 Fallback을 해야 한다는 뜻은 아니다.

### 10. Candidate 품질과 답변의 진실성은 다르게 검사한다

- **Retrieval 품질:** 정답 문서를 후보 안에 찾았는가? 순위가 개선됐는가?
- **Citation:** 답변이 표시한 출처 ID가 실제 Context에 들어 있는가?
- **Grounding:** 답변의 내용이 실제 원문과 일치하는가?

**정답 문서를 상위 3개에 넣었다고 LLM 답변이 자동으로 정확해지는 것은 아니다.** 생성된 문장도 별도 검증이 필요하다.

---

## PART D. Day27 용어 사전 — 원본 강의자료 용어 전부 확인하기

읽는 방법: **용어 → 쉬운 뜻 → 수업에서의 예시**. 아래 표는 원본 파일의 10장 용어집에 나온 항목을 중심으로 쉬운 문장으로 다시 풀었다.

### D-1. 검색 전 질문 변환 용어

| 용어 | 초보자용 쉬운 뜻 | 예시·기억 포인트 |
|---|---|---|
| **RAG** | 관련 근거를 검색해 LLM에게 주고 답변을 만드는 방식 | 질문 → 검색 → 근거 선택 → 답변 |
| **Retrieval** | 관련 문서나 청크를 **찾아오는 작업** | E5+pgvector 검색 |
| **Query Rewrite** | 질문의 뜻은 유지하면서 검색하기 쉬운 문장으로 바꾸기 | "그럼 취소는?" → "회의실 예약 취소 방법은?" |
| **Standalone Query** | 이전 대화를 몰라도 의미가 통하는 독립적인 검색 질문 | "회의실 예약 취소 방법은?" |
| **Conversation History** | 지금 질문을 해석할 때 참고하는 이전 대화 일부 | 직전 "회의실 예약 방법은?" |
| **Coreference** | "그것" 같은 말이 무엇을 가리키는지 연결하는 관계 | "그거" = 회의실 예약 |
| **Ellipsis** | 문장에서 어떤 말이 생략된 현상 | "그럼 취소는?"에서 대상 생략 |
| **Query Expansion** | 비슷한 뜻의 단어·표현을 더해 검색 폭 늘리기 | 예약 취소 + 예약 철회 |
| **Multi-query** | 표현이 다른 질문 여러 개로 검색하고 결과 합치기 | Q1 취소 절차 / Q2 철회 방법 |
| **Semantic Drift** | 질문을 바꾸는 도중 원래 의도가 달라지는 오류 | 요청하지 않은 "무료 취소" 조건 추가 |
| **Query Plan** | 원 질문·검색용 질문·검색 조건을 한데 관리하는 구조 | `original`, `search_query`, `history` |

### D-2. 후보 검색 관련 용어

| 용어 | 초보자용 쉬운 뜻 | 예시·기억 포인트 |
|---|---|---|
| **Candidate Retrieval** | 전체 DB에서 Reranker가 볼 후보를 1차로 찾는 과정 | 최대 10개 후보 검색 |
| **Candidate-k** | 첫 검색에서 가져올 후보의 **최대 개수** | `candidate_k=10` |
| **Top-k** | 상위 k개를 선택하는 방식 | 검색 상위 10개 |
| **Top-N** | 재정렬 후 최종으로 고르는 상위 N개 | Rerank 후 상위 3개 |
| **Context-k** | 모델 Context에 담을 근거 문서의 **최대 개수** | `context_k=3` |
| **Candidate Set** | Reranker가 실제로 평가하는 검색 후보 묶음 | `{C1,C2,C3}` |
| **Context Budget** | LLM에게 제공할 참고 문서의 길이·개수 제한 | 긴 문서는 일부만 선택 |
| **Deduplication** | 같은 검색 결과가 여러 번 나왔을 때 하나로 합침 | C2를 한 번만 남김 |
| **chunk_id** | 특정 Chunk를 구별하는 문자열 식별자 | 다중 검색에서 같은 Chunk 판별 |
| **citation_id** | 실제 답변 출처로 사용할 양의 정수 식별자 | `source_ids=[201]`는 정수 ID 예시 |
| **Metadata Filter** | 어떤 분류·버전·권한 범위의 문서를 검색할지 제한 | `category="시설"` |
| **Version Filter** | 공개된 현재 버전 문서만 검색하도록 제한 | 과거 규정 제외 |
| **Sentence Embedding** | 문장의 의미를 비교하도록 숫자 벡터로 나타낸 것 | E5-small의 384차원 |
| **Bi-Encoder** | 질문과 문서를 따로 숫자 벡터로 바꿔 비교하는 모델 방식 | E5 1차 검색 |

### D-3. Reranking·점수·평가 용어

| 용어 | 초보자용 쉬운 뜻 | 예시·기억 포인트 |
|---|---|---|
| **Reranker** | 이미 찾은 후보 문서의 관련성을 다시 계산해 순서를 바꾸는 기능 | 후보 3위 문서를 1위로 |
| **Cross-Encoder** | 질문과 문서를 **함께 읽고** 관련성 점수를 내는 모델 | `(질문, 문서)` 쌍 → 점수 |
| **Relevance Score** | 모델이 측정한 질문–문서 관련성 점수 | 높더라도 정답 확률은 아님 |
| **Logit** | Sigmoid 같은 변환 함수를 적용하기 전 원시 점수 | 부호·범위는 모델에 따름 |
| **Sigmoid** | 숫자를 0~1 사이 값으로 바꾸는 수학 함수 | 변환했다고 정확한 확률은 아님 |
| **Recall Ceiling** | 정답이 최초 후보에 없으면 재정렬로 찾을 수 없는 한계 | 후보에 C3 없음 → C3 순위 불가 |
| **Precision** | 검색해 가져온 문서 중 실제 관련 문서가 차지하는 비율 | 정식 지표는 후속 수업 |
| **Recall** | 실제 관련 문서 중 검색이 찾아낸 비율 | 놓친 문서가 얼마나 있는지 |
| **Hit@k** | Top-k 안에 관련 문서가 하나라도 있는지 확인 | True / False |
| **MRR** | 관련 문서가 처음 등장한 순위를 이용하는 평가 지표 | 정식 계산은 11/25 수업 |
| **nDCG** | 관련성 등급과 검색 결과 위치를 함께 보는 순위 평가 지표 | 정식 계산은 11/25 수업 |
| **Latency** | 작업 시작부터 결과를 받기까지 걸린 시간 | `rerank_ms`, `total_ms` |
| **Batch Size** | 모델이 한 번에 처리하는 입력 쌍의 수 | `batch_size=4` |
| **Fallback** | 실패했을 때 미리 정한 대체 처리 방식 | Reranker 실패 시 기록된 원 순위 사용 |

### D-4. LangChain·서비스 용어

| 용어 | 초보자용 쉬운 뜻 | 예시·기억 포인트 |
|---|---|---|
| **LangChain** | LLM 앱의 각 처리 기능을 연결하는 Python 라이브러리 | Prompt·검색·모델 호출 연결 |
| **Runnable** | 입력을 받아 실행하고 결과를 반환하는 공통 실행 단위 | Python 함수의 실행 객체 |
| **RunnableLambda** | Python 함수를 Runnable 형태로 감싸는 도구 | `RunnableLambda(rerank_step)` |
| **RunnableSequence** | 여러 Runnable을 왼쪽부터 오른쪽 순서로 연결한 실행 흐름 | `step_a | step_b` |
| **LCEL** | LangChain에서 Runnable을 조합해 표현하는 문법 | `|` 연산자 |
| **invoke()** | Runnable에 입력을 넣어 **실제 실행**하는 메서드 | `pipeline.invoke(request)` |
| **Document** | 본문과 출처 Metadata를 담은 **자료 객체** | `page_content`와 `metadata` |
| **ChatPromptTemplate** | 질문·근거·규칙을 채워 모델 입력 메시지를 만드는 틀 | `{question}`, `{context}` |
| **ChatPromptValue** | Prompt 틀에 값을 넣어 완성한 메시지 묶음 | system/human 메시지 |
| **AIMessage** | Chat Model이 반환한 응답 객체 | 답변 내용·메타데이터 |
| **Structured Output** | 모델이 정해 둔 필드 형태로 결과를 반환하도록 요청하는 방식 | `standalone_query`, `needs_clarification` |
| **State / State dict** | 한 단계의 입력·출력·중간값을 계속 담아 전달하는 Dictionary | `state["candidate_docs"]` |
| **Pipeline** | 여러 처리 단계를 연결한 전체 작업 흐름 | Rewrite → Retrieve → Rerank |
| **Service Layer** | API와 검색/모델 기능 사이에서 실제 업무 흐름을 조립하는 코드 계층 | `app/pipeline.py` |
| **Pydantic** | 요청이나 응답 데이터의 형식과 조건을 검사하는 Python 라이브러리 | `EnhancedRagRequest` |
| **FastAPI** | HTTP 요청을 받고 Python 처리 함수에 연결하는 웹 프레임워크 | `/api/v1/rag/enhanced` |
| **Observability** | 처리 단계가 잘 돌아갔는지 로그·시간·오류를 확인할 수 있는 상태 | Rewrite·Rerank 단계 기록 |
| **Grounding** | 답변의 주장이 실제 참고 문서의 내용과 맞는지 | 15분 규정과 답변 비교 |
| **Citation** | 답변에서 근거로 제시하는 출처 ID·원문 참조 | `source_ids`와 원문 위치 |
| **Agent Tool** | Agent가 필요할 때 호출할 수 있도록 제공하는 기능 | 향후 검색 Tool로 재사용 |

### D-5. GPU 관련 용어

| 용어 | 초보자용 쉬운 뜻 | 예시·기억 포인트 |
|---|---|---|
| **CUDA** | NVIDIA GPU에서 딥러닝 계산을 실행하도록 지원하는 플랫폼 | `device="cuda:0"` |
| **VRAM** | GPU가 계산할 때 사용하는 메모리 | 큰 모델/배치에서 부족할 수 있음 |
| **CUDA Synchronization** | 비동기로 실행되는 GPU 작업이 끝날 때까지 기다리기 | GPU 소요 시간 측정 전후 사용 |

### D-6. 본문을 이해할 때 함께 알아두면 좋은 추가 용어

*아래는 원본 용어집 표 이외에 본문에서 반복 등장하는 중요한 표현을 추가로 쉽게 풀이한 것이다.*

| 용어 | 쉬운 뜻 |
|---|---|
| **History** | 질문을 이해하기 위해 참고하는 이전 대화 기록 |
| **original question / `question`** | 사용자가 실제 입력한 원래 질문 |
| **search_query** | Rewrite로 만든 검색 전용 질문 |
| **search_queries** | Multi-query를 위해 준비한 검색 질문 여러 개 |
| **candidate_docs** | 첫 검색으로 얻은 후보 Document 목록 |
| **ranked_docs** | Reranker가 새 순서로 정렬한 Document 목록 |
| **Context Builder** | LLM에게 실제 넘길 문서를 선택하고 문자열로 만드는 Python 기능 |
| **included_ids** | Context에 실제로 들어간 자료의 정수 출처 ID 목록 |
| **Stage** | Pipeline 중 한 가지 작업을 담당하는 단계 |
| **OOM(Out of Memory)** | GPU 메모리(VRAM) 등 자원이 부족한 오류 |
| **Baseline** | 개선 전 비교 기준으로 삼는 기존 방식의 결과 |
| **Before/After** | 같은 조건에서 개선 전 결과와 개선 후 결과를 비교하는 방식 |
| **needs_clarification** | 이전 대화만으로 질문 대상을 알 수 없어 추가 확인이 필요한 상태 |
| **insufficient_context** | 처리는 정상이나 제공된 근거만으로 답변할 수 없는 상태 |
| **chunk_id vs citation_id** | Chunk 구분용 문자열과 답변 인용용 정수는 목적·형식이 다름 |

---

## PART E. 검색 실패 원인을 어디서 고칠지 판단하기

| 발견한 문제 | 실제 원인이 있을 수 있는 위치 | 먼저 점검할 방법 |
|---|---|---|
| "그럼 취소는?"처럼 대상이 생략돼 있음 | **Query Rewrite** | History에 대상을 확인할 근거가 있는지 |
| "취소"로 검색했는데 문서에는 "철회"만 나옴 | **Query Expansion/Multi-query** | 같은 뜻의 표현을 추가해 검색 |
| 정답 문서가 후보 목록에는 있지만 8위라 Context에서 빠짐 | **Reranking** | 후보를 다시 점수화해 순위 비교 |
| 정답 문서가 1차 후보에 아예 없음 | **Candidate Retrieval / Indexing** | 검색어·필터·후보 K·색인 확인 |
| 옛날 규정이 검색됨 | **Metadata / Version Filter** | 현재 공개 버전과 권한 범위 확인 |
| Reranker가 GPU 메모리 부족으로 실패 | **실행 환경·자원** | VRAM·Batch Size·후보 수 확인 |
| 검색은 성공했는데 모델이 내용을 틀리게 답함 | **Generation / Grounding** | 모델 응답과 원문 직접 대조 |
| DB 연결 자체가 실패 | **Retrieval 서비스 오류** | 자료가 없는 상황으로 오해하지 않기 |

**핵심:** "검색이 잘 안 됐다"고 해서 무조건 Reranking만 추가하면 안 된다. **어느 단계에서 실패했는지 먼저 확인해야 한다.**

---

## PART F. 성능 개선 여부를 제대로 평가하는 방법

### 1. Before/After란?

Before/After는 개선 전과 개선 후를 **가능한 한 같은 조건**에서 비교하는 것이다.

예를 들어 정답이 C3인 가상 질문에서:

| 비교 항목 | 기존 검색(Before) | 개선 검색(After) |
|---|---|---|
| 후보 순위 | C1, C2, C3 | C3, C1, C2 |
| C3 순위 | 3위 | 1위 |
| Top-2에 정답 포함? | 아니오 | 예 |

이 표는 **강의의 고정 예시**다. 실제 성능 향상을 증명하려면 실제 모델·DB로 반복 측정해야 한다.

### 2. 어떻게 비교해야 할까?

1. 테스트 질문과 **관련 문서 ID(정답 레이블)**를 결과를 보기 전에 정한다.
2. 문서 버전·Category·권한 범위를 통일한다.
3. **Rewrite만 적용**, **Rerank만 적용**, **둘 다 적용**을 나눠 비교한다.
4. Reranking만의 효과를 보고 싶다면 **동일한 Candidate Set**에 기존 순위와 새 순위를 적용한다.
5. Top-k 내 관련 문서 포함 여부뿐 아니라 내용·출처·Grounding도 확인한다.
6. 처리 시간(`rewrite_ms`, `retrieve_ms`, `rerank_ms`, `generation_ms`)과 오류 여부도 기록한다.

처리 단계가 늘면 품질이 좋아질 수도 있지만 응답 시간이나 비용도 늘 수 있다. 따라서 "순위가 변했다"만으로 무조건 좋은 시스템이라고 평가하지 않는다.

### 3. 지표는 어디까지 배웠나?

- **Hit@k:** Top-k 안에 관련 문서가 **하나라도 있는가?** → True/False로 확인할 수 있다.
- **Recall@k · MRR · nDCG:** 본격적인 공식과 계산은 원본 자료의 다음 수업(11/25) 범위다.
- **Grounding / Faithfulness:** 답변이 실제 근거를 얼마나 잘 따르는지는 검색 순위 평가와 별도로 확인해야 한다.

---

## PART G. 실습 시 기억할 기술·파일 위치

| 기술/파일 | 담당하는 역할 | 비고 |
|---|---|---|
| `app/query.py` | History 기반 Query Rewrite | 질문 의미 보존·확인 필요 분기 |
| `app/retrieval.py` | E5+pgvector 검색 호출 | Day26 검색 코드 재사용 |
| `app/rerank.py` | Cross-Encoder 점수화·재정렬 | 기존 후보만 재정렬 |
| `app/pipeline.py` | 처리 단계 연결·상태 전달·분기 | 원 질문·검색용 질문 보존 |
| `app/api.py` | FastAPI 요청/응답과 오류 처리 | 입력 오류 vs 서비스 오류 구분 |
| `coursekit.checkpoints.make_retriever` | 기존 검색 코드를 Runnable로 감싸는 프로젝트 함수 | LangChain 내장 함수 **아님** |
| `BAAI/bge-reranker-v2-m3` | 다국어 Cross-Encoder 모델 | CUDA GPU 기본 사용 |
| `intfloat/multilingual-e5-small` | 질문·문서 Embedding 모델 | 384차원, 기존 저장 벡터 유지 |
| PostgreSQL + pgvector | 문서/벡터 저장과 Top-k 검색 | 기존 Category·버전 조건 유지 |

*위 Python 파일 구조는 원본 강의의 교육용 구성 예시이며, 실제 프로젝트에서 파일을 반드시 이렇게 나눠야 하는 것은 아니다.*

### LangChain을 써도 자동으로 해결되지 않는 일

LangChain은 처리 단계를 연결해 주는 도구다. **권한 검사, 검색 필터 적용, 정답 근거 검증, 상태·오류 처리, 원문 출처 연결**은 서비스를 만드는 Python 코드가 책임져야 한다.

### 자주 만나는 상태·오류

| 상태/오류 | 뜻 | 처리 방향 |
|---|---|---|
| `needs_clarification` | 질문의 대상이 모호함 | 사용자에게 질문 확인 요청 |
| `insufficient_context` | 정상 검색했지만 답할 근거가 부족 | 없는 정보를 만들지 않기 |
| `rewrite_failed` | 질문 재작성 모델 호출·결과 처리 실패 | 검색과 분리해 오류 기록 |
| `retrieval_failed` | DB·Retriever 검색 실패 | "문서 없음"으로 숨기지 않기 |
| `rerank_failed` | Cross-Encoder 실행·점수 오류 | 오류 또는 **기록된** Fallback |
| `generation_failed` | LLM API·생성 오류 | 정상 답변으로 승인하지 않기 |
| HTTP 422 | 잘못된 사용자 요청 형식·값 | Pydantic 검증 등 |
| HTTP 502 | 모델/DB 등 외부 처리 실패 | 내부 비밀 정보 노출하지 않기 |

**원본 코드에 관한 주의:** API 예시에서 모든 `ValueError`를 422로 처리하면 Reranker 내부의 점수 오류까지 사용자 입력 오류로 보일 수 있다. 실제 구현에서는 **사용자 입력 오류와 내부 서비스 오류를 별도 예외 타입으로 분리**하는 편이 안전하다.

---

## PART H. 직접 이해한 것을 절차적으로 작성해 보는 복습

이 부분은 설명을 읽은 뒤 **파일을 덮고 네 말로 다시 쓰기 위한 공간**이다.

### 1. 빈칸 없이 순서 설명하기

```text
1. 사용자가 질문과 이전 대화 기록을 서버에 보낸다.
2. FastAPI / Pydantic은 _______________________________________________.
3. Query Rewrite 단계는 ________________________________________________.
4. (선택) Expansion / Multi-query 단계는 ________________________________.
5. Retriever는 ________________________________________________________.
6. Cross-Encoder Reranking은 __________________________________________.
7. Context Builder는 _________________________________________________ .
8. ChatPromptTemplate은 ________________________________________________.
9. Chat Model은 ________________________________________________________.
10. Validation과 FastAPI는 _____________________________________________ .
```

### 2. 핵심 확인 질문

1. 이번 수업에서는 왜 **원 질문**과 **검색용 질문**을 따로 보관할까?
2. Query Rewrite와 Reranking은 각각 Retriever의 **앞/뒤 중 어디**에 위치할까?
3. 이전 대화 없이 "그거 취소는?"이라는 질문을 받으면 어떻게 해야 할까?
4. Expansion과 Multi-query의 차이를 한 문장씩 설명해 봐.
5. `candidate_k=10`, `context_k=3`은 각각 무엇의 최대 개수일까?
6. E5와 Cross-Encoder의 **입력·출력**은 각각 어떻게 다를까?
7. 검색된 후보에 정답 문서가 없는데 Reranker가 그 문서를 찾을 수 있을까?
8. Cross-Encoder 점수가 0.9이면 정답이 90% 확실하다는 뜻일까?
9. 모델에게 전달한 Context에는 C3만 들어 있는데 다른 후보의 출처를 인용해도 될까?
10. `RunnableLambda(...)`와 `a | b`를 작성한 순간 실행이 끝난 것일까?
11. Reranker가 GPU 오류로 실패했다면 왜 조용한 Fallback을 피해야 할까?
12. 후보 순위가 좋아져도 Grounding 검사가 필요한 이유는 무엇일까?
13. Retrieval 결과가 0개인 경우와 DB 연결 오류는 어떻게 다를까?
14. Rewrite와 Reranking의 효과를 따로 비교하려면 어떤 조건을 고정해야 할까?

### 3. 반드시 기억할 최종 요약

```text
질문 + History
   ↓
Query Rewrite           — 질문의 생략된 내용을 복원
   ↓
Expansion / Multi-query — 필요할 때 다른 표현으로 후보 넓히기
   ↓
Retriever / E5 / pgvector — 관련 문서의 1차 후보 검색
   ↓
Cross-Encoder Reranking — 후보들을 질문과 다시 비교해 순위 변경
   ↓
Context Builder         — 실제 답변에 사용할 일부 문서와 출처 ID 선택
   ↓
Prompt                  — 원 질문 + 규칙 + 선택한 근거
   ↓
Chat Model              — 답변 생성
   ↓
Validation + Sources    — 형식·출처·원문 근거 검사
   ↓
FastAPI                 — 검증된 답변 반환
```

> **이번 수업의 핵심:** **"찾기 전에 질문을 바로잡고, 찾은 뒤에는 후보 순서를 바로잡는다."** 다만 후보 자체가 없거나 출처 내용이 잘못됐으면 그 문제는 다른 단계에서 해결해야 한다.

---

### 원본 자료에 관한 메모

- 정리는 업로드된 `Day27_LangChainRAG_Reranking_정리.md`의 개념, 8개 챕터 및 용어집을 기준으로 재구성했다.
- 강의의 데모 8종은 브라우저 안에서 고정 예시를 보여 주는 방식이며 실제 모델 점수·검색 순위·실행 시간의 관측 결과가 아니다.
- 원본 자료의 용어집 표 항목을 바탕으로 기본 용어를 정리했으며, 별도로 본문에 반복 등장하는 실습 변수를 추가했다.
- 후속 강의인 11/25 평가(Recall@k·MRR·nDCG)와 11/26 Grounding·회귀 테스트는 **이번 자료에서 다음 학습 주제로 소개한 범위**까지만 설명했다.
