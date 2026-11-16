# Day21 · Prompt·Context Engineering — 용어·개념 정리본

> 원본 강의 정리 문서를 **용어 사전 → 핵심 개념 → 검증 파이프라인 → 헷갈리는 구분 → 최소 코드 모음** 순으로 재구성했습니다.

---

## 0. 한눈에 보는 전체 흐름

```text
[Prompt 설계] → [Context 선택·직렬화] → (모델 호출) → [출력 검증 4단계] → [재시도/중단] → [A/B 평가·기록]
   1·2장            4장                                   6장              7장            5장·8장
                                   3장: 추론 근거(정보 부족 판단 기준)
```

| 장 | 핵심 질문 | 한 줄 요약 |
|---|---|---|
| 1장 | 무엇을 좋은 답이라 부를까? | 작업·범위·실패 조건을 명확히 지시 |
| 2장 | 무엇을 바꾸고 무엇을 고정할까? | 예제 분리, 한 번에 한 가지만 변경·기록 |
| 3장 | 어디서 정보 부족을 확인할까? | 자료로 확인 가능한 것과 부족한 것을 구분 |
| 4장 | 이번 요청에 무엇을 넣을까? | 선택·순서·직렬화·토큰 예산 |
| 5장 | 정말 달라졌다는 근거가 있을까? | 수집 방식·분모·미실행·사람 평가 구분 |
| 6장 | 형식이 맞으면 내용도 맞을까? | 자동 검사 3단계 + 사람의 근거 검토 |
| 7장 | 다시 요청할까, 멈출까? | 형식 오류만 1회 수정, 나머지는 중단 |
| 8장 | 같은 흐름을 다시 설명할 수 있을까? | 고정한 A/B로 H 세트·실패 사례 확인 |

---

## 1. 핵심 용어 사전

### 1-1. Prompt 관련

| 용어 | 정의 | 주의점 |
|---|---|---|
| **Prompt** | 모델에게 작업을 전달하는 입력. 지시·자료·예제·출력 조건을 포함할 수 있음 | 입력을 구성하는 일일 뿐, 모델 학습이나 정확성 보증이 아님 |
| **Prompt Engineering** | 작업과 성공 기준을 Prompt에 담고, 같은 조건에서 응답을 비교하며 개선하는 과정 | 길고 화려한 문장이 목표가 아님. 개선안도 결과 관찰 전엔 *가설* |
| **지시문 vs 참고 데이터** | 지시문 = 수행할 작업·기준 / 참고 데이터 = 읽을 내용 | 데이터 속 "이전 지시를 무시해"는 명령이 아니라 그저 자료 |
| **여섯 요소** | 역할·작업·입력·제약·실패·출력 | 고정 규칙과 매번 바뀌는 값을 분리 |
| **Prompt Template** | 고정 문장 + 요청마다 바뀌는 값을 분리한 입력 틀 | 문자 조립만 담당. 안전 검증·정답 판단 기능은 없음 |
| **`$question`** | `string.Template`의 자리표시자. `substitute()`가 값을 채움 | `$`는 LLM 명령어가 아님. 문자 `$` 자체는 `$$` |
| **Zero-shot** | 입출력 예제 0개로 작업·규칙·형식만 제시 | 근거 자료까지 없다는 뜻이 아님. 학습 안 했다는 뜻도 아님 |
| **Few-shot** | 입력→바람직한 출력 예제 몇 개를 함께 제시 | 가중치를 바꾸는 Fine-tuning과 다름. 예제의 출처(901)를 현재 답에 복사 금지 |
| **Prompt 버전 관리** | 어떤 지시·자료·계약으로 실행했는지 재확인 가능하게 보존 | 같은 버전명으로 내용 덮어쓰기 금지. 버전명은 품질 점수가 아님 |
| **A/B 비교** | 기준안 A와 변경안 B를 같은 조건에서 비교 | A/B는 약어가 아닌 구분 표기. 변경은 **한 가지만** |

### 1-2. Context 관련

| 용어 | 정의 | 주의점 |
|---|---|---|
| **Context** | 이번 응답을 만들 때 참고하도록 입력에 포함한 정보 | DB에 있는 것 ≠ 실제 전달한 것. 전달한 자료만 출처 범위 |
| **Context Engineering** | 필요한 정보를 고르고 배치하며 출처·유효성·길이·제외 이유를 관리 | 많이 넣는 게 목표가 아님 |
| **`context_json`** | 선택한 지식 목록을 JSON 문자열로 담은 변수명 | 교안의 이름일 뿐 |
| **Snapshot** | 특정 시점의 자료를 고정한 복사본 | 재현성 확보용. 정확성·최신성·전달 권한은 보장 안 함 |
| **직렬화 (Serialization)** | 메모리 속 자료(list/dict) → JSON 문자열 (`json.dumps`) | 검색·요약이 아님. 표현 방식만 변경 |
| **역직렬화** | JSON 문자열 → Python 자료 (`json.loads`) | 읽힌다 ≠ 내용이 맞다 |
| **Token** | 모델이 텍스트를 처리하는 단위 | 글자 수·단어 수와 1:1 아님. tokenizer로 확인 |
| **Context Window** | 한 번의 생성에서 처리 가능한 토큰 한도 | 입력뿐 아니라 **출력 공간**도 포함해 계산 |
| **Hash (SHA-256)** | 같은 내용인지 식별하는 요약값 | 정확성·권한 보증 아님. 원문·버전도 함께 보존 |

### 1-3. 추론·평가 관련

| 용어 | 정의 | 주의점 |
|---|---|---|
| **CoT (Chain-of-Thought)** | 중간 풀이가 포함된 예제를 제시하는 접근 | 긴 풀이 ≠ 정답. 내부 사고 공개 요구 대신 **결과와 근거를 검증** |
| **작업 분해** | 큰 요청을 입·출력이 분명한 작은 작업으로 나눔 | 반드시 여러 모델/Agent를 호출한다는 뜻 아님 |
| **Fixture** | 검사기 동작 확인용으로 미리 준비한 고정 입력(교사 작성 샘플) | 검사기 동작 확인용이지 모델 품질 지표가 아님 |
| **Evaluation set** | 비교할 질문 묶음 | 질문별 전달 문맥도 고정 |
| **Development(D) / Holdout(H)** | D = 개선용 질문 / H = 최종 확인용 질문 | A/B(Prompt 버전 축)와 **다른 축** |
| **source_mode** | 응답 수집 방식 기록 (`manual_chat`, `fixture` 등) | 작성 샘플과 실제 관찰을 섞지 않음 |
| **N / 분모** | 한 버전에서 비교 대상으로 정한 첫 응답 수 | 미실행·미평가 건수를 숨기지 않고 함께 표시 |

### 1-4. 출력 검증 관련

| 용어 | 정의 | 주의점 |
|---|---|---|
| **JSON** | 정해진 문법의 데이터 텍스트 형식 | 문법 통과 ≠ 사실 확인. `101`(숫자)과 `"101"`(문자열)은 다름 |
| **출력 계약 (Output Contract)** | 응답이 따라야 할 필드·자료형·상태별 규칙 | 교안이 정한 약속이지 모든 LLM의 고정 형식이 아님 |
| **JSON Schema** | 허용할 JSON 구조를 기술한 명세 (`properties`, `required`, `additionalProperties`) | 데이터가 아니라 *규칙*. 사실 여부는 판단 못 함 |
| **Pydantic** | 파이썬 자료형 선언으로 데이터 모델을 만들고 검사하는 라이브러리 | 여기서 "모델"은 LLM이 아니라 데이터 구조 |
| **Validation (검증)** | 받은 값이 미리 정한 조건을 충족하는지 검사 | 하나의 "통과"가 모든 정확성을 뜻하지 않음 |
| **출처 vs 근거 적합성** | 출처 = 인용한 자료 ID / 근거 적합성 = 그 자료가 답을 실제로 뒷받침하는가 | ID가 존재해도 틀린 인용일 수 있음 |
| **Retry (재시도)** | 실패 원인 확인 후 정해진 조건·횟수 안에서 재수행 | 모든 실패를 반복하지 않음 |

---

## 2. 장별 핵심 개념

### 1장. Prompt Engineering 개요

- **좋은 Prompt = 구체적인 작업 + 대상 + 범위 + 결과 형식.** "친절하게 답해"보다 "제공 지식으로 2문장 이내, 출처 표시, 없으면 부족하다고 답변"이 낫다.
- **성공 기준을 먼저 정의**한다. 위치 질문 → 근거 있는 위치 / 기간 정보가 없는 질문 → "정보 부족"이 올바른 응답.
- **지식 ID(101 등)는 자료를 찾기 위한 이름**이지 점수나 모델 요구 숫자가 아니다.
- **지시와 데이터의 분리**: 앱이 정하는 것(범위·출력 계약·부족 처리·중단 기준) vs 외부에서 받은 것(질문·문서 본문). 구분자·제목(`[규칙]`)은 표현일 뿐 **보안 장벽이 아니다.**

```text
[작업] 제공된 자료로 질문에 답하세요.
[질문] 파일은 어디에 보관하나요?
[자료] 101: 실습 파일은 개인 폴더에 보관합니다.
[조건] 자료에 없는 사실은 추측하지 마세요.
```

**여섯 요소 조립**

```python
parts = {
    "역할": "초보 수강생의 실습 도움말 작성자",
    "작업": "질문에 한국어 2문장 이내로 답하기",
    "입력": "사용자 질문과 선택된 지식",
    "제약": "제공한 근거 범위에서만 답하기",
    "실패": "없는 사실은 추측하지 않고 정보 부족 표시",
    "출력": "status, answer, source_ids, missing_information의 JSON",
}
instruction = "\n\n".join(f"[{k}]\n{v}" for k, v in parts.items())
```

**Template 사용**

```python
from string import Template
T = Template("[질문]\n$question\n[지식 데이터]\n$context_json\n[출력]\n$contract\n")
prompt = T.substitute(question=question, context_json=context_json, contract=contract_text)
# 값이 빠지면 KeyError. JSON의 { } 는 자리표시자가 아님.
```

### 2장. Prompt 종류와 관리

| 구분 | Zero-shot | Few-shot |
|---|---|---|
| 예제 | 없음 | 입력→출력 예제 몇 개 |
| 제공하는 것 | 지시 + 자료 + 질문 + 출력 형식 | 위 내용 + 예제 구역 |
| 위험 | 형식이 흔들릴 수 있음 | 예제 내용(901)을 현재 답에 복사 |

- 예제용 ID(901)는 현재 지식(101~103)과 혼동되지 않도록 **일부러 다른 번호**를 쓴다. → **예제에서 가져올 것은 형식이지 출처가 아니다.**
- **버전은 대상별로 따로 관리**: `answer_v1/v2`(Prompt), `knowledge_v1`(지식 스냅샷), `answer_schema_v1`(출력 계약), `dev_v1`(평가 질문 묶음). 이름만이 아니라 **원문·변경 이유·가설**도 함께 기록한다.
- **A/B 원칙**: 질문·문맥·모델·계약·예제·평가 기준은 고정, 바꾸는 것은 **한 가지**.

```python
experiment = {
    "base_version": "answer_v1",
    "changed": "근거 부족 시 응답 규칙 1개",
    "hypothesis": "근거 없는 단정이 줄어든다",   # 아직 '가설'
    "context_version": "knowledge_v1",
    "schema_version": "answer_schema_v1",
    "eval_set": "dev_v1",
}
```

### 3장. Prompt 추론 근거

- 확인할 수 있는 것: **인용한 자료 본문에 필요한 정보가 있는가 / 최종 답이 자료와 일치하는가**
- 증거가 되지 않는 것: "충분히 생각했습니다", 매우 긴 풀이 문장
- **작업 분해**: `질문 파악 → 근거 대조 → 답변·상태 결정 → 외부 검사`
  - 코드가 확인할 부분(ID·자료형) vs 사람이 읽을 부분(본문이 답을 뒷받침하는지)을 구분
- 질문에 필요한 정보가 **둘이면 둘 다 대조**. 일부만 있다고 전체 답이 가능한 것이 아님.
- **정보 부족은 정상 결과**일 수 있다.
  - `status = answered` → 답변 가능
  - `status = insufficient_context` → 제공 문맥에 정보 부족

### 4장. Context Engineering

**선택 기준 4가지**: 관련 있는가 / 보내도 되는가(권한) / 유효한 자료인가 / 출력 공간을 남기고 들어가는가
→ **넣지 않은 정보와 제외 이유도 기록**한다.

**저장 → 입력까지의 흐름**

```text
DB 저장 지식(101·102·103) → 스냅샷 고정(knowledge_v1) → 사람이 ID 선택 → 직렬화(context_json) → Prompt에 포함
```

```python
from copy import deepcopy
snapshot = deepcopy(rows)      # 원본 수정과 분리

import json
context_json = json.dumps(selected_rows, ensure_ascii=False)  # list → str
restored = json.loads(context_json)                           # str → list
```

**`build_context` 설계 포인트**

| 상황 | 처리 |
|---|---|
| 원본 ID 중복 | `ValueError` (덮어쓰기 방지) |
| 선택 ID 중복 | 처음 나온 순서 유지하며 중복 제거 |
| 존재하지 않는 ID | 조용히 빼지 않고 `ValueError` |
| 순서 | 정렬하지 않고 **선택한 순서** 유지 |

**Token 예산 (가상 값 — 실제 모델 한도 아님)**

```text
8,192(전체) − 800(지시·질문) − 600(예제) − 1,600(출력용) − 256(여유) = 4,936(근거에 남은 공간)
```

- 출력용 공간과 여유를 **먼저 남기고** 근거 공간을 계산.
- 예제를 1,000 늘리면 근거 공간도 1,000 줄어든다.
- 길이 안에 들어간다고 내용이 정확한 것은 아님. 실제 사용 시 해당 모델의 tokenizer로 확인.

### 5장. 응답 비교 및 평가

- **Fixture(작성 샘플) ≠ 관찰(실제 응답)** — `source_mode`로 구분 기록.
  - `{"status":` → `json_error` / `{}` → `schema_error`
- **평가 기준 예시**
  - 질문 "어디에 보관?" → "개인 폴더" + 101 인용
  - 질문 "몇 년 보관?" → 기간 정보 부족 알림
  - 답할 자료가 있는데 무조건 포기하는 것도 **적절한 응답이 아님**
- **D/H 분리 이유**: 같은 질문을 보며 고친 Prompt는 그 질문에만 맞을 수 있다. 개선용(D)과 최종 확인용(H)을 나눈다.
- **분모 투명성**

```python
planned = 3        # 비교 대상 첫 응답 수 (N)
executed = 2       # 실제 응답 수
json_passed = 2
human_judged = 1
human_ok = 1

json_rate  = json_passed / planned                          # 2/3
human_rate = human_ok / human_judged if human_judged else None
# 근거 적합률 1/1 은 전체 3건의 최종 성공률이 아님!
```

- **평가 필드 값 읽기**: `True` = 통과/적합, `False` = 실패/부적합, **`None` = 미평가·미수행** (False와 다름)
- 실행 기록에 남길 것: `case_id`, `variant`, `prompt_version`, `context_version`, `schema_version`, `source_mode`, `model_label`, `attempt`, `raw_text`, `json_ok`, `schema_ok`, `reference_ok`, `evidence_ok`
- 금지: API 키·비밀번호·실제 개인정보 기록, 관찰하지 않은 응답을 가짜로 채우기

### 6장. 출력 검증

**출력 계약 (4필드)**

| 필드 | 의미 |
|---|---|
| `status` | `answered` 또는 `insufficient_context` |
| `answer` | 사용자에게 보여줄 안내문 (1~500자) |
| `source_ids` | 인용한 자료 ID 목록 (양의 정수, 최대 3개, 중복 금지) |
| `missing_information` | 답하려면 더 필요한 정보 (최대 200자) |

**상태별 조합 규칙**

| status | source_ids | missing_information | answer |
|---|---|---|---|
| `answered` | 비어 있지 않음 | 빈 문자열 | 공백 불가 |
| `insufficient_context` | `[]` | 필요한 정보명 | `"제공된 지식으로는 답할 수 없습니다."` 고정 |

**JSON Schema 3요소**: `properties`(필드 정의) / `required`(필수 지정 — properties에 쓴다고 자동 필수 아님) / `additionalProperties: false`(추가 필드 금지)

**Pydantic 모델**

```python
from typing import Literal, Annotated
from pydantic import BaseModel, ConfigDict, Field

SourceID = Annotated[int, Field(strict=True, gt=0)]

class KnowledgeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "insufficient_context"]
    answer: str = Field(min_length=1, max_length=500)
    source_ids: list[SourceID] = Field(max_length=3)
    missing_information: str = Field(max_length=200)

schema = KnowledgeAnswer.model_json_schema()       # 명세 생성
obj = KnowledgeAnswer.model_validate(data)         # 데이터 검사
```

- `strict=True`: 자동 형 변환 제한 → `[101]` 통과 / `["101"]`, `[True]`, `[101.0]`, `[0]` 실패
- `extra="forbid"`: 선언하지 않은 필드 거부
- `[999]`는 구조상 통과 가능 → **`provided_ids` 확인은 별도**

**4단계 검증 파이프라인** (앞 단계 실패 시 뒤 단계는 **미수행**)

| 단계 | 확인 내용 | 오류 코드 | 수행 주체 |
|---|---|---|---|
| ① 문법 | JSON으로 읽히는가 | `json_error` | 자동 |
| ② 구조 | 필드·자료형이 계약에 맞는가 | `schema_error` | 자동 |
| ③ 참조·상태 | 허용 ID 범위·상태 조합이 맞는가 | `duplicate_source`, `unknown_source`, `status_rule`, `blank_answer` | 자동 |
| ④ 근거 | 인용 본문이 실제 답을 뒷받침하는가 | (`evidence_ok`) | **사람** |

```python
def validate_answer(raw, provided_ids, load_json):
    obj, error = parse_answer(raw, load_json=load_json)   # ①②
    if error: return None, error
    error = final_rules(obj, provided_ids)                # ③
    if error: return None, error
    return obj, None   # 자동 검사 통과일 뿐, evidence_ok를 True로 만들지 않음
```

**엄격 JSON 파서 (선택)**: `NaN/Infinity` 거부(`parse_constant`), 중복 키 거부(`object_pairs_hook`)

**틀린 인용 예시**

```text
질문: 서버는 어디에서 실행하나요?
답:   프로젝트 폴더에서 실행합니다.   출처: [101]
101 본문: 실습 파일은 개인 폴더에 보관합니다.
→ ID는 존재하지만 이 답의 근거는 아님 (③ 통과, ④ 실패)
```

**요청 vs 검사 vs 생성 제약**은 서로 다른 기능이다 — Prompt의 형식 요청 / 로컬 검증(Pydantic) / API의 생성 제약. 같은 것으로 설명하지 않는다.

### 7장. 재시도·중단

| 오류 유형 | 처리 |
|---|---|
| `json_error` / `schema_error` | 같은 조건으로 **형식 수정 1회** |
| 출처·상태 규칙 위반 | **중단**, 입력·규칙 점검 |
| 정상적인 정보 부족 | 생성 반복 종료, 적절성은 별도 검토 |

- **정책**: 최초 1회 + 형식 수정 1회 = **최대 2회** (이 교안의 정책이며 보편 정답 아님)
- 깨진 JSON은 형식 수정이 가능하지만, **없는 근거는 반복해도 생기지 않는다.**
- 수정 요청에는 **확인된 오류의 안전한 요약만** 전달 (traceback·경로·비밀 값 금지). "부족한 사실은 새로 만들지 마세요" 명시.
- 두 번째도 실패하면 **중단하고 실패 기록 반환** — 검증 안 된 raw를 성공처럼 반환하지 않는다.
- **첫 실패는 지우지 않는다**: attempt 1(실패)과 attempt 2(성공)를 별도 기록.

```python
def run_with_retry(generate, prompt, validate):
    history = []
    for attempt in (1, 2):
        raw = generate(prompt)
        obj, error = validate(raw)
        history.append({"attempt": attempt, "error": error})
        if error is None:
            return obj, history
        if error not in {"json_error", "schema_error"}:
            break                      # 규칙 위반은 자동 재생성 안 함
        if attempt == 2:
            break
        prompt = repair_prompt(prompt, error)
    return None, history
```

### 8장. 전체 연결

- 공통 규칙은 `T_full` 템플릿에 두고, **`evidence_rule`만 A/B 변경 대상**으로 하여 나머지는 동일하게 유지.
- 평가 포인트: *B의 점수가 높은가*보다 **비교 조건·검증 결과·중단 기준을 정확히 설명하는가**.
- 오늘의 계약·검사 흐름은 이후 **API · RAG · Agent**로 이어진다. (자동 검색·장기 Agent 메모리는 후속 범위)

---

## 3. 헷갈리는 개념 비교표

| 비교 | 차이 |
|---|---|
| Prompt Engineering vs Fine-tuning | 입력 구성·개선 vs 모델 가중치 학습 |
| Zero-shot의 "Zero" | 예제 0개 (근거·규칙은 제공) |
| Few-shot 예제 vs 현재 근거 | 형식 참고용(901) vs 이번 답의 출처(101) |
| 지시문 vs 참고 데이터 | 따를 규칙 vs 읽을 자료 (데이터 안 명령은 실행 대상 아님) |
| DB의 지식 vs 전달한 문맥 | 저장된 전체 vs 이번에 선택·전달한 것만 출처 범위 |
| 직렬화 vs 검색/요약 | 표현 방식 변환 vs 내용 선택·축약 |
| 글자 수 vs Token 수 | 1:1 아님. tokenizer로 측정 |
| JSON 문법 통과 vs Schema 통과 | 읽을 수 있음 vs 정해진 구조 |
| Schema 통과 vs 근거 적합 | 구조가 맞음 vs 자료가 답을 뒷받침 |
| 출처 ID 존재 vs 근거 적합성 | 허용 범위 안 vs 실제 주장 뒷받침 |
| `101` vs `"101"` vs `true` | 정수 vs 문자열 vs 불리언 (strict에서 모두 다름) |
| `None` vs `False` | 미평가·미수행 vs 실패 |
| A/B vs D/H | Prompt 버전 축 vs 질문 묶음(개선용/최종확인용) 축 |
| Fixture vs 실제 관찰 | 작성 샘플 vs 모델에서 얻은 응답 |
| 자동 통과율 N vs 사람 평가 분모 | 비교 대상 전체 vs 실제 검토한 수 |
| 모델 호출 실패 재시도 vs 형식 수정 재시도 | 통신 오류 대응 vs 출력 형식 복구 (이번 정책은 후자) |
| 요청(Prompt) vs 검증(Pydantic) vs 생성 제약(API) | 부탁 / 사후 검사 / 생성 시점 제한 — 서로 대체 불가 |

---

## 4. 시험·복습용 체크리스트

- [ ] Prompt를 작성하는 것이 왜 "정확성 보증"이 아닌지 설명할 수 있다
- [ ] 지시문과 참고 데이터의 차이, 데이터 속 명령을 따르지 않는 이유를 안다
- [ ] Zero-shot / Few-shot의 차이와 "예제 ID를 복사하지 않는 이유"를 안다
- [ ] A/B 비교에서 왜 **한 가지만** 바꾸는지 안다
- [ ] 스냅샷·직렬화·역직렬화의 의미와 역할을 구분한다
- [ ] 토큰 예산 계산(전체 − 지시 − 예제 − 출력 − 여유)을 직접 할 수 있다
- [ ] 개선용(D)과 최종 확인용(H) 질문을 나누는 이유를 안다
- [ ] 미실행·미평가를 분모와 함께 보여야 하는 이유를 안다
- [ ] JSON → Schema → 참조·상태 규칙 → 근거 검토 4단계를 순서대로 설명한다
- [ ] 출처 ID가 맞아도 근거가 틀릴 수 있는 예를 들 수 있다
- [ ] `answered` / `insufficient_context` 각각의 필드 조합 규칙을 안다
- [ ] 재시도 대상(`json_error`, `schema_error`)과 중단 대상을 구분한다
- [ ] 수정 후 통과해도 첫 실패 기록을 보존하는 이유를 안다

---

## 5. 장별 핵심 키워드 요약

| 장 | 키워드 |
|---|---|
| 1장 Prompt 개요 | Prompt, Prompt Engineering, 역할·작업·제약·출력 |
| 2장 종류와 관리 | Zero/Few-shot, Template, 버전 관리 |
| 3장 추론 근거 | CoT, 필요한 정보, 근거 대조, 정보 부족 |
| 4장 Context | Context Engineering, 선택·직렬화, Token 예산 |
| 5장 비교·평가 | 고정 조건, 평가 세트, 실행 기록, 지표 |
| 6장 출력 검증 | JSON, JSON Schema, Pydantic, 출처·근거 검증 |
| 7장 재시도·중단 | 오류 분류, 형식 수정, 횟수 제한, 중단 조건 |
| 8장 전체 연결 | Prompt, Context, 검증·평가, 일일프로젝트 |

---

> **한 문장 요약**: *지시(Prompt)와 자료(Context)를 분리해 설계하고, 응답은 형식 → 구조 → 참조 → 근거 순으로 검증하며, 비교·재시도의 모든 조건과 실패 기록을 투명하게 남긴다.*
