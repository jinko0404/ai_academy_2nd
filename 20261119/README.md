# Day 24 — Parsing · Chunking · Indexing 초보자용 통합 학습 정리

> **기준 자료:** 사용자가 제공한 `Day24_Parsing_Chunking_Indexing_정리(1).md` (2026-11-19 수업 정리)
>
> **정리 방식:** 전체 목적 → 실제 실행 순서 → 각 단계의 담당 코드·입력·출력·이유 → 용어 사전 → 복습
>
> **범위:** 문서를 수집해서 검색 가능한 상태로 준비하고, 실제 검색까지 확인한다. **LLM을 호출해 최종 답변을 만드는 Basic RAG는 다음 차시의 내용이다.**
>
> **읽을 때 유의:** 아래의 문장·문서·점수 예시는 원리를 설명하기 위한 교육용 가상 예시다. 실제 DB 실행 성공이나 검색 성능을 측정했다는 뜻이 아니다.

---

# PART A. 오늘 뭘 만들려는 걸까?

## A-1. 오늘의 목표를 한 문장으로

**컴퓨터에 보관된 여러 형식의 문서(PDF·TXT·Markdown·HTML)를, AI가 관련 내용을 검색할 수 있도록 손질하고 작게 나눈 뒤 벡터와 출처를 붙여 PostgreSQL에 저장하는 것.**

어제(Day23)는 **이미 정리된 짧은 지식 문장**을 Embedding하고 pgvector로 찾는 방법을 배웠다. 오늘(Day24)은 **원본 파일에서 그 검색용 문장을 어떻게 안전하게 만들어 내는지** 배운다.

- **Day23의 핵심 질문:** 이미 있는 문장 중 관련 문장을 어떻게 찾지?
- **Day24의 핵심 질문:** PDF나 HTML 같은 실제 파일을 어떻게 검색 가능한 문서 조각으로 만들지?
- **다음 차시의 핵심 질문:** 찾아온 문서 조각을 LLM에게 주고 어떻게 근거 있는 답변을 만들지?

## A-2. 가장 중요한 구분: 두 가지 실행 경로

### 경로 ① 문서를 미리 준비하는 작업 — **Ingestion / Indexing**

이 작업은 **사용자가 질문하기 전에** 문서를 등록하거나 수정할 때 실행한다.

```text
[관리자가 검색할 문서를 등록함]
          ↓
① Collect         파일을 수집하고 원본을 보존
          ↓
② Parse           파일 구조를 읽어 텍스트와 위치 추출
          ↓
③ Clean           의미를 유지하며 공백·줄바꿈 정리
          ↓
④ Chunk           검색하기 좋은 크기로 분할
          ↓
⑤ Validate        내용·위치·길이·출처·누락 검사
          ↓
⑥ Embed           조각마다 384차원 벡터 생성
          ↓
⑦ Index/Publish   조각·벡터·출처·버전을 DB에 저장하고 공개
          ↓
[검색 가능한 문서 준비 완료]
```

### 경로 ② 사용자가 질문하는 작업 — **Retrieval**

이 작업은 질문이 들어올 때마다 실행한다. Day23에서 배웠던 검색을 **오늘 만든 Chunk DB**에 적용한다.

```text
[사용자가 질문함]
          ↓
FastAPI가 질문을 받고 검색 함수를 호출  ← 앞 수업과 연결한 예시 구조
          ↓
질문 검사
          ↓
질문을 `query: ` 형태로 E5 Embedding
          ↓
PostgreSQL: 공개된 버전·분류로 검색 대상 제한
          ↓
pgvector 거리 계산 → 가까운 Chunk 최대 k개 선택
          ↓
검색 함수가 Chunk 본문 + 출처 + ID를 반환
          ↓
[나중에 Context → Prompt → LLM으로 연결]
```

**핵심:** 문서의 벡터는 사전에 만들어 저장하고, 질문 벡터는 질문이 들어올 때 만든다. **매 질문마다 원본 PDF 전체를 다시 읽고 나누는 게 아니다.**

## A-3. 누가 무엇을 실행하는지 구분하기

| 이름 | 정체 | 실제 하는 일 | 주의 |
|---|---|---|---|
| 사용자 | 질문하는 사람 | 질문 입력 | 수업의 문서 준비 작업은 사용자의 질문보다 먼저 수행 |
| 문서 등록 프로그램 | Python으로 작성한 처리 코드 | 파일 수집부터 저장까지 함수들을 순서대로 호출 | 독립된 사람이 아니라 코드 흐름 |
| Parser | 파일 형식별 읽기 기능 | PDF 페이지, HTML 요소, MD 제목 등 추출 | ‘Parsing’은 **작업**, ‘Parser’는 **도구/코드** |
| Cleaner | 정제 함수·모듈 | 공백·줄바꿈 등 정리 | 원문 의미를 바꾸면 안 됨 |
| Chunker | 문서 분할 함수·모듈 | 본문을 여러 Chunk로 나눔 | ‘Chunking’은 작업, ‘Chunker’는 수행 코드 |
| Validator | 검증 함수·모듈 | Chunk 위치·길이·누락 검사 | 실패 자료를 정상 검색 대상으로 넣지 않음 |
| Embedding 모델(E5) | 사전학습 모델 | 텍스트를 384차원 벡터로 변환 | DB 저장 기능은 없음 |
| PostgreSQL | 데이터베이스 | 원문 정보·청크·벡터·버전 저장, SQL 실행 | pgvector와 별도 데이터베이스가 아님 |
| pgvector | PostgreSQL 확장 | 벡터 타입과 거리 연산 제공 | 문서를 읽거나 문장을 직접 임베딩하지 않음 |
| 검색 함수 / Retriever | Python 검색 기능 | 질문 검사 → 벡터 생성 요청 → SQL 조회 → 결과 정리 | ‘Retrieval’은 검색 **과정** |
| FastAPI | 웹 API 프레임워크 | 요청을 Python 처리 함수에 연결 | Day24 핵심 과제 자체는 문서 준비·검색 확인이며, 이 연결은 앞 수업의 이해를 위한 예시 |

---

# PART B. 원본 파일 하나가 검색 가능한 자료가 되는 전체 절차

## 공통 예시: `attendance.md`라는 출석 안내 파일

예시 파일을 아래처럼 가정하자.

```markdown
# 출석 안내

## 지각과 결석
수업 시작 후 10분을 초과하면 결석으로 기록합니다.

## 문의
출석 관련 문의는 운영실에 문의하세요.
```

실제 수업 자료에는 출석 안내 Markdown 문서가 `DOC-ATTEND`라는 ID로 등장한다. 위 내용은 설명용 문장이다. 이제 **같은 파일을 끝까지 추적**한다.

## STEP 0. 문서 처리 시작 전에 준비할 것

**담당:** 개발자가 만든 문서 등록 프로그램과 설정(Config)

- 어떤 폴더의 파일을 처리할지 정한다.
- 지원할 형식(`.txt`, `.md`, `.html`, `.pdf`)을 정한다.
- Embedding 모델(E5-small), Chunk Size, Overlap, Pipeline 버전을 정한다.
- PostgreSQL 연결과 사용할 실습 전용 스키마를 준비한다.

**왜?** 어떤 데이터를, 어떤 규칙으로 처리했는지 나중에 재현할 수 있어야 하기 때문이다.

> 강의의 `coursekit`은 여러 처리 함수를 묶은 수업용 코드 모음이다. 다음 단계의 함수명은 자료에 제시된 함수와 역할을 기준으로 설명한다. 각 단계가 모두 독립 실행 프로그램이라는 뜻은 아니다.

## STEP 1. Collect — 원본 파일 수집

**담당:** Python 파일 수집 코드 (`pathlib.Path` 등)

**입력:** `attendance.md` 파일 자체

**실행 순서**

1. Python 코드가 지정된 폴더에서 파일 목록을 확인한다.
2. 읽을 수 있는 파일인지, 지원하는 형식인지 확인한다.
3. **원본 파일의 바이트(bytes)**를 읽어 보존한다.
4. 파일에 문서 ID, 버전, 위치, 크기, SHA-256 해시를 기록한다.
5. 결과를 **Manifest(처리 대상 장부)**에 적는다.

**출력 예시**

```text
doc_id         = DOC-ATTEND
source_version = v1
source_uri     = 원본 attendance.md의 위치
raw_sha256     = 원본 bytes의 해시
status         = collected
```

**왜 필요한가?** 이후 파싱·정제 규칙이 잘못되더라도 **원본에서 다시 시작**할 수 있어야 한다.

**주의:** 수집 성공(`collected`)은 텍스트 추출 성공을 뜻하지 않는다. 스캔 PDF도 파일 수집은 성공할 수 있다.

## STEP 2. Parse — 파일 구조를 읽고 텍스트를 추출

**담당:** 파일 형식별 Parser (수업 코드의 `parse_one()` 등)

**입력:** 수집한 원본 파일

**실행 순서**

1. 파일 형식에 맞는 읽기 방법을 선택한다.
2. 내용과 함께 제목·절·문단·표·PDF 페이지처럼 **문장의 의미를 설명하는 구조**를 확인한다.
3. 문서 안에서 처리 가능한 단위인 **ParsedUnit**을 만든다.
4. 해당 내용이 원본의 어디에 있는지 **Source Locator**를 기록한다.

**Markdown 예시**

```text
원본 Markdown
  # 출석 안내
  ## 지각과 결석
  수업 시작 후 10분을 초과하면 결석으로 기록합니다.
       ↓ Parse
ParsedUnit
  unit_id: sec-1
  heading_path: 출석 안내 > 지각과 결석
  raw_text: 수업 시작 후 10분을 초과하면 결석으로 기록합니다.
  source_locator: attendance.md#지각과-결석
```

**출력:** 제목 경로와 위치 정보를 포함한 ParsedUnit

**왜 필요한가?** 단순히 글자만 추출하면 “10분 초과”와 “결석”의 관계, 표의 행·열 관계, 어떤 페이지의 내용인지 잃어버릴 수 있다.

**파일별로 달라지는 점**

| 파일 종류 | 읽는 방법 | 특히 조심할 점 |
|---|---|---|
| TXT | UTF-8로 bytes를 문자열로 변환 | 글자 깨짐, 인코딩 오류 |
| Markdown | 제목(`##`), 목록 등 구조 해석 | 상위 제목과 소제목의 관계 |
| HTML | Beautiful Soup 등으로 본문 추출 | 메뉴·광고·스크립트 제외, 표의 행·열 보존 |
| PDF | pypdf 등으로 페이지별 텍스트 추출 | 읽는 순서·표·페이지 위치 |
| 이미지 중심 PDF | 텍스트층 여부 확인 후 OCR 필요성 판단 | pypdf는 이미지 글자를 OCR하지 않음 |

**중요:** PDF에서 글자가 추출되지 않았다고 무조건 스캔 PDF라고 확정할 수는 없다. 손상·암호화·빈 페이지 등의 가능성도 확인한다.

## STEP 3. Clean — 검색용 텍스트를 조심스럽게 정리

**담당:** 정제 함수 (`clean_prose()` 등)

**입력:** ParsedUnit의 `raw_text`

**실행 순서**

1. Windows식·다른 줄바꿈을 통일한다.
2. 일반 본문의 불필요한 연속 공백을 줄인다.
3. 필요하면 NFC 방식으로 글자 표현을 통일한다.
4. 숫자·부정어·단위·조건·예외가 바뀌지 않았는지 확인한다.
5. **원래 추출 텍스트(`raw_text`)를 지우지 않고**, 정제된 `clean_text`를 별도로 만든다.

**예시**

```text
raw_text   = "수업 시작 후  10분을  초과하면 결석으로 기록합니다."
                         ↓ Clean
clean_text = "수업 시작 후 10분을 초과하면 결석으로 기록합니다."
```

**출력:** CleanUnit(`clean_text`, `cleaner_version`, `clean_hash` 등)

**왜 필요한가?** 검색에 방해가 되는 불필요한 형식 차이는 줄이되, 원문에 쓰인 **사실을 그대로 남기기 위해서**다.

**절대로 하면 안 되는 정제 예시**

```text
원문: 제출할 수 없습니다.
오류: 제출할 수 있습니다.   ← '없'을 삭제해서 의미가 뒤집힘
```

**주의:** `strip()` 같은 공백 삭제도 코드 들여쓰기·표 정렬에는 위험할 수 있다. 동일 규칙을 모든 문서에 기계적으로 적용하지 않는다.

## STEP 4. Chunk — 긴 글을 검색용 조각으로 나누기

**담당:** 분할 함수 (`make_chunks()`, Chunker)

**입력:** CleanUnit의 `clean_text`와 제목·문단 정보

**실행 순서**

1. 어디서 나눌지 기준을 정한다(글자 수, Token 수, 제목·문단 구조 등).
2. **한 조각만 읽어도 주요 조건과 결론을 이해할 수 있게** 나눈다.
3. 인접한 조각의 경계가 끊기지 않도록 필요하면 Overlap을 둔다.
4. 각 조각이 `clean_text`에서 차지한 문자 시작·끝 위치를 기록한다.
5. 최종 Embedding 입력을 구성하고 실제 Token 수를 확인한다.

**좋은 분할**

```text
Chunk A: 수업 시작 후 10분을 초과하면 결석으로 기록합니다.
Chunk B: 출석 관련 문의는 운영실에 문의하세요.
```

**나쁜 분할**

```text
Chunk A: 수업 시작 후 10분을 초과하면
Chunk B: 결석으로 기록합니다.
```

두 번째 경우에는 조건과 결과가 갈라져 검색 후 답하기 어려워진다.

### Chunk Size와 Overlap 예시

원문 길이를 **71자**, `size=40`, `overlap=10`으로 설정했다면:

```text
Chunk 1: [0, 40)   → 원문 0~39번 문자
Chunk 2: [30, 70)  → 원문 30~69번 문자
Chunk 3: [60, 71)  → 원문 60~70번 문자
```

- 인접 Chunk가 10자씩 겹친다.
- 저장된 Chunk 길이의 합은 40+40+11=91자다.
- 실제 원문의 고유 구간은 71자 그대로다.
- 대괄호 `[start, end)`에서 **start는 포함, end는 제외**다.
- `char_start`와 `char_end`는 **정제된 `clean_text` 기준 위치**다. PDF 바이트 위치가 아니다.

**왜 필요한가?** 문서 전체를 하나의 벡터로 만들면 서로 다른 주제가 섞이고, 모델 입력 한도를 넘거나 필요한 부분을 정확히 찾기 어려울 수 있기 때문이다.

### 384와 512를 헷갈리지 말자

- **384차원:** E5-small이 출력하는 벡터의 숫자 개수
- **최대 512 Token(수업 모델 기본값):** 모델에 넣을 수 있는 입력 길이 기준. 실제 설정을 확인해야 한다.
- **40자:** 분할 예시에서 사용한 글자 수. 40 Token과는 다르다.
- **`passage:` 접두어와 제목을 포함한 최종 입력 전체**를 Tokenizer로 세어야 한다.

## STEP 5. Validate — 만든 Chunk가 믿을 만한지 확인

**담당:** 검증 함수 (`validate_span()` 등)

**입력:** Chunk, 정제 원문, 출처·모델 설정

**실행 순서**

1. Chunk의 시작·끝 위치가 유효한지 검사한다.
2. `clean_text[start:end]`가 Chunk 본문과 **정확히 같은지** 검사한다.
3. `passage:` 등 최종 Embedding 입력의 Token 수가 한도 안에 있는지 확인한다.
4. 빠진 구간이 없는지 **Coverage**를 계산한다.
5. 동일한 `chunk_id`가 잘못 중복되었는지 확인한다.
6. 문서 ID, 버전, 원문 위치가 연결되어 있는지 검사한다.
7. 문제가 있으면 이유를 기록하고 검색 대상에서 **격리(Quarantine)**한다.

**예시**

```python
# 검증 원리를 보여주는 코드 조각
assert chunk_body == clean_text[char_start:char_end]
```

**왜 필요한가?** 잘못 잘린 자료를 Embedding하고 저장하면, 나중에 검색 결과가 그럴듯해 보여도 답변 근거 자체가 틀릴 수 있기 때문이다.

### Coverage의 뜻

**Coverage = Chunk들이 덮고 있는 원문 고유 문자 위치 수 ÷ clean_text 전체 문자 수**

```text
원문 20자
Chunk 구간: [0,8), [6,14), [12,20)
→ 원문의 0~19번 위치가 모두 포함됨 → Coverage 100%
```

단, **Coverage 100%가 원본 PDF를 완벽하게 읽었다는 증거는 아니다.** Parsing 단계에서 이미 내용이 빠졌다면 그 빠진 내용은 계산 대상에도 없기 때문이다.

**Overlap과 Duplicate도 구분하자.** Overlap은 일부 문맥을 의도적으로 겹치는 설계이고, Duplicate는 동일 Chunk가 실수로 중복 생성된 문제다.

## STEP 6. Metadata — 조각마다 '신분증' 붙이기

**담당:** 문서 처리 코드의 Metadata 구성 단계

**입력:** 검증된 Chunk와 원문·처리 이력

**실행 순서**

1. 이 조각이 **어떤 원본 문서**에서 왔는지 기록한다.
2. 원문이 **몇 번째 버전**인지 기록한다.
3. 어떤 페이지·절·문자 위치의 내용인지 기록한다.
4. 문서 분류(`category`)와 처리 규칙 버전을 기록한다.
5. 각각을 구별할 `chunk_id`를 부여한다.

**출력 예시**

```text
doc_id           = DOC-ATTEND
source_version   = v1
unit_id          = sec-1
chunk_id         = 이 조각의 고유 문자열 ID
category         = 학사
source_locator   = attendance.md#지각-결석
char_start/end   = 정제 본문 안의 문자 시작/끝
pipeline_version = ingest-e5-v1
```

**왜 필요한가?** 나중에 검색 결과가 나오면 “어디서 가져왔어?”, “이전 규정이야, 최신 규정이야?”에 답할 수 있어야 하기 때문이다.

**용어 구분**

- `doc_id`: **문서**를 구별하는 ID
- `unit_id`: 그 문서의 **절·페이지**를 구별하는 ID
- `chunk_id`: 검색용 **한 조각**을 구별하는 ID(문자열)
- `source_locator`: 사람이 **원본 안의 위치**를 찾을 정보
- `source_version`: **원문 내용 개정판** (`v1`, `v2`)
- `pipeline_version`: **처리 방식**의 버전(분할·정제·모델 규칙)
- `citation_id`: 검색 결과의 출처 참조에 사용할 **정수 ID**(저장 단계에서 관리)

> Metadata는 분할·검증 과정에서도 함께 만들어지고 전달될 수 있다. 여기서 별도 STEP으로 뺀 것은 **역할을 이해하기 위해서**다. 실제 코드가 반드시 검증 이후에만 Metadata를 생성한다는 뜻은 아니다.

## STEP 7. Embed — 통과한 Chunk를 숫자 벡터로 변환

**담당:** Python 코드가 호출하는 E5 Embedding 모델 (`encode_passages()` 등)

**입력:** 검증을 통과한 Chunk의 `embedding_text`

**실행 순서**

1. Chunk 본문과 필요하면 제목 경로를 준비한다.
2. 문서에 해당하므로 `passage: `를 붙인다.
3. 최종 문자열을 E5-small 모델에 전달한다.
4. 모델이 각 Chunk를 **384개의 숫자로 이루어진 벡터**로 바꾼다.
5. 차원, NaN·Inf 여부, 벡터 길이 정규화 조건을 확인한다.
6. **Chunk와 벡터가 1:1로 연결되는지** 확인한다.

```text
Chunk 본문: 수업 시작 후 10분을 초과하면 결석으로 기록합니다.
          ↓
입력: passage: 출석 안내 / 지각과 결석 ...
          ↓
E5-small 모델
          ↓
[숫자1, 숫자2, 숫자3, ... 숫자384]
```

**왜 필요한가?** 질문도 나중에 같은 E5 모델로 숫자로 바꾸면, 질문 벡터와 문서 벡터 사이의 거리를 계산할 수 있기 때문이다.

**매우 중요:** Chunk를 저장할 때와 질문할 때 **동일한 모델 계약**(모델·접두어·정규화·차원)을 따라야 한다. 같은 384차원이라도 다른 모델의 벡터를 무조건 섞으면 안 된다.

## STEP 8. Index & Publish — PostgreSQL에 저장하고 검색에 공개

**담당:** Python의 DB 처리 코드 (`stage_version()`, `publish_version()` 등) + PostgreSQL/pgvector

**입력:** Chunk 본문, Embedding 벡터, Metadata, 문서/처리 버전

**실행 순서**

1. PostgreSQL에 문서 정보와 버전 정보를 기록한다.
2. Chunk **본문·벡터·원문 위치·ID를 한 행으로 연결해 저장**한다.
3. 새로운 문서 버전은 **staging(준비 중)**으로 다룬다.
4. 검증이 끝나면 **ready(사용 준비 완료)** 상태가 된다.
5. 짧은 DB Transaction 안에서 **현재 공개 버전 포인터**를 변경한다.
6. 포인터가 새 버전을 가리키면 그 버전이 **published(검색 대상)**가 된다.

### 문서를 어디에 저장하나?

| 수업 테이블 | 내용 | 쉬운 비유 |
|---|---|---|
| `ingest_documents` | 문서 ID와 현재 공개할 버전 | 문서 목록 장부 |
| `ingest_versions` | 원문·처리 버전별 준비/실패 상태 | 개정 이력 장부 |
| `ingest_chunks` | 본문·벡터·출처·ID·Metadata | 실제 검색할 문서 조각 보관함 |

pgvector는 `ingest_chunks`의 `embedding vector(384)` 같은 벡터 자료형과 거리 연산을 지원한다.

### 왜 버전을 관리하나?

```text
기존 공개 버전 v1: Chunk A, B, C
새 버전 v2 준비 중: Chunk A, B

v2 준비/검증 중 → 사용자 검색에는 v1 유지
v2 검증 성공 + 공개 전환 → 사용자 검색에는 v2만 사용
v2 검증/저장 실패 → 사용자 검색에는 기존 v1 유지
```

이전 버전의 Chunk가 DB에 남아 있어도, **현재 공개 버전 조건**으로 검색에서 제외할 수 있다. `UPSERT`만으로 사라진 Chunk가 자동 삭제되지는 않는다.

**주의:**

- `ready`는 곧바로 ‘사용자에게 공개됨’을 뜻하지 않는다. 현재 공개 포인터가 그 버전을 가리켜야 한다.
- **Transaction**은 여러 DB 변경을 함께 확정하거나(Commit) 문제가 생기면 취소(Rollback)하도록 묶는 것이다.
- Embedding 모델 실행 내내 긴 DB Transaction을 열어 두지 않는다.
- 문서 인덱싱 작업 전체와 `CREATE INDEX`(예: HNSW 검색 보조 인덱스 생성)는 다른 개념이다.

## STEP 9. Retrieve — 저장된 Chunk가 실제로 검색되는지 확인

**담당:** Python 검색 함수 (`encode_query()`, `search_current()` 등) + PostgreSQL/pgvector

**입력:** 사용자 질문, 필요한 분류 조건, `k` 값

**실행 순서**

1. Python 검색 함수가 질문과 검색 조건을 확인한다.
2. 질문에는 `query: ` 접두어를 붙여 **동일 E5 모델**로 384차원 벡터를 만든다.
3. PostgreSQL에 검색 SQL을 요청한다.
4. SQL의 `WHERE` 조건으로 **현재 공개된 문서 버전**과 `category` 등 검색 대상 범위를 제한한다.
5. pgvector의 Cosine Distance(` <=> `)를 계산한다.
6. 거리가 작은 순서(`ASC`)로 최대 `k`개를 고른다.
7. Python 검색 함수가 결과의 **본문·출처·ID·거리**를 받아 반환한다.

```text
질문: "15분 지각하면 어떻게 되나요?"
   ↓
질문 벡터 생성: query: 15분 지각하면 어떻게 되나요?
   ↓
[현재 공개된 출석 규정 Chunk만 검색]
   ↓
가까운 Chunk 후보와 원문 위치 반환
   ↓
(다음 수업) Context에 넣을 근거 선별 → LLM 답변 생성
```

**반환하는 결과의 주요 정보**

| 필드 | 쉬운 의미 |
|---|---|
| `citation_id` | 최종 답변의 출처에 사용할 정수 번호 |
| `chunk_id` | 어떤 검색용 조각인지 나타내는 문자열 ID |
| `body` | 검색된 실제 본문 |
| `doc_id` | 원본 문서 식별자 |
| `source_locator` | 원본 파일의 어느 절·페이지인지 |
| `source_version` | 어느 버전의 문서인지 |
| `distance` | 질문 벡터와의 거리(작을수록 가까움) |

**여기까지가 오늘 강의의 범위다.** LLM을 호출해 사용자에게 자연어 답을 만드는 과정은 **이후 Basic RAG**에서 연결한다.

---

# PART C. 입·출력으로 다시 보는 실행 순서

| 순서 | 실제 수행 코드/구성요소 | 입력 | 출력 | 왜 필요한가? |
|---|---|---|---|---|
| 1 수집 | Python 파일 수집 코드 | 원본 파일 | Raw File + Manifest | 원본·이력 보존 |
| 2 파싱 | `parse_one()` / 형식별 Parser | Raw File | ParsedUnit | 구조·위치 추출 |
| 3 정제 | `clean_prose()` | `raw_text` | `clean_text` | 의미를 유지한 형식 정리 |
| 4 분할 | `make_chunks()` | 정제 원문·크기·Overlap | Chunk/Span | 검색 가능한 단위 준비 |
| 5 검증 | `validate_span()` 및 추가 검사 | Chunk·원문·모델 조건 | 통과/격리/보고 | 잘못된 자료 차단 |
| 6 메타데이터 | 처리 코드의 ID·출처 구성 | 문서·조각·버전 | Metadata | 출처·공개판 확인 |
| 7 벡터화 | `encode_passages()` / E5 | `passage:` + Chunk | 벡터 384개 값 | 질문과 비교할 숫자 생성 |
| 8 저장/공개 | `stage_version()` / `publish_version()` | Chunk·벡터·출처 | 공개 버전 DB | 검색할 준비 마침 |
| 9 검색 | `encode_query()` / `search_current()` | 질문·category·k | Chunk+출처+거리 | 질문과 가까운 자료 확인 |

> **실제 처리 코드에서는 일부 검증과 Metadata 작성 시점이 서로 얽혀 있을 수 있다.** 이 표는 개념 이해를 위해 단계를 선형으로 펼친 것이다.

---

# PART D. 초보자를 위한 주요 개념 12가지

## 1. Document Ingestion은 ‘문서를 검색할 수 있게 준비하는 전 과정’이다

- **뭐야?** 파일 수집부터 검색할 데이터 저장·공개까지의 작업.
- **왜 필요해?** AI가 원본 PDF를 그냥 받았다고 검색용 데이터베이스가 자동 완성되는 것은 아니기 때문.
- **헷갈림:** Ingestion과 Retrieval은 다르다. **전자는 사전 준비, 후자는 질문에 따른 검색**이다.

## 2. Parsing은 ‘읽기’, Cleaning은 ‘정리’다

- Parsing은 파일 내부의 글과 구조를 **꺼내는** 작업.
- Cleaning은 꺼낸 글의 불필요한 형식을 **다듬는** 작업.
- 예를 들어 PDF의 2페이지 본문을 찾는 일은 Parsing, 과도한 공백을 정리하는 일은 Cleaning.

## 3. Chunk는 ‘원본 일부를 담은 검색용 작은 조각’이다

- 한 책을 한 번에 찾기보다, 필요한 **절·문단**을 찾는 것에 가깝다.
- 너무 작으면 조건과 결과가 떨어지고, 너무 크면 주제가 섞일 수 있다.

## 4. Token과 글자는 다르다

- 글자 수는 사람이 보는 문자 개수, Token은 모델 Tokenizer가 나눈 처리 단위다.
- 문서가 100글자여도 Token이 꼭 100개인 것은 아니다.
- 384차원은 벡터 **출력 크기**, 최대 512 Token은 **입력 길이** 문제다.

## 5. Overlap은 ‘조각 사이 일부 겹침’이다

- 앞뒤 두 Chunk에 같은 문장이 일부 포함되도록 하는 기법이다.
- 경계 문맥을 보존하는 데 도움을 줄 수 있지만 중복 저장이 늘어난다.

## 6. Offset/Span은 ‘문서 조각이 어디서 잘렸는지’다

- 예: `clean_text[30:70]` = 정제 문서의 30번째부터 69번째 문자까지.
- **PDF 페이지 번호와 혼동하지 말 것.** 원본 위치는 `source_locator`가 따로 기록한다.

## 7. Validation은 ‘잘못된 근거를 저장하지 않도록 하는 안전 검사’다

- Chunk의 실제 내용과 원문 Slice가 같은지, 길이를 넘지 않았는지, 누락·중복은 없는지 검사한다.
- 실패하면 **왜 실패했는지 기록하고 격리**한다.

## 8. Metadata는 ‘출처 정보가 적힌 꼬리표’다

- 본문 외에 문서 ID, 페이지·절, 작성/처리 버전, 분류 등을 담는다.
- AI의 근거 출처를 찾고 최신 규정만 검색할 때 필요하다.

## 9. Embedding은 ‘텍스트를 비교 가능한 숫자 벡터로 바꾸기’다

- E5-small을 사용하면 Chunk마다 384개의 숫자 배열이 생성된다.
- 이 숫자 자체가 정답이나 참/거짓을 뜻하지 않는다.

## 10. Indexing은 ‘찾을 수 있도록 등록하는 작업’이다

- 오늘의 **문서 인덱싱**은 Chunk·벡터·출처를 DB에 등록하고 공개 상태를 관리하는 것.
- SQL의 **데이터베이스 인덱스**(예: HNSW)는 조회를 빠르게 하려는 별도 자료구조다.

## 11. 공개 버전은 ‘지금 검색에 사용되는 문서 판’이다

- 새로운 v2가 준비되어도 검증되기 전에는 v1을 계속 노출할 수 있다.
- v2에 오류가 생겼다면 기존 v1을 유지한다. 단, 접근 철회·삭제 요청은 별도의 우선 정책으로 다뤄야 한다.

## 12. Retrieval은 ‘답변 생성’이 아니라 ‘근거 후보 찾기’다

- 검색 결과가 나왔다고 반드시 질문에 답할 근거가 있다는 뜻은 아니다.
- 이번 수업은 검색 결과의 **본문과 출처를 반환**하는 것까지 다룬다.

---

# PART E. 용어 사전 — 쉬운 뜻 + 사용되는 지점

> 원본 자료의 §10 용어 카드를 바탕으로 **초보자가 외우기 쉬운 한국어 설명**으로 풀었다. 아래의 `예시/언제`는 기억을 돕기 위한 간단한 사례다.

## E-1. 문서·파일·수집 용어

| 용어 | 초보자용 뜻 | 예시 / 언제 쓰나 |
|---|---|---|
| Document | 처리할 문서 하나 | `attendance.md` |
| Document Ingestion | 문서를 검색 가능한 데이터로 만드는 전체 과정 | PDF를 Chunk+벡터로 등록 |
| Collect / Collection | 처리할 원본 파일을 모으는 일 | `data/raw`에서 파일 읽기 |
| Raw File | 수정하지 않은 원본 파일 | 원본 PDF/TXT bytes |
| Derived Data | 원본을 가공해 나온 결과 | ParsedUnit, CleanUnit, Chunk |
| bytes | 파일의 이진 데이터 단위 | 파일 읽기를 `read_bytes()`로 수행 |
| UTF-8 | 글자를 bytes로 표현하는 흔한 규칙 | 한글 ‘가’는 UTF-8에서 3 bytes |
| Encoding | 글자를 bytes로 바꾸기 | `text.encode('utf-8')` |
| Decoding | bytes를 글자로 바꾸기 | `raw.decode('utf-8')` |
| SHA-256 | bytes를 일정 길이의 해시 값으로 바꾸는 함수 | 원본 변경 여부 확인 |
| Hash | 입력으로부터 계산한 요약값 | 파일 내용 비교의 단서 |
| Manifest | 처리 대상 목록과 상태를 적은 장부 | 6건 중 성공/보류/미지원 집계 |
| `doc_id` | 문서 자체를 구별하는 번호/이름 | `DOC-ATTEND` |
| `source_version` | 원본 문서의 개정판 | `v1`, `v2` |
| `source_uri` / URI | 원본 파일이나 리소스를 가리키는 위치 | 파일 URI |
| File extension | 파일명 뒤에 붙는 형식 표시 | `.md`, `.pdf` |
| `pathlib.Path` | Python에서 파일 경로를 다루는 도구 | `Path('data/raw')` |
| `status` | 해당 처리의 현재 결과 상태 | `collected`, `unsupported` |

## E-2. Parsing 및 원문 구조 용어

| 용어 | 초보자용 뜻 | 예시 / 언제 쓰나 |
|---|---|---|
| Parsing | 파일 구조를 해석하며 읽기 | PDF 페이지 단위 읽기 |
| Parser | Parsing 작업을 수행하는 코드/라이브러리 | `pypdf`, Beautiful Soup |
| Text Extraction | 읽은 구조에서 글자를 꺼내기 | `page.extract_text()` |
| ParsedUnit | 파싱 후 나온 문서 속 단위 | PDF 2페이지, Markdown의 한 절 |
| `raw_text` | 파싱으로 뽑아 낸, 아직 정제하지 않은 글 | 연속 공백이 남은 본문 |
| Markdown | `#`, `-` 등으로 구조를 적는 문서 형식 | 강의 노트 `.md` |
| Heading | 제목 | `## 지각과 결석` |
| Heading Path | 큰 제목에서 작은 제목으로 이어지는 경로 | `출석 안내 > 지각과 결석` |
| HTML | 웹 문서의 태그 기반 형식 | `<main>`, `<table>` |
| HTML element | HTML 문서 안의 하나의 요소 | `<main> ... </main>` |
| Beautiful Soup | HTML 구조를 다루는 Python 라이브러리 | 본문 영역 선택 |
| `get_text()` | HTML 요소 속 글자를 이어서 꺼내는 기능 | 표의 행/열 의미 보장 X |
| PDF | 인쇄·화면 배치를 중심으로 만든 문서 형식 | 페이지·표 순서 확인 필요 |
| pypdf / PdfReader | PDF 텍스트와 페이지를 읽는 도구 | `PdfReader(...).pages` |
| OCR | 이미지에 찍힌 글자를 문자 데이터로 인식 | 스캔 PDF 처리 시 필요할 수 있음 |
| Scanned PDF | 내용이 주로 이미지인 PDF | 글자가 보여도 문자 추출 불가할 수 있음 |
| OCR Text Layer | 이미지와 함께 저장된 OCR 인식 문자층 | 오인식 가능 |
| `unit_id` | 문서 속 특정 절·페이지 구별 | `page-2`, `sec-1` |
| Source Locator | 사람이 원본의 구체 위치를 찾는 정보 | `career.pdf#page=2` |
| `page_number` | 실제 PDF 페이지 번호 | Markdown/TXT는 보통 `None` |

## E-3. Cleaning 및 문자열 위치 용어

| 용어 | 초보자용 뜻 | 예시 / 언제 쓰나 |
|---|---|---|
| Text Cleaning | 의미는 유지하며 글자 형식 정리 | 연속 공백 줄이기 |
| CleanUnit | 정제가 끝난 처리 단위 | `clean_text`와 버전 포함 |
| `clean_text` | 정제 후 본문 | Chunking의 직접 기준 텍스트 |
| Whitespace | 공백·탭·줄바꿈 같은 빈 문자 | 두 칸 공백을 한 칸으로 |
| CRLF / CR / LF | 운영체제·문서별 줄바꿈 표현 | `\r\n`, `\r`, `\n` |
| Unicode Normalization | 같은 글자의 표현 방식을 일정하게 만들기 | 여러 형태의 ‘가’ 맞춤 |
| NFC | 정준적으로 같은 문자를 조합해 통일하는 방식 | 오늘 일반 본문의 기본 선택 |
| NFKC | 호환 문자까지 바꾸는 정규화 | `①`가 `1`로 바뀔 수 있어 주의 |
| Regular Expression | 문자열 패턴을 찾거나 바꾸는 규칙 | 연속 공백 찾기 |
| `strip()` | 문자열 양 끝 공백 제거 | 일반 본문에는 유용, 코드 들여쓰기 주의 |
| Offset | 기준 문자열에서의 위치 숫자 | `char_start=30` |
| Span | 시작~끝의 연속 구간 | `[30,70)` |
| Character Slice | Python 문자열의 일부 잘라 가져오기 | `text[30:70]` |
| Span Map | 정제 전후 문자 위치 대응 기록 | 원본 위치 역추적 필요할 때 |
| `cleaner_version` | 정제 규칙의 버전 | `clean-v1` |
| `clean_hash` | 정제 본문 bytes의 해시 | 원본 해시와 다른 대상 |
| Idempotency / 멱등성 | 같은 작업을 재실행해도 결과가 바뀌지 않는 성질 | 이미 정제된 글을 다시 정제해도 같음 |

## E-4. Chunking 및 Token 용어

| 용어 | 초보자용 뜻 | 예시 / 언제 쓰나 |
|---|---|---|
| Chunk | 검색하려고 잘라 둔 문서 조각 | 출석 기준 문단 한 개 |
| Chunking | 문서를 Chunk로 분할하는 과정 | 문단별로 자르기 |
| Chunker | Chunking을 수행하는 코드 | `make_chunks()` |
| Chunk Size | 조각 하나의 길이 기준 | 40자, 160 Token 등 |
| Overlap | 이웃 조각 간 일부 겹치는 구간 | 끝 10자를 다음에도 포함 |
| Fixed Character Window | 지정한 문자 수로 일정하게 분할 | 40자씩 나누기 |
| Token-based Chunking | Token 개수를 기준으로 분할 | 모델 입력 한도 고려 |
| Structure-aware Chunking | 제목·문단·표 같은 구조를 고려해 분할 | ‘10분 초과 → 결석’을 함께 유지 |
| Semantic Chunking | 의미 변화가 나타나는 지점으로 분할 | 오늘은 개념 소개 수준 |
| Token | 모델이 입력을 처리할 때 나누는 단위 | 글자 수와 같지 않음 |
| Tokenizer | 문장을 Token ID로 바꾸는 도구 | 모델 입력 길이 계산 |
| Special Tokens | 모델이 입력 경계 등을 구분하는 토큰 | 최종 토큰 수에 포함해 계산 |
| `truncation=False` | 입력을 임의로 자르지 않고 길이 확인 | 너무 긴 Chunk 찾아내기 |
| Embedding Input | 실제 모델에 넣는 최종 텍스트 | `passage: 제목 + 본문` |
| `query:` / `passage:` | E5 모델의 질문/문서 입력 역할 표시 | 질문에는 `query:`, 문서에는 `passage:` |
| Embedding Dimension | 결과 벡터의 숫자 개수 | E5-small → 384차원 |
| Max Tokens | 모델에 입력할 수 있는 Token 상한 | 수업 모델 기본 512(실제 확인) |
| Trade-off | 하나를 늘리면 다른 비용도 늘어나는 관계 | Overlap↑ → 문맥↑·중복↑ |
| Generator / `yield` | 값을 한 번에 모으지 않고 하나씩 내보내기 | Chunk를 순서대로 생성 |

## E-5. 품질 검사·오류·평가 용어

| 용어 | 초보자용 뜻 | 예시 / 언제 쓰나 |
|---|---|---|
| Validation | 만들어진 결과가 규칙을 지키는지 검사 | Chunk 길이·위치 확인 |
| Validator | 검사를 실행하는 함수·모듈 | `validate_span()` |
| Invariant / 계약 | 처리 내내 반드시 성립해야 할 조건 | Chunk 본문 = 원문 Slice |
| Span Contract | 조각의 범위와 실제 텍스트가 일치해야 한다는 조건 | `text == clean_text[start:end]` |
| Token Contract | 모델의 입력 길이·벡터 규칙 지키기 | 너무 긴 입력 거부 |
| Identity Contract | Chunk가 정확한 문서/버전에 연결돼야 한다는 조건 | doc_id·unit_id 일치 |
| Coverage | 정제 본문에서 Chunk가 빠짐없이 덮은 비율 | 전체 20자 중 20자 덮음 = 100% |
| Duplicate | 실수로 같은 조각을 중복 생성한 상태 | 같은 chunk_id 반복 |
| Interval Merge | 여러 문자 구간을 합쳐 중복 없이 길이 계산 | Coverage 계산 |
| Quarantine | 문제 있는 자료를 이유와 함께 격리 | 토큰 초과 Chunk 보류 |
| `needs_ocr` | 이미지 글자 인식이 필요해 보류한 상태 | 스캔 PDF |
| `unsupported` | 현재 처리기가 지원하지 않는 형식 | `.xyz` |
| `span_mismatch` | Chunk 본문과 원문 위치 내용이 다름 | 다른 버전의 clean_text 사용 |
| `too_many_tokens` | 모델 입력 한도를 넘음 | 긴 본문 다시 분할 |
| `missing_source` | 출처 정보와 연결되지 않음 | doc_id/버전 누락 |
| Regression Check | 규칙 수정 후 예전 정상 사례도 검사 | 정제 규칙 교체 후 부정어 유지 |
| Recall@k | 상위 k개에서 정답 근거를 얼마나 찾았는지 보는 지표 | 이후 평가 차시에서 본격 학습 |
| MRR | 첫 번째 정답 근거가 얼마나 앞 순위에 있는지 보는 지표 | 이후 평가 차시 |
| nDCG | 관련성 등급과 순위를 함께 고려하는 검색 평가 지표 | 이후 평가 차시 |
| Faithfulness | 생성 답변이 제공 근거와 일치하는 정도 | 실제 답변 생성 이후 평가 |

## E-6. Metadata·DB·버전·검색 용어

| 용어 | 초보자용 뜻 | 예시 / 언제 쓰나 |
|---|---|---|
| Metadata | 본문에 붙이는 출처·분류·버전 설명 | `category='학사'` |
| `category` | 자료의 내용 분류 | 학사/시설/지원 |
| Metadata Filter | 특정 조건의 자료만 검색 대상으로 제한 | 학사 분야만 검색 |
| `WHERE` | SQL에서 검색 대상 행에 조건을 거는 부분 | `WHERE category = ...` |
| Top-k | 거리 기준으로 최대 k개 후보 선택 | `k=3` → 최대 3개 |
| Exact Search | 조건에 맞는 검색 대상의 거리를 비교하는 정확 검색 | 작은 수업 DB에서 기준선 |
| ANN | 근사 최근접 이웃 검색 | 속도를 위해 일부 후보만 탐색 |
| HNSW | ANN을 위한 그래프 방식 DB 인덱스 | 데이터가 많을 때 탐색 가속 |
| B-tree | 정렬/조건 검색에 쓰이는 일반적인 DB 인덱스 | 분류값 검색 보조 |
| Document Indexing | 문서를 처리해 검색 가능하게 등록·공개 | 오늘의 주제 전체 |
| Database Index | DB 조회를 빠르게 도와주는 자료구조 | `CREATE INDEX` |
| PostgreSQL | 데이터를 저장하고 SQL을 실행하는 DB | 원문·버전·청크 표 관리 |
| pgvector | PostgreSQL에서 벡터를 저장/비교하는 확장 | `vector(384)`, `<=>` |
| Cosine Distance | 벡터 방향 차이를 나타내는 거리 | 작을수록 가까움 |
| `<=>` | pgvector의 Cosine Distance 연산자 | `ORDER BY embedding <=> ... ASC` |
| `ASC` | 오름차순(작은 것부터) 정렬 | 작은 거리 먼저 |
| `vector(384)` | 384개 실수로 된 벡터 저장형 | E5-small 결과와 일치 |
| JSONB | PostgreSQL의 JSON 저장 자료형 | 추가 Metadata 저장 |
| Primary Key / PK | 표의 행을 유일하게 구분하는 키 | citation_id |
| Foreign Key / FK | 다른 표의 행과 연결을 지키는 키 | Chunk가 버전 레코드 참조 |
| `chunk_id` | Chunk를 구별하는 문자열 ID | 원문·버전·구간으로 안정적 생성 |
| Deterministic ID | 입력·규칙이 같으면 같은 ID 생성 | 정렬된 JSON과 SHA-256 |
| Serialization | 데이터를 정해진 순서/형식으로 문자열화 | 해시 계산 전에 dict → JSON |
| Content Hash | 본문 자체 내용이 같은지 확인하는 해시 | 출처가 같다는 뜻은 아님 |
| `pipeline_version` | 정제·분할·Embedding 등 처리 규칙의 버전 | `ingest-e5-v1` |
| UPSERT | 키 충돌 시 삽입 대신 갱신 등을 수행 | 같은 문서 재처리 |
| Staging | 새로운 버전을 준비하는 상태 | v2 저장·검증 중 |
| Ready | 검증을 통과한 버전의 상태 | 아직 공개 아닐 수도 있음 |
| Published / Current Pointer | 사용자 검색에 내보낼 버전을 가리키는 상태·값 | 현재 v2 검색 |
| State Machine | 허용된 상태와 상태 변화 규칙 | staging→ready, staging→failed |
| Transaction | DB 변경을 한 단위로 묶는 기능 | 버전 공개 전환 |
| COMMIT | Transaction의 DB 변경 확정 | 새 공개 버전 적용 |
| ROLLBACK | Transaction의 미확정 변경 취소 | 오류 시 기존 공개판 유지 |
| `register_vector(conn)` | Python DB 연결에서 vector 자료형 연동 등록 | 연결마다 등록 |
| SQL Parameter Binding | SQL 명령과 값을 분리하여 전달 | Psycopg `%s` 자리표시자 |
| `normalize_embeddings=True` | 생성 벡터의 길이를 1로 맞추는 옵션 | E5 검색 계약에 사용 |
| `np.isfinite` | 숫자에 NaN·무한대가 없는지 검사 | 벡터 품질 확인 |
| `citation_id` | 출처 표시용 정수 ID | `source_ids: list[int]`와 연결 |
| Citation Mapping | 정수 출처 ID와 문자열 Chunk ID/원문 위치 대응 | 최종 출처 검증 |
| Retriever Contract | 검색 함수가 반환하기로 약속한 결과 형식 | ID·본문·출처·거리 |
| End-to-End | 처음부터 끝까지 실제 연결을 검사 | 원본 → 검색 결과 |
| Pipeline / Config / Report | 처리 순서 묶음 / 설정 / 처리 결과 보고 | 실행 조건과 오류 기록 |

---

# PART F. 실제 예시 하나를 처음부터 끝까지 따라가기

**질문:** “수업에 15분 늦으면 어떻게 되나요?”

아래는 오늘의 두 경로가 연결되는 **설명용 시나리오**다.

### ① 사전 준비 — 관리자 또는 개발자가 문서 등록

```text
A. attendance.md 파일을 수집한다.
B. Markdown Parser가 '출석 안내 > 지각과 결석' 구역을 읽는다.
C. 연속 공백 등만 정리하고 '10분 초과 → 결석' 조건은 보존한다.
D. '10분 초과하면 결석'이 한 Chunk에서 읽히도록 분할한다.
E. Chunk 본문·위치·토큰 수·중복·출처를 검사한다.
F. 문서 ID·버전·절 위치·분류를 붙인다.
G. `passage:`를 붙여 E5-small로 벡터를 만든다.
H. Chunk·벡터·Metadata를 PostgreSQL에 저장한다.
I. 검증된 버전을 공개한다.
```

### ② 질문을 받았을 때 — 검색 함수가 실행

```text
A. 사용자 질문을 전달받는다.
B. 질문 검사를 통과한다.
C. `query: 수업에 15분 늦으면 어떻게 되나요?`를 Embedding한다.
D. PostgreSQL에서 현재 공개된 학사 관련 Chunk 등을 검색 대상으로 삼는다.
E. pgvector의 거리 계산으로 상위 k개 후보를 찾는다.
F. Python 검색 함수가 Chunk 본문·ID·출처 위치를 반환한다.
G. 다음 수업에서 관련 Chunk를 Context에 넣고 LLM 답변 생성에 사용한다.
```

**잘못된 상황은 어떻게 처리할까?**

| 발생 위치 | 문제 | 처리 방식 |
|---|---|---|
| Collect | 지원하지 않는 `.xyz` 파일 | `unsupported` 기록, 정상 처리로 보고하지 않음 |
| Parse | 이미지 PDF에서 문자를 읽지 못함 | 확인 후 `needs_ocr`로 보류 가능 |
| Clean | ‘제출 불가’를 ‘제출 가능’으로 변경 | 의미 손상으로 거부·규칙 수정 |
| Chunk | 10분 초과 조건과 결석 결과가 분리 | Chunk 경계 변경 |
| Validate | 원문 Slice와 Chunk 본문 불일치 | `span_mismatch` 기록·격리 |
| Embed | Token 길이 초과 | 재분할 또는 격리 |
| Publish | 새 버전 v2 저장·검증 실패 | 기존 v1 공개 유지 |
| Retrieve | 질문과 가까운 후보지만 답에 필요한 내용 없음 | 검색 성공과 답변 가능성을 구분 |

---

# PART G. 특히 헷갈리는 개념 비교

| A | B | 차이 |
|---|---|---|
| Parsing | Cleaning | 파일의 구조·글자 **읽기** vs 읽은 글자 **정리** |
| Cleaning | Chunking | 글자 형식 **정돈** vs 검색용 단위로 **자르기** |
| Chunking | Embedding | 본문을 **나누기** vs 나눈 본문을 **벡터로 바꾸기** |
| Embedding | pgvector | 벡터를 **생성하는 모델** vs 벡터를 **저장·비교하는 DB 확장** |
| Chunk | Token | 검색용 **문서 조각** vs 모델 입력을 이루는 **작은 단위** |
| 384차원 | 512 Token | 모델 **출력 크기** vs 모델 **입력 길이 제한** |
| `raw_text` | `clean_text` | 파일에서 **처음 꺼낸 글** vs 규칙에 따라 **정리한 글** |
| `raw_sha256` | `clean_hash` | 원본 파일 bytes의 해시 vs 정제된 문자열 bytes의 해시 |
| Span / Offset | Source Locator | `clean_text` 안 문자 인덱스 vs 사람이 원본에서 찾는 절·페이지 |
| Overlap | Duplicate | 의도적으로 경계 **겹치기** vs 잘못된 **중복 생성** |
| Coverage 100% | 원문 완전 보존 | 정제 텍스트의 위치가 다 덮임 vs 원본에서 의미·글자가 빠지지 않음(별도 검사) |
| `source_version` | `pipeline_version` | 문서 내용 개정판 vs 문서 처리 규칙 버전 |
| `doc_id` | `chunk_id` | 원본 문서 ID vs 문서 안 검색 조각 ID |
| `chunk_id` | `citation_id` | 문자열 검색 조각 ID vs 답변의 출처 표시용 정수 ID |
| Metadata `category` | 접근 권한 | 자료 분류 vs 사용자가 그 자료를 볼 수 있는 권한(별도 확인) |
| Filter | Top-k | 검색할 **대상 범위 제한** vs 그 대상에서 **가까운 k개 선택** |
| Indexing Pipeline | HNSW Index | 문서를 검색 가능하게 준비·공개하는 전체 작업 vs 벡터 검색 속도 개선 보조 구조 |
| Ready | Published | 검사 통과 상태 vs 현재 공개 포인터가 가리켜 검색에 쓰이는 상태 |
| Ingestion | Retrieval | 질문 전 문서 **준비** vs 질문 시 자료 **검색** |
| Retrieval | RAG 답변 생성 | 참고할 자료 **반환** vs 자료로 LLM이 **자연어 답변 생성** |

---

# PART H. 강의 실습을 볼 때 기억할 조건

- 강의에서 사용하는 모델: `intfloat/multilingual-e5-small`, **384차원**.
- 문서 입력은 `passage: `, 질문 입력은 `query: `.
- 모델 최대 입력 Token 수(기본 **512**)는 실제 Tokenizer 설정으로 확인한다.
- 실습 환경은 **PostgreSQL 18.6 + pgvector + 기존 Python 가상환경**이다.
- 수업용 5문서 + 미지원 형식 1문서 = **총 6건**의 설명용 입력 시나리오이며, 목표 상태는 `parsed 4 · needs_ocr 1 · unsupported 1`이다. 이는 자료의 예상 상태이지 실측 결과가 아니다.
- 강의의 기본 실행은 오프라인 준비 중심이므로 **실제 E5 실행·DB 저장이 자동으로 끝난 것으로 보면 안 된다.**
- 실제 Tokenizer/E5 측정: 강의 안내의 `python main.py --live-e5`.
- 실제 pgvector 연동: 강의 안내의 `python main.py --live-db`.
- DB 준비: `python tools/setup_db.py --apply` (실습 전용 스키마 준비; 기존 테이블 임의 변경 금지).
- 실제 라이브 실행에는 로컬 환경, 패키지, DB 접속·권한, 모델 파일이 필요할 수 있다.
- 스캔 PDF는 별도 OCR 여부를 확인하고, 비정상 결과를 정상 등록된 것처럼 처리하지 않는다.

---

# PART I. 복습: 네 말로 직접 절차를 설명해 보기

> 아래 빈칸은 **내 설명을 안 보고** 직접 적는 용도다. 다 적은 다음 ChatGPT에게 **순서·담당 코드·입력/출력·이유·오류 처리**를 기준으로 평가받으면 된다.

## I-1. 나만의 전체 흐름 정리

1. **문서가 들어온다.** 누가 무엇을 수집하고 기록하는가?
   - 내 설명:
2. **Parsing을 실행한다.** 어떤 파일 정보와 원본 위치를 꺼내는가?
   - 내 설명:
3. **Cleaning을 실행한다.** 무엇을 바꾸고 무엇을 절대 바꾸면 안 되는가?
   - 내 설명:
4. **Chunking을 실행한다.** 왜 자르고 어느 위치를 기록하는가?
   - 내 설명:
5. **Validation을 실행한다.** 어떤 실패를 검사하고 실패하면 어디로 보내는가?
   - 내 설명:
6. **Metadata를 연결한다.** 어떤 문서·버전·원본 위치를 남기는가?
   - 내 설명:
7. **Embedding 모델을 호출한다.** 무슨 텍스트를 입력하고 어떤 숫자가 나오는가?
   - 내 설명:
8. **PostgreSQL에 저장·공개한다.** 새 버전이 실패하면 어떻게 되는가?
   - 내 설명:
9. **질문이 들어오면 검색한다.** 검색 함수에서 DB까지 무엇이 전달되고 어떤 결과를 반환하는가?
   - 내 설명:

## I-2. 자가 점검 질문 12개

1. 어제 검색에 사용한 문장을 오늘은 원본 PDF·HTML로부터 어떻게 만드는가?
2. `collected`인데 `needs_ocr`일 수 있는 이유는?
3. Parsing과 Text Extraction은 어떻게 다른가?
4. 부정어·숫자를 지우는 Cleaning이 위험한 이유는?
5. `char_start=30`은 원본 PDF의 30번째 바이트인가?
6. Chunk Size 40자와 Token 40개는 같은가?
7. 왜 `passage:`와 Heading을 붙인 뒤 Token 수를 다시 확인해야 할까?
8. Coverage가 100%인데도 원본 내용이 빠질 수 있는 이유는?
9. `doc_id`, `chunk_id`, `citation_id`는 각각 무엇을 구분하나?
10. 왜 `WHERE category`를 적용한 다음 Top-k를 선택해야 할까?
11. `ready`인 v2가 존재해도 v1이 검색될 수 있는 이유는?
12. Day24의 마지막 검색 결과는 왜 아직 LLM의 최종 답변이 아닌가?

---

# PART J. 시험 전 30초 핵심 요약

> **원본 파일 수집 → Parsing(구조와 글 추출) → Cleaning(의미 보존 정제) → Chunking(검색용 분할) → Validation(위치·길이·누락 검사) → Metadata(출처·버전) → Embedding(E5, 384차원) → PostgreSQL + pgvector 저장·공개 → 질문 Embedding → 공개된 Chunk 필터 → 거리 기반 Top-k 검색 → 본문·ID·출처 반환.**
>
> 오늘은 **검색 가능한 근거를 준비하고 찾아오는 단계까지**다. 검색된 근거로 LLM 답변을 만드는 것은 다음 RAG 단계다.

**최우선 암기 포인트:**

- **원본은 절대 덮어쓰지 않는다.**
- **Cleaning은 의미를 바꾸면 안 된다.**
- **Chunk는 조건과 결론이 함께 남도록 나눈다.**
- **문자 수, Token 수, 벡터 차원은 서로 다르다.**
- **검증 실패는 조용히 삭제하지 말고 이유를 남겨 격리한다.**
- **출처와 버전을 함께 저장하고, 공개된 버전만 검색한다.**
- **검색 결과가 있다고 정답이 보장되는 것은 아니다.**

---

## 기준 자료 및 정리 방법

이 노트는 첨부된 Day24 원본 Markdown의 **전체 흐름(§0·§9), 8개 챕터(§1~§8), 용어 카드(§10), 복습·실습 조건(§11~§14)**을 바탕으로 초보자용으로 재구성했다. 원본에 나오는 API/함수와 **개념 설명을 위해 재구성한 단계**는 구분했다. 예시 출력과 숫자는 교육용이며 사용자 컴퓨터에서 실행·검증한 실제 결과가 아니다.
