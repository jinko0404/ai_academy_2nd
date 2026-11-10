# Day 17 | FastAPI 핵심 용어·기본 개념 정리

> 기준 자료: `Day17_FastAPI_지식관리API.md`  
> 학습 범위: FastAPI로 JSON 파일 기반 지식 CRUD API 만들기  
> 구분: **필수**는 오늘 직접 사용하는 개념, **확장**은 원본 강의의 응용·심화·후속 내용

## 1. 전체 그림

**오늘 만드는 것:** 사용자가 HTTP 요청을 보내면 FastAPI가 입력을 확인하고 Python 함수로 연결하여 JSON 파일에 지식을 등록·조회·수정·삭제하는 서비스.

```text
사용자 또는 /docs
    ↓ HTTP 요청 (Method + Path + 필요 시 Body)
Uvicorn (서버 실행)
    ↓
FastAPI (경로 연결·입력 검증)
    ↓
처리 함수 (CRUD)
    ↓
JSON 파일 (원본 데이터 저장)
    ↓
HTTP 상태코드 + JSON 응답
```

**중요:** 오늘의 지식 관리 API는 RAG 검색이나 AI Agent 자체가 아니다. 원본 지식을 관리하는 기반 기능이다.

## 2. 반드시 알아야 할 핵심 용어

| 용어 | 쉬운 설명 | 수업 예시 |
|---|---|---|
| **API** | 프로그램끼리 정해진 방식으로 요청·응답하는 연결 창구 | `/api/knowledge` |
| **FastAPI** | Python으로 API 서버를 만드는 프레임워크 | `app = FastAPI()` |
| **Uvicorn** | FastAPI 앱을 실행하고 HTTP 요청을 받는 서버 | `python -m uvicorn main:app --reload` |
| **HTTP** | 클라이언트와 서버가 요청·응답을 주고받는 통신 규칙 | GET 요청 → 200 응답 |
| **Client(클라이언트)** | 서버에 요청을 보내는 쪽 | 브라우저, `/docs`, 다른 프로그램 |
| **Server(서버)** | 요청을 받아 처리하고 결과를 돌려주는 쪽 | 실행 중인 FastAPI 앱 |
| **Endpoint(엔드포인트)** | 특정 기능을 호출하는 Method와 URL 경로의 조합 | `GET /api/knowledge/2` |
| **Route(라우트)** | 요청의 Method·Path를 처리 함수와 연결하는 규칙 | `@app.get('/health')` |
| **Method(메서드)** | 요청에서 수행할 동작의 종류 | GET, POST, PUT, DELETE |
| **Path(경로)** | 서버 안에서 접근할 자원의 주소 | `/api/knowledge/2` |
| **Path Parameter** | URL 경로 안에 들어가는 값 | `{knowledge_id}`에 `2` 전달 |
| **Query Parameter** | URL의 `?` 뒤에 붙는 조회 조건 | `?category=api` |
| **Request Body** | 생성·수정할 때 요청 본문으로 보내는 데이터 | `title`, `content`, `category` |
| **Header** | 요청·응답의 부가 정보 | `Content-Type: application/json` |
| **Response(응답)** | 서버가 돌려주는 상태코드와 결과 | `200` + JSON |
| **Status Code(상태코드)** | 요청 처리 결과를 나타내는 숫자 | 200, 201, 204, 404, 422 |
| **JSON** | 데이터를 키와 값으로 표현하는 텍스트 형식 | `{"id": 1, "title": "가상환경"}` |
| **CRUD** | 생성(Create)·조회(Read)·수정(Update)·삭제(Delete) | POST·GET·PUT·DELETE |
| **Pydantic** | 입력값의 자료형·필수 여부·길이 등을 검증하는 도구 | `BaseModel`, `Field` |
| **Schema(스키마)** | 데이터가 어떤 필드와 규칙을 가져야 하는지 정한 구조 | `KnowledgeWrite` |
| **Validation(검증)** | 입력이 정해진 형식과 조건을 만족하는지 확인 | 빈 `content` → 422 |
| **Persistence(영속성)** | 서버를 껐다 켜도 저장된 데이터가 남는 성질 | `knowledge.json` 저장 후 재시작 |

## 3. CRUD와 HTTP Method

| 작업 | Method | 요청 경로 | 정상 응답 | 데이터 변화 |
|---|---|---|---|---|
| 목록 조회 | GET | `/api/knowledge` | 200 | 없음 |
| 상세 조회 | GET | `/api/knowledge/{knowledge_id}` | 200 | 없음 |
| 새 지식 등록 | POST | `/api/knowledge` | 201 | 새 항목 추가, `next_id` 증가 |
| 기존 지식 수정 | PUT | `/api/knowledge/{knowledge_id}` | 200 | 같은 ID의 세 필드 교체 |
| 지식 삭제 | DELETE | `/api/knowledge/{knowledge_id}` | 204 | 해당 항목만 제거 |

- **GET**은 데이터를 읽기만 한다.
- **POST**는 서버가 새 ID를 발급한다.
- **PUT**은 이 수업에서 `title`, `content`, `category`를 **모두** 보내야 하며, ID와 항목 수는 그대로다.
- **DELETE** 성공 시 **204 응답 본문은 비어 있다**.
- **확장: PATCH**는 바꿀 필드만 보내는 부분 수정 방식으로 소개되지만, 오늘 필수 구현 대상은 아니다.

## 4. HTTP 상태코드 암기표

| 코드 | 뜻 | 이번 수업에서 나타나는 상황 |
|---|---|---|
| **200 OK** | 정상 처리 | GET 조회, PUT 수정 |
| **201 Created** | 새 자원 생성 | POST 등록 성공 |
| **204 No Content** | 정상 처리, 응답 본문 없음 | DELETE 성공 |
| **404 Not Found** | 대상이 없음 | 존재하지 않는 지식 ID |
| **405 Method Not Allowed** | 해당 경로에서 지원하지 않는 Method | 잘못된 Method 요청 |
| **422 Unprocessable Entity** | 입력 검증 실패 | 정수가 아닌 ID, 필수 필드 누락, 길이 오류 |
| **500 Internal Server Error** | 서버 내부 처리 오류 | 파일 접근·처리 실패 등 |

**헷갈리기 쉬운 차이:** `/api/knowledge/9999`는 ID가 없어 **404**, `/api/knowledge/foo`는 `knowledge_id: int`에 맞지 않아 **422**.

## 5. Path · Query · Body 구분

```http
GET /api/knowledge/2
```
- **Path Parameter:** `2`는 조회 대상의 ID.

```http
GET /api/knowledge?category=api
```
- **Query Parameter:** `category=api`는 목록 필터 조건. 생략하면 전체 목록.

```http
POST /api/knowledge
Content-Type: application/json

{"title":"실습 환경 안내","content":"가상환경을 사용한다.","category":"setup"}
```
- **Body:** 새로 저장할 지식의 필드 값. **ID는 보내지 않고 서버가 발급**한다.

## 6. Python·FastAPI 코드 표현 해석

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Knowledge API")

class KnowledgeWrite(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)

@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    # 요청 경로의 ID를 정수로 받아 처리한다.
    ...
```

| 표현 | 의미 |
|---|---|
| `from ... import ...` | 다른 모듈의 기능 가져오기 |
| `FastAPI(...)` | API 앱 객체 생성 |
| `@app.get(...)` | GET 요청 경로를 함수에 등록하는 데코레이터 |
| `def 함수명(...):` | 요청을 처리할 Python 함수 정의 |
| `knowledge_id: int` | ID를 정수형으로 받겠다는 타입 선언 |
| `BaseModel` | 입력·응답 데이터 구조를 정의하는 Pydantic 기반 클래스 |
| `Field(min_length=..., max_length=...)` | 문자열 길이 등의 검증 조건 |
| `payload: KnowledgeWrite` | 요청 본문을 지정한 모델로 검증해 받음 |
| `payload.model_dump()` | Pydantic 모델을 Python `dict`로 변환; 저장까지 하지는 않음 |
| `response_model=Knowledge` | 응답 데이터의 구조를 검증·필터링 |
| `raise HTTPException(404, ...)` | HTTP 오류를 발생시킴 (`return`과 다름) |
| `return` | 처리 결과를 반환 |

### 이번 수업의 입력 규칙

| 필드 | 타입 | 허용 길이 | 필수 여부 |
|---|---|---|---|
| `title` | `str` | 1~80자 | 필수 |
| `content` | `str` | 1~2,000자 | 필수 |
| `category` | `str` | 1~30자 | 필수 |
| `id` | `int` | 서버 발급 | POST·PUT 입력에는 없음 |

`KnowledgeWrite`는 입력 모델이고, `Knowledge`는 여기에 `id`를 더한 응답 모델이다. **형식 검증에 성공했다고 내용의 사실성이나 수정 권한까지 보장되는 것은 아니다.**

## 7. JSON 저장 구조와 ID 관리

```json
{
  "next_id": 4,
  "entries": [
    {"id": 1, "title": "가상환경", "category": "setup", "content": "프로젝트별 패키지를 분리한다."},
    {"id": 2, "title": "서버 실행", "category": "api", "content": "uvicorn으로 개발 서버를 켠다."},
    {"id": 3, "title": "입력 오류", "category": "api", "content": "422이면 입력값을 확인한다."}
  ]
}
```

| 요소 | 의미 |
|---|---|
| `entries` | 지식 항목을 담은 목록(`list`) |
| `entry` | 목록에 들어 있는 지식 한 건(`dict`) |
| `id` | 지식 항목을 구별하는 고유 번호 |
| `next_id` | 다음 POST에서 발급할 번호; API 응답에서 제외 |
| `index` | 리스트 안의 현재 위치(0부터 시작) |

**중요:** `id`와 `index`는 다르다. ID 2를 삭제해도 ID 3은 3 그대로다. 따라서 `len(entries)+1`을 새 ID로 사용하지 않고, `next_id`를 별도로 관리한다. 삭제·수정할 때도 `next_id`는 감소하지 않는다.

## 8. 저장 관련 함수·자료구조

| 표현 | 역할 |
|---|---|
| `read_data()` | JSON 파일을 읽어 Python 자료구조로 반환 |
| `write_data(data)` | 변경한 데이터를 JSON 파일에 저장 |
| `json.load(f)` | JSON 파일 → Python 객체 |
| `json.dump(data, f, ...)` | Python 객체 → JSON 파일 |
| `data["entries"]` | 지식 항목 목록 가져오기 |
| `entry["id"] == knowledge_id` | 요청한 ID와 항목의 ID 비교 |
| `.append(entry)` | 목록 끝에 새 항목 추가 |
| `enumerate(entries)` | 각 항목의 index와 값을 함께 가져오기 |
| `entries[i] = updated` | 기존 위치의 항목 교체 |
| `.pop(i)` | 특정 index의 항목 제거 |
| `with LOCK:` | 같은 프로세스 안에서 읽기·변경·저장 작업 보호 |
| `Path(__file__).resolve().parent` | 현재 파일을 기준으로 경로 계산 |
| 임시 파일 작성 후 `replace()` | 파일을 먼저 임시 경로에 쓰고 실제 파일과 교체 |

**주의:** 이 수업의 `LOCK`은 한 서버 프로세스의 동시 접근을 보호하기 위한 것이며, 여러 서버 프로세스나 DB 트랜잭션까지 보장하지 않는다. 파일 읽기 실패 시 빈 목록으로 간주해 덮어쓰면 안 된다.

## 9. 개발환경·실행 관련 용어

| 용어 | 설명 |
|---|---|
| **가상환경(`venv`)** | 프로젝트별 Python 패키지를 분리하는 환경 |
| **`pip`** | Python 패키지 설치 도구 |
| **`main.py`** | 기본 구조에서 FastAPI 앱과 API 처리 함수를 두는 파일 |
| **`storage.py`** | JSON 읽기·쓰기·잠금 등 저장 기능을 두는 파일 |
| **`/health`** | 앱이 기본 요청에 응답하는지 확인하는 경로 |
| **`/docs`** | API 문서를 보고 요청을 직접 시험하는 화면 |
| **`127.0.0.1`** | 내 컴퓨터를 가리키는 로컬 주소 |
| **`8000`** | 예제 서버가 사용하는 포트 번호 |
| **`--reload`** | 개발 중 코드 변경 시 서버를 자동 재시작하는 옵션 |

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install fastapi uvicorn
python -m uvicorn main:app --reload
```

- `main:app`은 `main.py` 안의 `app` 객체를 뜻한다.
- 브라우저에서 `http://127.0.0.1:8000/docs`를 열어 GET·POST·PUT·DELETE를 테스트한다.
- **Try it out**으로 입력을 준비하고 **Execute**로 요청한다. 실제 결과는 **Server response**의 상태코드와 본문을 확인한다.
- 서버 종료는 실행 중인 터미널에서 `Ctrl+C`.

## 10. 확장 개념: 코드 구조 분리

> 이 절은 원본 강의의 **심화** 범위이며 기본필수 구현에 요구되지 않는다.

| 구성요소 | 담당 역할 |
|---|---|
| **`main.py`** | 앱 생성 및 Router 등록 |
| **Router** | HTTP 경로·Method·응답 처리 |
| **Schema** | 요청·응답의 필드와 검증 규칙 |
| **Repository** | 실제 데이터 조회·변경·저장 |
| **`APIRouter`** | 관련 경로를 하나로 묶는 객체 |
| **`include_router()`** | Router를 FastAPI 앱에 연결 |
| **Template(Jinja2)** | JSON 데이터를 HTML 화면으로 표시하는 기능 |
| **TestClient / pytest** | 요청 결과와 저장 상태를 자동 검사하는 테스트 도구 |

```text
main.py → Router → Repository → knowledge.json
             ↕
           Schema (데이터 규칙)
```

- `APIRouter(prefix="/api/knowledge", tags=["knowledge"])`에서 `prefix`는 경로 앞부분, `tags`는 `/docs`에서 표시할 그룹 이름이다.
- 기본필수는 `main.py + storage.py + data/knowledge.json` 구조만 이해해도 된다.

## 11. 후속 학습과 연결되는 개념

| 용어 | 이번 강의에서의 위치 |
|---|---|
| **PostgreSQL** | 이후 JSON 파일 저장을 대체·확장할 관계형 DB |
| **SQL / JOIN / ERD / Transaction** | 다음 DB 수업에서 다룰 데이터 조작·관계·일관성 개념 |
| **RAG** | 원본 지식을 바탕으로 관련 근거를 검색하고 LLM 답변에 활용하는 후속 기능 |
| **LLM** | 검색한 근거를 사용해 답변을 생성할 수 있는 언어 모델 |
| **Embedding / Vector DB** | 이후 의미 기반 검색에 활용할 표현·저장 기술 |
| **AI Agent** | 필요 작업을 판단하고 도구 사용을 제안·연결하는 후속 시스템 |
| **Tool** | Agent가 이용하도록 별도로 정의·연결하는 기능; API가 자동으로 Tool이 되는 것은 아님 |

**구분 암기:** 지식 관리 API = 원본 관리 / RAG = 근거 검색·답변 생성 / Agent = 도구 선택·작업 흐름.

## 12. 시험·실습 직전 10문장 암기

1. FastAPI는 **요청 경로 연결·입력 검증·응답**을 담당한다.
2. Uvicorn은 **FastAPI 앱을 실행하는 서버**다.
3. CRUD는 **Create·Read·Update·Delete**다.
4. GET=조회, POST=생성, PUT=수정, DELETE=삭제다.
5. Path는 **대상 ID**, Query는 **조회 조건**, Body는 **보낼 데이터**다.
6. 정상 상태코드는 조회·수정 **200**, 생성 **201**, 삭제 **204**다.
7. 없는 ID는 **404**, 잘못된 입력은 기본 **422**다.
8. Pydantic은 **입력 형식**을 검증하며, 내용의 사실성까지 확인하지 않는다.
9. 새 ID는 `len(entries)+1`이 아니라 **`next_id`**로 관리한다.
10. 응답만 확인하지 말고 **JSON 파일과 서버 재시작 후 상태**까지 확인한다.

## 13. 스스로 확인하는 질문

- `GET /api/knowledge/2`가 실행될 때 Uvicorn·FastAPI·처리 함수·JSON 파일은 각각 무엇을 하는가?
- `POST` 요청에서 `id`를 직접 보내지 않는 이유는?
- `PUT`에 `category`를 빠뜨리면 어떤 상태코드가 나오는가?
- `DELETE`가 성공했을 때 왜 응답 본문이 비어 있는가?
- `id`와 `index`는 왜 같은 값으로 취급하면 안 되는가?
- 서버를 재시작한 뒤 수정한 데이터가 남는지 어떻게 확인하는가?
- 오늘 만든 지식 관리 API와 RAG·AI Agent는 어떻게 다른가?

---

*이 문서는 업로드된 Day 17 강의자료를 바탕으로 핵심 용어와 기본 개념을 학습용으로 재구성했다. 확장 항목은 원본 강의에서 응용·심화 또는 후속 학습으로 소개된 내용이다.*
