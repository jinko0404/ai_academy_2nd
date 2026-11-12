# Day 19 | FastAPI–PostgreSQL 핵심 개념·용어 총정리

> **학습 자료:** `Day19_20261112.md` (2026-11-12 오후 수업)  
> **범위:** INTRO, PM 01~05, RAG/Agent, 부록 A01~A19  
> **정리 원칙:** 원본 교안에 등장한 개념과 정책을 중심으로 설명한다. 실제 PostgreSQL 환경에서의 실행 성공 여부는 교안에서도 **미검증**으로 명시한다.

## 0. 수업 목표와 한눈에 보는 구조

**핵심 목표:** `/docs`에서 요청한 지식의 생성·조회·수정·삭제가 PostgreSQL의 **같은 ID를 가진 실제 행**에 반영되는지 확인한다.

```text
사용자 / Swagger UI(/docs)
  ↓ HTTP 요청 (JSON)
FastAPI (Route / Endpoint)
  ↓ Pydantic으로 입력 검증
처리 함수 (비즈니스 로직)
  ↓ SQLAlchemy ORM + Session
Engine / Connection Pool → Psycopg 드라이버
  ↓ SQL 통신
PostgreSQL (categories, knowledge 테이블)
  ↓ commit 후 결과
FastAPI → HTTP 상태 코드 + JSON 응답
```

- **오전:** FastAPI → JSON 파일, pgAdmin → PostgreSQL을 각각 실습.
- **오후:** FastAPI → ORM/Session → PostgreSQL을 하나로 연결.
- **오늘 구현하지 않은 것:** 실제 LLM 답변 생성, 벡터 검색, RAG 인덱싱, Agent 도구 자동 등록, 인증·권한 시스템.
- **핵심 검증:** `POST 응답 id = SQL 조회 knowledge.id = GET 재조회 id`.

---

## 1. 웹·API 기본 용어

| 용어 | 쉬운 뜻 | 수업에서의 예 |
|---|---|---|
| **API** (Application Programming Interface) | 프로그램끼리 요청·응답하는 약속 | 지식 관리 API |
| **HTTP** | 웹에서 요청과 응답을 주고받는 통신 규칙 | GET, POST, PUT, DELETE |
| **Client** (클라이언트) | 서버에 요청하는 쪽 | `/docs`에서 실행하는 사용자 |
| **Server** (서버) | 요청을 받아 처리하는 쪽 | FastAPI 앱 |
| **FastAPI** | Python으로 API 서버를 만드는 프레임워크 | `app = FastAPI(...)` |
| **Endpoint** (엔드포인트) | 호출할 수 있는 API의 특정 주소·동작 | `POST /api/knowledge` |
| **Route** (라우트) | URL·HTTP 메서드를 처리 함수와 연결하는 규칙 | `@app.post(...)` |
| **Path** | URL의 경로 | `/api/knowledge/17` |
| **Path Parameter** | URL 경로에 포함된 값 | `{knowledge_id}` |
| **Query Parameter** | URL의 `?` 뒤에 전달하는 조회 조건 | `?category=learning` (응용) |
| **Request** (요청) | 클라이언트가 서버에 보내는 정보 | POST + JSON Body |
| **Response** (응답) | 서버가 돌려주는 상태와 데이터 | 201 + 새 지식 JSON |
| **Request Body** | 요청 본문 | `title`, `content`, `category` |
| **JSON** | 키와 값으로 데이터를 표현하는 형식 | `{"title":"실습 안내"}` |
| **Swagger UI** | API를 브라우저에서 시험하는 화면 | `http://127.0.0.1:8001/docs` |
| **API Contract** (계약) | 경로·입력·출력·상태코드에 대한 약속 | POST는 201, DELETE는 204 |
| **Uvicorn** | FastAPI 앱을 실행하는 ASGI 서버 | `python -m uvicorn main:app --port 8001 --reload` |
| **ASGI** | Python 웹 서버와 앱 사이의 인터페이스 규격 | Uvicorn으로 FastAPI 실행 |
| **CRUD** | Create·Read·Update·Delete | 등록·조회·수정·삭제 |

### HTTP 메서드와 이번 교안의 계약

| 메서드 | 경로 | 의미 | 정상 상태 |
|---|---|---|---|
| GET | `/api/knowledge` | 전체 목록을 ID 오름차순 조회 | **200**, 빈 목록이면 `[]` |
| GET | `/api/knowledge/{knowledge_id}` | 특정 ID 하나 조회 | **200** |
| POST | `/api/knowledge` | 새 지식 생성 | **201** + DB가 발급한 ID |
| PUT | `/api/knowledge/{knowledge_id}` | 기존 지식의 입력 3개 필드 전체 교체 | **200**, 기존 ID 유지 |
| DELETE | `/api/knowledge/{knowledge_id}` | 해당 지식 삭제 | **204**, 응답 본문 없음 |

**중요:** 이 교안에서 PUT은 `title`, `content`, `category` **세 필드 모두 필수**다. 일부만 보내면 422이다. DELETE 204는 `{}`나 `null`조차 반환하지 않는다.

### 자주 나오는 HTTP 상태 코드

| 코드 | 뜻 | 이 수업에서의 상황 |
|---|---|---|
| **200 OK** | 정상 처리 | 조회·수정 성공 |
| **201 Created** | 생성 성공 | POST가 DB commit까지 성공 |
| **204 No Content** | 성공, 응답 본문 없음 | 삭제 성공 |
| **404 Not Found** | 대상 없음 | 없는 지식 ID |
| **409 Conflict** | 충돌 | 교안에서 지정한 알려진 UNIQUE/FK 충돌 |
| **422 Unprocessable Entity** | 요청 입력 검증 실패 | 누락·길이 초과·잘못된 Path 형식 |
| **500 Internal Server Error** | 예상치 못한 서버·DB 오류 | 예상 밖 CHECK/SQL/설정 오류 등 |
| **503 Service Unavailable** | DB 사용 불가 | 확인된 연결 실패·시간초과 등 |

> **주의:** 409·503 매핑은 교안에서 정한 오류 처리 정책이다. 모든 DB 오류가 자동으로 409나 503이 되는 것은 아니다.

---

## 2. PostgreSQL·관계형 데이터베이스 기본 용어

| 용어 | 쉬운 뜻 | 이번 수업 예 |
|---|---|---|
| **DB** (Database) | 체계적으로 저장된 데이터 집합 | `day19_crud_b` |
| **DBMS** | DB를 관리하는 소프트웨어 | PostgreSQL |
| **RDBMS** | 테이블 간 관계를 사용하는 DBMS | PostgreSQL |
| **Table** (테이블) | 같은 구조의 데이터를 담는 표 | `categories`, `knowledge` |
| **Row / Record** (행) | 데이터 한 건 | 지식 한 개 |
| **Column / Field** (열·필드) | 데이터의 속성 | `title`, `content` |
| **Schema** (DB 스키마) | DB 객체를 묶는 이름 공간·구조 | `public.knowledge`의 `public` |
| **SQL** | DB를 조회·변경하는 언어 | `SELECT`, `INSERT` |
| **DDL** | 테이블 구조를 정의하는 SQL | `CREATE TABLE`, `ALTER TABLE` |
| **DML** | 행을 조회·변경하는 SQL | SELECT·INSERT·UPDATE·DELETE |
| **PK** (Primary Key) | 행을 유일하게 식별하는 기본 키 | `knowledge.id` |
| **FK** (Foreign Key) | 다른 테이블의 키를 참조하는 외래 키 | `knowledge.category_id` |
| **JOIN** | 관계를 따라 여러 테이블의 정보를 합쳐 조회 | 지식에 분류 이름 붙이기 |
| **Constraint** (제약조건) | 잘못된 데이터 저장을 막는 DB 규칙 | PK·FK·CHECK·UNIQUE |
| **NOT NULL** | NULL 값 금지 | 제목·본문 필수 |
| **UNIQUE** | 중복 값 금지 | `categories.name` |
| **CHECK** | 지정한 조건을 만족해야 저장 가능 | 제목 1~80자 |
| **IDENTITY** | DB가 ID 값을 자동 발급 | `GENERATED ALWAYS AS IDENTITY` |
| **Sequence** (시퀀스) | 번호를 발급하는 DB 객체 | rollback 후에도 ID 번호가 건너뛸 수 있음 |
| **RESTRICT** | 참조 중인 부모 삭제를 제한 | 지식이 있는 분류 삭제 제한 |
| **NULL** | 값이 없음을 나타내는 DB 값 | 선택적인 `source_label` |
| **VARCHAR(n)** | 최대 n자 문자열 자료형 | `VARCHAR(30)` |
| **TEXT** | 문자열 본문 자료형 | `content` |
| **BIGINT** | 큰 정수 자료형 | `id`, `category_id` |
| **pgAdmin** | PostgreSQL을 관리·조회하는 GUI 도구 | Query Tool에서 SQL 실행 |

### 두 테이블의 관계

```text
public.categories                    public.knowledge
------------------                   -------------------------
id (PK) ──────────────────────────── category_id (FK)
name (UNIQUE)                        id (PK)
                                     title
                                     content
```

- `categories.id = 1`, `categories.name = 'learning'`이라면 `knowledge.category_id = 1`은 해당 분류를 가리킨다.
- **PK는 식별**, **FK는 참조 관계 보호**, **JOIN은 연결된 데이터 조회**다. 서로 대체하는 기능이 아니다.
- `knowledge.title`은 **중복 허용**이고, `categories.name`은 **중복 금지**다.
- `category_id`는 DB 내부 숫자이며 API에서는 `category`라는 문자열을 받거나 돌려준다.

### 주요 SQL 문법

```sql
-- 조회
SELECT id, title FROM public.knowledge ORDER BY id;

-- 조건 조회
SELECT * FROM public.knowledge WHERE id = 2;

-- 분류명 함께 조회
SELECT k.id, k.title, c.name AS category
FROM public.knowledge AS k
JOIN public.categories AS c ON c.id = k.category_id;

-- 추가 (DB가 새 ID를 발급)
INSERT INTO public.knowledge (title, content, category_id)
VALUES ('실습 안내', '가상환경을 확인합니다.', 1)
RETURNING id;

-- 수정
UPDATE public.knowledge SET content = '수정된 안내' WHERE id = 2;

-- 삭제
DELETE FROM public.knowledge WHERE id = 2;
```

**SQL 키워드 미니 사전:** `SELECT` 조회, `FROM` 대상 테이블, `WHERE` 조건, `ORDER BY` 정렬, `JOIN` 결합, `ON` 결합 조건, `AS` 별칭, `INSERT INTO` 추가, `VALUES` 입력값, `RETURNING` 변경 후 값 반환, `UPDATE` 수정, `SET` 수정할 값, `DELETE` 삭제, `COUNT` 개수, `LEFT JOIN` 왼쪽 행 보존 결합, `IS NULL` NULL 검사, `GROUP BY` 그룹화, `HAVING` 그룹 조건.

---

## 3. Python·FastAPI 코드 용어

| 용어 | 의미 | 코드에서 보이는 모습 |
|---|---|---|
| **Module** | Python 코드 파일 | `main.py`, `db.py` |
| **Import** | 다른 모듈의 기능 가져오기 | `from sqlalchemy import select` |
| **Class** | 데이터와 동작을 정의하는 틀 | `class Knowledge(Base)` |
| **Object / Instance** | 클래스로 만든 실제 객체 | `Knowledge(title=...)` |
| **Function** | 호출 가능한 처리 코드 | `def create_knowledge(...)` |
| **Parameter** | 함수가 받는 입력 변수 | `payload`, `db` |
| **Return** | 함수 결과 반환 | `return result` |
| **Decorator** | 함수에 경로·동작 등의 의미를 붙임 | `@app.post(...)` |
| **Type Hint** | 값의 타입 표시 | `knowledge_id: int` |
| **Annotated** | 타입에 추가 메타데이터 부여 | `Annotated[Session, Depends(get_db)]` |
| **Dependency Injection** (의존성 주입) | 필요한 객체를 프레임워크가 준비·전달 | `Depends(get_db)` |
| **yield** | 값을 제공하고 나중에 정리 흐름을 이어감 | Session 생성·종료 |
| **Context Manager** | `with` 블록 전후에 준비·정리 실행 | `with transaction(db):` |
| **Exception** (예외) | 처리 중 발생한 오류 | `IntegrityError` |
| **raise** | 예외를 발생시키거나 다시 전달 | `raise HTTPException(...)` |
| **Response Model** | 응답에 사용할 데이터 형식 | `response_model=KnowledgeRead` |
| **`/health`** | 앱 응답 확인용 경로 | `{"status":"ok"}` |

**구분:** `/health`가 200을 반환해도 DB 연결과 테이블까지 정상이라는 보장은 없다.

---

## 4. Pydantic · API 입력/응답 검증

**Pydantic:** API에서 들어오거나 나가는 데이터의 타입·필수 여부·길이 등을 검사하고 구조화하는 도구.

| 용어 | 뜻 | 예 |
|---|---|---|
| **BaseModel** | Pydantic 데이터 모델의 기반 | `class KnowledgeWrite(BaseModel)` |
| **Field** | 필드에 검증 조건 설정 | `Field(min_length=1, max_length=80)` |
| **Validation** | 데이터가 규칙에 맞는지 확인 | 빈 본문 → 422 |
| **KnowledgeWrite** | POST·PUT 요청용 입력 모델 | title·content·category |
| **KnowledgeRead** | 응답용 모델 | 입력 3개 + id |
| **Serialization** | 객체를 응답 가능한 형태로 변환 | `KnowledgeRead(...)` |
| **`from_attributes=True`** | 객체 속성에서 값 읽기 허용 | 내부 FK를 분류 문자열로 자동 변환하지는 않음 |

```python
class KnowledgeWrite(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)

class KnowledgeRead(KnowledgeWrite):
    id: int
```

### 복사해서 사용할 수 있는 POST/PUT Body

```json
{
  "title": "실습 환경 안내",
  "content": "실행 전에 가상환경을 확인합니다.",
  "category": "learning"
}
```

**오전 실습과의 차이:** 오전 JSON 파일 API에서는 `category_id`를 사용한 예시가 있었지만, **이 오후 교안의 API 계약은 `category` 문자열**이다. DB에 저장할 때 해당 분류를 찾아 `category_id`로 변환한다.

**이번 계약의 세부 규칙:** 필드 누락·NULL·길이 위반은 422, 추가 입력 필드는 기본 Pydantic 정책에 따라 무시, 공백만 있는 문자열은 길이 조건상 허용. 이 규칙을 임의로 강화하지 않는다.

---

## 5. SQLAlchemy · ORM과 DB 연결

**SQLAlchemy:** Python에서 SQL과 DB 작업을 다루는 라이브러리. 이 수업에서는 ORM과 Session을 사용한다.

| 용어 | 정확한 역할 | 기억할 점 |
|---|---|---|
| **ORM** (Object-Relational Mapping) | Python 클래스·객체와 DB 테이블·행을 연결 | 클래스 = 테이블, 객체 = 행에 대응 |
| **ORM Model** | 테이블의 열·관계를 Python으로 선언한 클래스 | `Category`, `Knowledge` |
| **DeclarativeBase / Base** | ORM 클래스 선언의 공통 기반 | `class Base(DeclarativeBase)` |
| **Metadata** | ORM이 알고 있는 테이블·열 등 구조 정보 | 실제 DB를 자동 변경하지 않음 |
| **Mapped** | ORM 속성 타입 선언 | `id: Mapped[int]` |
| **mapped_column** | Python 속성과 DB 열을 매핑 | `mapped_column(BigInteger, ...)` |
| **ForeignKey** | 외래 키 관계 선언 | `public.categories.id` |
| **CheckConstraint** | CHECK 제약 선언 | `char_length(title) BETWEEN 1 AND 80` |
| **Engine** | DB 연결을 얻고 재사용하기 위한 기반 | 앱 전체에서 재사용 |
| **Connection Pool** | DB 연결을 보관·재사용하는 관리 기능 | 매 요청 새 연결을 무조건 만들지 않음 |
| **Driver** | DB 서버와 실제 통신하는 라이브러리 | Psycopg |
| **Psycopg** | Python ↔ PostgreSQL 드라이버 | `postgresql+psycopg` |
| **Session** | ORM 객체·SQL·트랜잭션을 관리하는 작업 공간 | 요청마다 생성·정리 |
| **sessionmaker** | Session을 생성하는 팩토리 | `SessionLocal()` |
| **`URL.create`** | DB 접속 URL을 안전하게 구성 | 비밀번호 특수문자 처리 |
| **`create_engine`** | Engine 생성 | 생성만으로 DB 접속 성공 확인 불가 |
| **`pool_pre_ping`** | 재사용 연결의 생존 여부 확인 | 모든 장애 해결 기능은 아님 |
| **`select`** | ORM 조회 문 구성 | `select(Knowledge)` |
| **`where`** | 조건 지정 | `.where(Category.name == name)` |
| **`order_by`** | 결과 정렬 | `.order_by(Knowledge.id)` |
| **`execute`** | SQL 문 실행 | `db.execute(stmt)` |
| **`scalar`** | 결과에서 단일 스칼라 값/객체를 받는 방식 | `db.scalar(select(Category)...)` |
| **`get`** | 기본 키로 ORM 객체 조회 | `db.get(Knowledge, knowledge_id)` |

### ORM Model과 Pydantic Schema를 헷갈리지 말 것

| 구분 | ORM Model | Pydantic Schema |
|---|---|---|
| 목적 | DB 저장 구조와 매핑 | API 입력·출력 검증 |
| 예시 | `Knowledge` | `KnowledgeWrite`, `KnowledgeRead` |
| 분류 표현 | `category_id: int` | `category: str` |
| 규칙 | PK·FK·CHECK·자료형 | 필수 값·길이·응답 필드 |
| 주의 | 선언만으로 기존 DB 열 변경 안 됨 | FK 숫자를 이름으로 자동 치환 안 함 |

### 연결 설정의 구성 요소

Git에 공개할 문서에는 DB 계정·비밀번호·접속 설정값의 예시를 포함하지 않습니다.

- **`.gitignore`:** Git에 포함하지 않을 파일을 지정합니다.
- **`pydantic-settings` / `BaseSettings`:** 환경변수 등을 설정 객체로 읽습니다.
- **`SecretStr`:** 출력에서 비밀번호를 가리는 타입으로, 암호화 저장 기능은 아닙니다.
- **Host:** DB 서버 위치, **Port:** 서비스 접속 번호, **DB_NAME:** 접속할 데이터베이스 이름.
- **API 포트 8001 ≠ PostgreSQL 포트 5432.**
- **`pool_pre_ping=True`:** 풀의 연결 상태 점검.
- **`hide_parameters=True`:** SQL 오류·로그에서 바인드 파라미터 노출을 줄이기 위한 설정.
- **`connect_timeout=5`:** 연결 시도 시간 제한(초).
- **`autoflush=False`:** 자동 flush 동작을 비활성화한 수업 설정.
- **`expire_on_commit=False`:** commit 뒤 ORM 객체 속성을 자동 만료시키지 않는 수업 설정.

### 요청마다 Session을 만드는 이유

```python
SessionLocal = sessionmaker(bind=engine, autoflush=False,
                            expire_on_commit=False)

def get_db():
    with SessionLocal() as db:
        yield db

DbSession = Annotated[Session, Depends(get_db)]
```

**Engine은 공유·재사용하지만 Session의 작업 상태는 요청별로 분리**한다. `yield`로 전달하고 요청 처리 후 Session을 정리한다. `close()`는 `commit()`이 아니다.

---

## 6. CRUD 처리 흐름과 주요 메서드

### Create — POST

```text
POST Body 입력
→ Pydantic 검증
→ category 문자열로 기존 분류 조회
→ 없으면 Category 객체 추가
→ db.flush()로 category.id 확보
→ Knowledge 객체 추가
→ db.flush()로 knowledge.id 확보
→ db.commit() 성공
→ 201 + id 반환
```

- `get_or_create_category`: 기존 분류면 ID 재사용, 없으면 생성.
- 분류 생성과 지식 생성은 **한 트랜잭션**으로 묶는다.
- ID는 PostgreSQL이 발급하며 JSON 파일 방식의 `next_id`를 직접 관리하지 않는다.

### Read — GET

- 목록: `select(Knowledge, Category.name).join(...).order_by(Knowledge.id)`로 지식과 분류명 조회.
- 상세: `db.get(Knowledge, knowledge_id)`로 PK 조회.
- 결과가 없으면 목록은 `200 []`, 상세는 `404`.
- `to_read(row, category.name)`로 내부 `category_id`를 외부 분류명으로 표현.

### Update — PUT

- 먼저 해당 지식이 있는지 확인(없으면 404).
- `title`, `content`, `category_id`를 변경.
- **기존 ID와 지식 행 수는 유지**.
- 없는 ID에 PUT해도 새 지식을 만들지 않는다.

### Delete — DELETE

- 먼저 지식 존재 확인 후 `db.delete(row)`.
- commit이 성공하면 204, **본문은 비움**.
- 해당 지식만 삭제하며 부모 `categories` 행은 자동 삭제하지 않는다.

### Session 메서드 구별 (매우 중요)

| 메서드 | 하는 일 | 오해하면 안 되는 점 |
|---|---|---|
| `db.add(obj)` | 객체를 Session에 등록 | 즉시 DB에 최종 저장되는 것은 아님 |
| `db.flush()` | 대기 변경 SQL을 DB로 전송, ID 확보 가능 | **확정 아님** |
| `db.commit()` | 미확정 변경을 최종 확정 | 모든 동시성 문제를 해결하진 않음 |
| `db.refresh(obj)` | DB 값을 다시 객체로 읽기 | 저장 명령이 아님 |
| `db.rollback()` | 현재 트랜잭션의 미확정 변경 취소·실패 상태 정리 | 발급된 시퀀스 번호까지 되돌리지 않음 |
| `db.close()` | Session 자원·상태 정리 | 저장 확정이 아님 |
| `db.delete(obj)` | 객체를 삭제 대상으로 표시 | commit 전 최종 삭제 확정 아님 |

---

## 7. 트랜잭션과 오류 처리

### 트랜잭션 (Transaction)

**여러 DB 변경을 하나의 논리적인 작업으로 묶어 성공하면 함께 확정하고 실패하면 함께 취소하는 범위.**

```text
새 분류 추가 + 새 지식 추가
         ↓
    같은 Transaction
      ├─ 정상 → COMMIT → 둘 다 저장
      └─ 실패 → ROLLBACK → 둘 다 취소
```

- **Atomicity (원자성):** 묶인 작업이 전부 성공하거나 전부 취소되는 성질.
- **COMMIT:** 현재 트랜잭션의 변경 확정.
- **ROLLBACK:** 현재 미확정 변경 취소. 이미 commit한 작업을 되돌리지는 못함.
- **flush:** DB에 SQL을 실행해 ID를 받을 수 있지만 commit과 다름.
- **트랜잭션 경계:** 어디부터 어디까지 하나의 commit/rollback 대상으로 볼지 정한 범위.
- **Context manager:** `with transaction(db):`로 성공·실패 종료를 공통 처리.

```python
from contextlib import contextmanager

@contextmanager
def transaction(db):
    try:
        yield
        db.commit()
    except Exception:
        db.rollback()
        raise
```

이 교안의 도우미에서는 **쓰기 작업 전체에 한 번만 사용**, 내부에서 별도 `commit()`이나 중첩 `transaction()`을 호출하지 않는다.

### 예외·오류 용어

| 용어 | 뜻 | 수업에서의 예 |
|---|---|---|
| **HTTPException** | HTTP 오류 응답을 위한 예외 | `HTTPException(404, "Knowledge not found")` |
| **IntegrityError** | DB 무결성 제약 위반 관련 예외 | UNIQUE·FK 등 |
| **DBAPIError** | DB 드라이버 계열 오류를 감싼 SQLAlchemy 예외 | DB 작업 실패 |
| **OperationalError** | DB 연결·운영 관련 오류 종류 | 연결 실패 후보 |
| **Exception Handler** | 예외를 HTTP 응답으로 변환하는 함수 | `@app.exception_handler(...)` |
| **SQLSTATE** | PostgreSQL 오류 코드 | `23505`, `23503`, `08*` |
| **Constraint Name** | 위반된 제약조건의 이름 | `knowledge_category_fk` |
| **Conflict** | 충돌 상태 | 알려진 제약 위반 → 409 |
| **PendingRollbackError** | 실패한 Session의 트랜잭션을 정리하지 않은 상태에서 후속 사용 시 생길 수 있는 오류 | 먼저 rollback 확인 |
| **Concurrency** (동시성) | 여러 요청이 동시에 같은 데이터에 접근·변경 | 분류 동시 생성 경쟁 |

| SQLSTATE | 뜻 | 교안 정책 |
|---|---|---|
| `23505` | UNIQUE 위반 | 알려진 `categories_name_key`라면 409 |
| `23503` | FK 위반 | 알려진 `knowledge_category_fk`라면 409 |
| `08*` | 연결 계열 오류 | 확인된 연결 실패 시 503 후보 |

**주의:** 빈 `content`를 POST해 422가 나오는 것은 **DB에 쓰기 전에 거절된 사례**다. 이것만으로 DB의 rollback이 검증되지는 않는다. rollback 검증은 별도 DB에서 실제 제약 위반 후 같은 Session의 후속 조회까지 확인한다.

### 디버깅 순서

1. API 포트(8001)와 DB 포트(보통 5432)를 구분한다.
2. 실행 환경의 DB Host·Port·DB_NAME·계정이 pgAdmin과 같은지 확인한다.
3. `current_database()`, `current_user`로 실제 접속 대상을 확인한다.
4. ORM 모델의 열·제약과 실제 테이블 구조를 비교한다.
5. 실패했다면 rollback 여부를 확인한다.
6. POST 응답을 못 받았다고 무조건 재시도하지 말고, 먼저 DB에서 생성 여부를 확인한다.

---

## 8. Migration · Alembic과 스키마 변경

**Migration:** DB의 테이블·열·제약 같은 **구조를 버전별로 변경하고 기록하는 과정**. **Alembic:** SQLAlchemy 프로젝트에서 이 변경 이력을 관리하는 도구.

| 용어 | 뜻 | 수업 예 |
|---|---|---|
| **Migration** | DB 구조 변경 과정 | `source_label` 열 추가 |
| **Alembic** | DB 변경 이력 관리 도구 | `python -m alembic ...` |
| **Revision** | 한 번의 구조 변경을 나타내는 버전 파일·ID | `001_initial`, `002_source` |
| **`revision`** | 현재 변경 버전 ID | `"002_source"` |
| **`down_revision`** | 직전 변경 버전 ID | `"001_initial"` |
| **`upgrade()`** | 다음 구조로 변경하는 코드 | `op.add_column(...)` |
| **`downgrade()`** | 이전 구조로 되돌리는 코드 | `op.drop_column(...)` |
| **`head`** | 현재 이력 계보의 최신 Revision | `upgrade head` |
| **`current`** | DB에 현재 적용된 Revision 확인 | `alembic current` |
| **`history`** | Revision 이력 조회 | `alembic history` |
| **Autogenerate** | 모델과 DB의 차이를 바탕으로 변경 초안 생성 | 반드시 사람이 검토 |
| **Stamp** | DDL 실행 없이 이력 위치만 표시 | DB 구조를 수리하지 않음 |
| **Nullable** | NULL 허용 여부 | `source_label`은 `nullable=True` |
| **`create_all`** | 없는 테이블을 생성하는 기능 | 기존 열 변경·이력 관리의 대체가 아님 |

### 이번 Migration 예제

```text
연습 DB: day19_migration_lab
001_initial → categories, knowledge 생성
       ↓
가상 지식 1건 저장 및 commit
       ↓
002_source → knowledge.source_label VARCHAR(120) NULL 추가
       ↓
기존 지식 1건과 실제 열, Alembic 이력 확인
```

```python
revision = "002_source"
down_revision = "001_initial"

def upgrade():
    op.add_column("knowledge",
        sa.Column("source_label", sa.String(120), nullable=True),
        schema="public")

def downgrade():
    op.drop_column("knowledge", "source_label", schema="public")
```

```bash
python -m alembic current
python -m alembic history
python -m alembic upgrade 001_initial
# 가상 데이터 1건 저장 후
python -m alembic upgrade head
python -m alembic current
```

- **중요:** ORM Model에 열을 추가해도 기존 DB에 자동 반영되지 않는다.
- **중요:** downgrade로 열을 삭제하면 그 열에 저장된 데이터는 잃을 수 있다.
- **중요:** autogenerate는 최종 정답이 아닌 초안이다. 이름 변경을 삭제+추가로 잘못 표현할 수 있다.
- **중요:** `stamp`는 실제 구조 변경을 실행하지 않는다.
- **중요:** 주 프로젝트 DB에서 연습용 Migration을 실행하지 않는다.

### 세 DB의 역할 분리

| DB | 목적 | 주의 |
|---|---|---|
| `day19_crud_b` (기본필수) | 실제 수업 API가 쓰는 기존 지식 DB | 기존 데이터 보존, DROP·재초기화 금지 |
| `day19_migration_lab` | Migration 연습 전용 | 001 → 002, 실제 열과 데이터 확인 |
| `day19_api_test` | 자동 테스트 전용 | 주 DB와 다른 DB인지 실제 접속 확인 |

응용은 `day19_crud_a`, 심화는 `day19_crud_h`를 사용한다. `day19_build_*`는 주 API 연결 대상으로 자동 선택하지 않는다.

---

## 9. 테스트·통합 검증 용어

| 용어 | 뜻 | 이번 수업의 확인 방법 |
|---|---|---|
| **Integration Test** (통합 테스트) | 여러 구성요소가 연결되어 제대로 동작하는지 검사 | API 응답 + 실제 DB 행 대조 |
| **Contract Test** | API가 약속한 입력·출력·상태를 지키는지 검사 | POST 201, PUT 200, DELETE 204 |
| **TestClient** | FastAPI 앱에 테스트 요청을 보내는 도구 | 심화 자동 테스트 |
| **Fixture** | 테스트 전후 환경·데이터를 준비·정리하는 장치 | 전용 DB Session 준비 |
| **Dependency Override** | 앱의 의존성을 테스트용으로 바꿈 | `get_db` → 테스트 DB Session |
| **Assertion** | 결과가 기대와 같은지 확인하는 코드 | `assert response.status_code == 201` |
| **Mock** | 실제 구성요소 대신 사용하는 모의 객체 | Mock 성공 ≠ 실제 PostgreSQL 성공 |
| **Seed Data** | 초기 준비 데이터 | 오전의 기존 지식 |
| **Regression** (회귀) | 변경 후 기존 기능이 망가지는 현상 | 수정·Migration 후 재검증 |
| **Acceptance Criteria** | 과제 완료 판정 기준 | A 유지, B 수정, C 삭제 |
| **Persistence** (영속성) | 프로그램이 재시작돼도 데이터가 남음 | 서버 재시작 후 GET 확인 |

### 일일프로젝트 A/B/C 검증 순서

1. 시작 지식 수 **N**과 실제 DB 이름을 기록한다.
2. A·B·C 세 지식을 POST로 생성한다 → **N+3**, 각 ID 기록.
3. B를 PUT으로 수정한다 → **B의 ID는 그대로**, 본문 변경.
4. 빈 본문 POST → **422**, 없는 ID에 정상 PUT → **404**, 기존 데이터 변화 없음.
5. C를 DELETE한다 → **204**, 이후 C GET은 **404**.
6. 서버를 재시작하고 새 DB 연결로 확인한다 → **A·수정된 B·기존 행 보존**, 최종 **N+2**.

**검증의 3종 증거:** ① HTTP 상태 코드, ② API 응답 JSON, ③ 같은 ID의 실제 DB 행. 이것으로 운영 환경의 모든 장애·백업 복구·동시성 문제가 해결됐다고 주장하지 않는다.

---

## 10. 프로젝트 파일·도구·실행 용어

| 파일·도구 | 역할 |
|---|---|
| `main.py` | FastAPI 앱, 라우트, 입력/응답 모델·작은 도우미(기본필수) |
| `settings.py` | 환경변수 등 설정 읽기 |
| `db.py` | Engine, SessionLocal, `get_db`, `transaction` |
| `models.py` | `Category`, `Knowledge` ORM 클래스 |
| `schemas.py` | API 입력·응답 모델(응용·심화 분리 가능) |
| `routers/` | 경로·HTTP 처리 역할 분리(심화) |
| `services/` | 업무 로직 역할 분리(심화) |
| `repositories/` | DB 접근 역할 분리(심화) |
| `tests/` | 테스트 코드(심화) |
| `migration_demo/` | 별도 Migration 연습 |
| `.gitignore` | Git 추적 제외 규칙 |
| `.venv/` | Python 가상환경 폴더 |
| `pip` | Python 패키지 설치 도구 |
| `pgAdmin Query Tool` | SQL 직접 실행·결과 확인 |
| `--reload` | 개발 중 코드 변경 시 서버 재시작 옵션 |

```bash
# 필요한 패키지 예시
python -m pip install "sqlalchemy>=2.0,<2.1"
python -m pip install "psycopg[binary]>=3,<4"
python -m pip install pydantic-settings alembic

# 기본필수: 프로젝트 최상위에서 실행
python -m uvicorn main:app --port 8001 --reload

# 심화 폴더 구조라면
python -m uvicorn app.main:app --port 8001 --reload
```

- `main:app`은 `main.py`의 `app` 객체를 의미한다.
- 브라우저에서 여는 주소는 `http://127.0.0.1:8001/docs`다.
- 터미널 명령과 브라우저 주소를 구분한다.
- 교안은 Windows 직접 실행과 실제 PostgreSQL 연결·CRUD·Migration을 제작 시점에 검증하지 않았다고 명시한다.

---

## 11. RAG·LLM·Agent 관련 용어와 오늘 수업의 경계

| 용어 | 쉬운 설명 | 오늘 수업과의 관계 |
|---|---|---|
| **LLM** (Large Language Model) | 언어를 이해·생성하는 대규모 모델 | 후속 수업 대상 |
| **RAG** (Retrieval-Augmented Generation) | 외부 자료를 검색해 근거를 제공한 뒤 답변 생성 | 오늘은 원본 지식 저장 기반만 구축 |
| **Retriever** | 질문에 관련된 자료를 찾아오는 구성요소 | 오늘 ORM 조회가 곧 의미 검색은 아님 |
| **Embedding** (임베딩) | 텍스트 등을 벡터로 표현하는 기술 | 후속 과정 |
| **Vector Search** (벡터 검색) | 벡터 간 유사성을 이용한 검색 | 오늘 구현하지 않음 |
| **Index / Indexing** (인덱스·인덱싱) | 검색을 위해 데이터를 별도로 조직·반영 | 원본 DB 변경 후 별도 동기화 필요 |
| **Chunk** (청크) | 검색·처리를 위해 나눈 문서 조각 | 후속 RAG 작업 |
| **AI Agent** | 목표에 따라 도구 등을 선택·실행하는 AI 시스템 | 오늘 API가 자동으로 Agent 도구가 되지는 않음 |
| **Tool** (도구) | Agent가 호출할 수 있는 외부 기능 | API 계약·권한·연결 설정이 별도 필요 |
| **Source / Provenance** (출처) | 정보의 근거·원천 | `source_label` 열 추가만으로 출처 검증 완성 아님 |
| **Authorization** (권한) | 누가 어떤 작업을 수행할 수 있는지 결정 | 오늘 구현 범위 밖 |

```text
[오늘 구현] 원본 지식 관리
담당자 → FastAPI → ORM·Session → PostgreSQL

[이후 별도 구현] 검색·답변
원본 지식 → 동기화·인덱싱 → Retriever → LLM/Agent → 답변·출처
```

**꼭 기억:** PostgreSQL은 원본 지식 저장 계층이다. LLM, Retriever, Agent와 동일한 역할이 아니다.

---

## 12. 헷갈리기 쉬운 개념 20개 비교

| A | B | 핵심 차이 |
|---|---|---|
| FastAPI | PostgreSQL | API 처리 서버 / 데이터 저장 DB |
| SQLAlchemy | Psycopg | ORM·DB 작업 도구 / PostgreSQL 통신 드라이버 |
| Engine | Session | 연결 기반 재사용 / 요청별 작업 상태 |
| ORM Model | Pydantic Model | DB 구조 매핑 / API 데이터 검증 |
| `category` | `category_id` | 외부 문자열 / 내부 FK 숫자 |
| PK | FK | 행 식별 / 다른 테이블 참조 |
| FK | JOIN | 참조 무결성 규칙 / 조회 결합 |
| NOT NULL | CHECK | NULL 금지 / 조건식 검사 |
| UNIQUE | PK | 중복 금지 / 행 식별 기본 키 |
| `add()` | `commit()` | 객체 등록 / 변경 확정 |
| `flush()` | `commit()` | SQL 반영·ID 확인 가능(미확정) / 확정 |
| `rollback()` | `close()` | 미확정 작업 취소 / 자원 정리 |
| `refresh()` | `flush()` | DB → 객체 다시 읽기 / 객체 변경 → DB 실행 |
| 404 | 422 | 대상이 없음 / 입력 형식·값이 잘못됨 |
| 200 | 201 | 일반 성공 / 새 자원 생성 성공 |
| 204 | 200 + `{}` | 본문 없는 성공 / 본문 있는 성공 |
| GET 상세 | GET 목록 | 없는 ID는 404 / 결과 없으면 `[]` |
| POST | PUT | 새 행 생성 / 기존 행 수정(ID 유지) |
| ORM 클래스 변경 | Migration | Python 선언 변경 / 실제 DB 구조 변경 |
| API CRUD | RAG 검색 | 원본 저장·관리 / 관련 문서 검색·생성 연계 |

---

## 13. 필수 암기 체크리스트

- [ ] FastAPI → Pydantic → ORM/Session → PostgreSQL의 흐름을 설명할 수 있다.
- [ ] GET/POST/PUT/DELETE와 200/201/204/404/422를 구별한다.
- [ ] POST·PUT Body는 `title`, `content`, `category`임을 안다.
- [ ] DB의 `category_id`와 API의 `category`가 다른 이유를 안다.
- [ ] PK·FK·JOIN·UNIQUE·CHECK·NOT NULL을 구별한다.
- [ ] Engine은 재사용하고 Session은 요청별로 생성·정리한다.
- [ ] `add`, `flush`, `commit`, `rollback`, `refresh`, `close`를 구별한다.
- [ ] 분류와 지식은 한 트랜잭션으로 확정·취소한다.
- [ ] 422 입력 오류와 실제 DB 오류 rollback 검증은 다르다.
- [ ] ORM Model 변경만으로 기존 DB 열이 바뀌지 않음을 안다.
- [ ] Alembic Revision·upgrade·downgrade·head·autogenerate·stamp를 설명한다.
- [ ] 주 DB, Migration 연습 DB, 테스트 DB를 분리해야 하는 이유를 안다.
- [ ] API 성공 응답과 같은 ID의 SQL 결과를 대조할 수 있다.
- [ ] 오늘의 API가 자동으로 RAG 검색·Agent 도구가 되지는 않음을 안다.

## 14. 스스로 확인하는 핵심 질문 10개

1. `POST /api/knowledge`가 201을 반환하기 전에 반드시 성공해야 할 작업은?
2. `flush()`를 호출한 뒤 `rollback()`하면 새 지식은 남을까?
3. `Session.close()`만 호출해도 변경이 확정될까?
4. `category="learning"`은 DB의 어떤 열과 연결되는가?
5. 없는 ID에 PUT하면 이 교안에서는 새 행을 생성하는가?
6. 목록 조회 결과가 0건이면 상태 코드는 무엇인가?
7. `knowledge.title`과 `categories.name` 중 중복이 금지된 것은?
8. ORM 클래스에 `source_label`을 선언하면 기존 DB에 바로 생기는가?
9. 422 응답을 관찰하는 것만으로 DB rollback이 검증되는가?
10. POST 응답 ID가 실제로 저장됐음을 어떤 두 가지 방법으로 재확인할까?

<details>
<summary>정답 확인</summary>

1. 저장 작업의 **commit 성공**. 2. 미확정 변경은 취소된다. 3. 아니다. 4. 분류명으로 `categories`를 조회·생성해 얻은 `categories.id`를 `knowledge.category_id`에 저장한다. 5. 아니다, 404. 6. 200 + `[]`. 7. `categories.name`. 8. 아니다, Migration 필요. 9. 아니다. 10. 같은 ID로 **SQL 조회**하고 **API GET 재조회**한다.
</details>

---

## 자료 범위·주의사항

이 문서는 업로드한 `Day19_20261112.md`의 **INTRO, PM 01~05, RAG/Agent, 부록 A01~A19**를 학습용으로 재구성한 것이다. 일부 용어의 한국어 풀이와 기초 Python·HTTP 정의는 이해를 돕는 설명이며, 수업의 구체적인 입력·응답·DB·오류·Migration 정책은 원본 교안을 따른다. 특히 **실제 PostgreSQL·Windows·pgAdmin 실행은 원본 교안 제작 당시 미검증**이라고 명시되어 있으므로 실습 환경에서 별도로 확인해야 한다.
