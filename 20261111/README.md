# Day 18 · PostgreSQL 핵심 개념·용어 정리

> 출처: `Day18_PostgreSQL_입문_지식관리DB.md` (2026-11-11 강의자료). 원본의 개념·예시·실습 규칙을 주제별로 재구성한 복습 노트입니다. 아래 SQL의 ID와 결과는 **초기 데이터 S0 기준**이며, 실제 실행 후에는 달라질 수 있습니다.

## 목차
1. [전체 구조와 도구](#1-전체-구조와-도구)
2. [관계형 데이터의 기본 단위](#2-관계형-데이터의-기본-단위)
3. [키·관계·ERD](#3-키관계erd)
4. [자료형과 제약조건](#4-자료형과-제약조건)
5. [SQL 분류와 조회](#5-sql-분류와-조회)
6. [데이터 변경과 RETURNING](#6-데이터-변경과-returning)
7. [JOIN과 집계](#7-join과-집계)
8. [트랜잭션과 오류 복구](#8-트랜잭션과-오류-복구)
9. [인덱스와 실행계획](#9-인덱스와-실행계획)
10. [AI·API 연결 개념](#10-aiapi-연결-개념)
11. [헷갈리는 개념 비교](#11-헷갈리는-개념-비교)
12. [실습용 핵심 SQL 모음](#12-실습용-핵심-sql-모음)

---

## 1. 전체 구조와 도구

| 용어 | 뜻 | 수업 예시·핵심 |
|---|---|---|
| Database (DB) | 관련 데이터와 구조를 관리하는 논리적 공간 | `knowledge_lab` |
| DBMS | 데이터베이스를 관리하고 요청을 처리하는 시스템 | PostgreSQL |
| RDBMS | 테이블과 관계를 중심으로 데이터를 관리하는 DBMS | PostgreSQL |
| PostgreSQL | SQL·제약조건·트랜잭션 등을 지원하는 관계형 DBMS | 지식 원본 저장 |
| Server | DB 요청을 받아 처리하는 서버 | PostgreSQL 서버 |
| Client | 서버에 접속해 요청을 보내는 도구 | pgAdmin, psql |
| pgAdmin | 그래픽 환경에서 PostgreSQL을 관리하는 도구 | Query Tool 사용 |
| psql | 터미널 기반 PostgreSQL 클라이언트 | SQL 및 `\dt` 등 전용 명령 |
| SQL | DB 구조를 정의하고 데이터를 조회·변경하는 언어 | `SELECT`, `INSERT` 등 |
| Query Tool | SQL 실행과 결과 확인을 위한 pgAdmin 기능 | Data Output / Messages |
| Schema | DB 안에서 테이블 등의 이름을 구분하는 논리적 이름 공간 | `public` (디스크 폴더 아님) |
| Host | 접속할 서버 주소 | `localhost`는 로컬 서버 예시 |
| Port | 서버 접속에 사용하는 포트 | `5432`는 기본 예시 |
| User / Password | DB 인증에 사용하는 계정과 비밀번호 | `postgres` / 직접 설정한 비밀번호 |

**구조를 큰 단위부터 읽기**

```text
PostgreSQL Server
└── Database: knowledge_lab
    └── Schema: public
        ├── Table: categories
        └── Table: knowledge
            └── Row → Column → Value
```

- pgAdmin을 닫아도 PostgreSQL 서버가 자동으로 종료되는 것은 아닙니다.
- 서버 등록 이름은 화면의 별칭일 뿐, 그 이름으로 DB가 자동 생성되지 않습니다.
- `SELECT version();`은 서버 버전을 확인합니다. `psql --version`은 클라이언트 버전을 확인합니다.

## 2. 관계형 데이터의 기본 단위

| 용어 | 동의어·설명 | 예시 |
|---|---|---|
| Table | Relation(릴레이션), 행·열로 이루어진 표 | `knowledge` |
| Row | Tuple(튜플), Record(레코드), 데이터 한 건 | `id=2`인 지식 |
| Column | Attribute(속성), Field(필드), 공통 항목 | `title`, `category_id` |
| Value | 행과 열이 만나는 한 칸의 데이터 | `서버 실행` |
| ID | 한 행을 식별하는 값 | `knowledge.id` |
| S0 / Seed | 실습을 시작하기 위해 준비한 초기 예시 데이터 | 분류 3건·지식 3건 |

**S0 분류 테이블 `categories`**

| id | name |
|---:|---|
| 1 | learning |
| 2 | environment |
| 3 | reference |

**S0 지식 테이블 `knowledge`**

| id | title | category_id |
|---:|---|---:|
| 1 | 실습 파일 보관 | 1 |
| 2 | 서버 실행 | 2 |
| 3 | 질문 작성 | 1 |

- `knowledge.category_id=2`는 `categories.id=2`를 가리키며, 분류명은 `environment`입니다.
- 실제 발급된 ID는 조회해야 합니다. **ID는 행 수나 행의 위치가 아닙니다.**
- 전날 JSON 파일이 오늘 PostgreSQL로 자동 이관된 것은 아닙니다.

## 3. 키·관계·ERD

| 용어 | 역할 | 주의할 점 |
|---|---|---|
| Primary Key (PK, 기본키) | 각 행을 유일하게 식별하며 NULL을 허용하지 않음 | `knowledge.id` |
| Foreign Key (FK, 외래키) | 다른 테이블의 유효한 키를 참조하도록 검사 | `knowledge.category_id → categories.id` |
| IDENTITY | 값을 생략할 때 새 번호를 발급 | 번호 생성 자체와 PK의 유일성 보장은 별개 |
| ERD | 테이블의 필드·키·관계를 표현한 설계도 | 그림만으로 FK 제약이 생성되지는 않음 |
| Parent / Child | 참조되는 부모 테이블 / 참조하는 자식 테이블 | `categories` / `knowledge` |
| 1:N 관계 | 부모 한 건에 자식 여러 건이 연결될 수 있는 관계 | 분류 1개 → 지식 0개 이상 |
| N:M 관계 | 양쪽에 여러 항목이 연결되는 관계 | 지식 ↔ 태그, 중간 테이블 필요(부록의 선택 설계) |
| `ON DELETE RESTRICT` | 자식이 참조 중인 부모의 삭제를 거부 | 지식이 있는 분류를 삭제할 수 없음 |
| `CASCADE` | 관련 행으로 변경·삭제를 전파하는 정책의 한 형태 | 이번 공통 설계에서는 사용하지 않음 |

```text
categories (부모)                 knowledge (자식)
──────────────────               ─────────────────────
id          PK       1 ── 0..N    id           PK
name        UNIQUE                title
                                  content
                                  category_id  FK → categories.id
```

**구분:** PK는 *행 식별*, FK는 *참조 유효성 검사*, JOIN은 *조회할 때 행 연결*입니다. FK가 있어도 JOIN이 자동 실행되지는 않습니다.

## 4. 자료형과 제약조건

### 4.1 자료형

| 자료형 | 의미 | 이번 DB의 용도 |
|---|---|---|
| `BIGINT` | 큰 정수형 | `id`, `category_id` |
| `INTEGER` | 정수형 | 부록 연습용 테이블의 ID |
| `VARCHAR(n)` | 최대 n자 길이의 문자열 | `name VARCHAR(30)`, `title VARCHAR(80)` |
| `TEXT` | 텍스트 문자열 | `content` (길이는 별도 CHECK로 제한) |

### 4.2 제약조건

| 제약조건 | 의미 | 수업 적용 |
|---|---|---|
| `PRIMARY KEY` | 중복·NULL이 없는 식별자 | 각 테이블의 `id` |
| `FOREIGN KEY` | 존재하는 부모 키만 참조 | `knowledge.category_id` |
| `NOT NULL` | NULL 금지 | `name`, `title`, `content`, `category_id` |
| `UNIQUE` | 같은 값 중복 금지 | `categories.name` |
| `CHECK` | 정해진 조건 검사 | 문자열 길이 범위 |
| `DEFAULT` | 값을 생략했을 때 기본값 사용 | 명시적 NULL을 자동으로 바꾸지는 않음 |
| `CONSTRAINT` | 제약조건에 이름을 붙여 정의 | `knowledge_content_length` |

**수업의 저장 규칙**

| 필드 | 저장 규칙 |
|---|---|
| `categories.name` | 1~30자, 필수, 중복 불가 |
| `knowledge.title` | 1~80자, 필수, **중복 허용** |
| `knowledge.content` | 1~2,000자, 필수 |
| `knowledge.category_id` | 필수, 존재하는 분류 ID 참조 |
| `id` | BIGINT + IDENTITY + PK |

### 4.3 NULL과 빈 문자열

- `NULL`: 값이 없거나 알려지지 않은 상태.
- `''`: 길이가 0인 **실제 문자열 값**.
- `NOT NULL`은 NULL만 막으므로, 빈 본문을 막으려면 길이 `CHECK`도 필요합니다.
- NULL 검사에는 `= NULL` 대신 `IS NULL`을 사용합니다.

```sql
SELECT NULL IS NULL;       -- true
SELECT '' IS NULL;         -- false
SELECT char_length('');   -- 0
```

## 5. SQL 분류와 조회

### 5.1 구조 정의와 데이터 조작

| SQL | 역할 | 핵심 형태 |
|---|---|---|
| `CREATE DATABASE` | DB 생성 | `CREATE DATABASE knowledge_lab;` |
| `CREATE TABLE` | 테이블·열·제약 정의 | `CREATE TABLE categories (...);` |
| `CREATE INDEX` | 인덱스 생성 | `CREATE INDEX ... ON ...;` |
| `SELECT` | 데이터 조회 | `SELECT ... FROM ...;` |
| `INSERT` | 행 추가 | `INSERT INTO ... VALUES ...;` |
| `UPDATE` | 기존 행 수정 | `UPDATE ... SET ... WHERE ...;` |
| `DELETE` | 행 삭제 | `DELETE FROM ... WHERE ...;` |

`CREATE` 등의 구조 정의는 **DDL**, `INSERT`·`UPDATE`·`DELETE` 등의 데이터 변경은 **DML**이라고 부릅니다. 원본 강의는 주로 개별 SQL의 실제 동작에 초점을 맞춥니다.

### 5.2 SELECT에 자주 쓰는 문법

| 문법 | 뜻 | 예시 |
|---|---|---|
| `SELECT` | 반환할 열 선택 | `SELECT id, title` |
| `FROM` | 조회할 테이블 지정 | `FROM knowledge` |
| `WHERE` | 행 필터링 | `WHERE id = 2` |
| `AND` | 두 조건을 모두 만족 | `category_id = 1 AND ...` |
| `LIKE` | 문자열 패턴 검색 | `LIKE '%실습%'` |
| `%` | LIKE 패턴에서 임의 길이의 문자 | `%실습%`: '실습' 포함 |
| `ORDER BY` | 결과 정렬 | `ORDER BY id` |
| `DESC` | 내림차순 | `ORDER BY id DESC` |
| `LIMIT` | 최대 반환 행 수 제한 | `LIMIT 2` |
| `AS` | 열·테이블 별칭 지정 | `c.name AS category` |
| `IS NULL` | NULL 여부 검사 | `WHERE content IS NULL` |
| `IN` | 지정한 값들 중 하나인지 검사 | `WHERE id IN (2, 3)` |

```sql
SELECT id, title
FROM knowledge
WHERE category_id = 1
  AND title LIKE '%실습%'
ORDER BY id DESC
LIMIT 2;
```

**읽는 순서(의미 이해용):** `FROM → WHERE → SELECT → ORDER BY`. 이는 물리적 실행 순서를 고정한 설명이 아닙니다.

- 조건에 맞는 행이 없어서 **0행**이 반환되는 것은 정상적인 조회 결과일 수 있습니다.
- 문자열 리터럴은 작은따옴표로 표시합니다.
- `LIKE`는 문자 패턴 검색이지 임베딩 기반 의미 검색이 아닙니다.

## 6. 데이터 변경과 RETURNING

| 문법 | 뜻 | 실습 시 주의 |
|---|---|---|
| `INSERT INTO` | 새 행 등록 | FK가 가리키는 부모가 존재해야 함 |
| `VALUES` | 입력할 값 지정 | 열 순서와 값 순서 대응 |
| `INSERT ... SELECT` | 조회 결과를 이용해 행 등록 | 조회가 0행이면 등록도 0행일 수 있음 |
| `UPDATE` | 기존 행 변경 | `WHERE`로 대상을 제한 |
| `SET` | 변경할 열·값 지정 | `SET content = '...'` |
| `DELETE` | 행 삭제 | WHERE 생략 시 전체 행이 대상 |
| `RETURNING` | 추가·변경·삭제된 행의 값 반환 | 실제 발급 ID·변경 대상 확인 |
| CRUD | Create, Read, Update, Delete | 등록·조회·수정·삭제 |

```sql
-- 추가 후 실제 ID 확인
INSERT INTO knowledge (title, content, category_id)
VALUES ('임시 안내', '한 번만 실행합니다.', 1)
RETURNING id, title, category_id;

-- 특정 행만 수정
UPDATE knowledge
SET content = '실행 전에 가상환경을 확인합니다.'
WHERE id = 2
RETURNING id, content;

-- 특정 행만 삭제
DELETE FROM knowledge
WHERE id = 3
RETURNING id, title;
```

> 위 문장들은 **문법 예시**입니다. 실제 실습에서는 먼저 대상 SELECT를 실행하고 `BEGIN`으로 묶은 뒤, 목적에 따라 `ROLLBACK` 또는 `COMMIT`으로 종료하세요.

- 없는 ID를 UPDATE하면 `UPDATE 0` / `RETURNING` 0행이 나올 수 있습니다.
- `DELETE`는 행을 지우며 테이블 구조 자체를 없애지는 않습니다.
- ID는 제목으로 대체할 수 없습니다. 제목 중복이 허용되기 때문입니다.

## 7. JOIN과 집계

### 7.1 JOIN

| 용어 | 의미 | S0 기준 |
|---|---|---|
| `JOIN` / `INNER JOIN` | 연결 조건을 만족하는 행만 결합 | 지식 3행에 분류명 표시 |
| `LEFT JOIN` | 왼쪽 테이블의 행은 모두 유지 | 지식 없는 `reference`도 남음 |
| `ON` | 테이블 간 연결 조건 | `c.id = k.category_id` |
| 테이블 별칭 | 긴 테이블 이름을 짧게 표시 | `knowledge AS k`, `categories AS c` |

```sql
-- 지식과 분류 이름을 연결
SELECT k.id, k.title, c.name AS category
FROM knowledge AS k
JOIN categories AS c ON c.id = k.category_id
ORDER BY k.id;
```

```sql
-- 지식이 0개인 분류까지 조회
SELECT c.name, k.id AS knowledge_id, k.title
FROM categories AS c
LEFT JOIN knowledge AS k ON k.category_id = c.id
ORDER BY c.id, k.id;
```

S0에서 LEFT JOIN 결과는 총 **4행**이며, `reference`의 지식 ID와 제목은 NULL입니다.

### 7.2 집계

| 용어 | 의미 | 주의 |
|---|---|---|
| `GROUP BY` | 같은 기준의 행을 그룹으로 묶음 | 분류별 집계 |
| `COUNT(*)` | 결과 행의 개수를 셈 | LEFT JOIN의 빈 분류도 보존 행 1개로 셀 수 있음 |
| `COUNT(k.id)` | NULL이 아닌 지식 ID 개수 | 지식이 없는 분류는 0 |
| `SUM` | 합계 | 본문 글자 수 합계 |
| `MIN` / `MAX` | 최솟값 / 최댓값 | 본문 길이 범위 |
| `HAVING` | 집계한 그룹을 조건으로 필터링 | `HAVING COUNT(k.id) >= 2` |
| `char_length()` | 문자열 길이를 문자 수로 계산 | 빈 문자열 길이는 0 |

```sql
SELECT c.name AS category,
       COUNT(k.id) AS knowledge_count
FROM categories AS c
LEFT JOIN knowledge AS k ON k.category_id = c.id
GROUP BY c.id, c.name
ORDER BY c.id;
```

S0 결과: `learning=2`, `environment=1`, `reference=0`.

**WHERE vs HAVING:** WHERE는 그룹으로 묶기 **전의 행**을, HAVING은 그룹으로 묶은 **후의 그룹**을 필터링합니다.

## 8. 트랜잭션과 오류 복구

| 용어 | 의미 | 핵심 |
|---|---|---|
| Transaction (트랜잭션) | 여러 작업을 확정·취소하는 단위 | 같은 연결에서 처리 |
| `BEGIN` | 트랜잭션 시작 | 이후 변경은 종료 전까지 미확정 |
| `COMMIT` | 트랜잭션 변경 확정 | 재접속 후에도 변경값 확인 |
| `ROLLBACK` | 미확정 변경 취소 | 이미 COMMIT한 변경은 취소할 수 없음 |
| 원자성 (Atomicity) | 묶인 작업을 하나의 단위로 처리 | 실패한 변경 묶음을 ROLLBACK하여 함께 취소 |
| Auto commit | 도구의 자동 확정 설정 | 수업은 켬; 명시적 BEGIN 구간은 별도 관리 |
| Auto rollback on error | 오류 시 자동 롤백 관련 설정 | 수업은 끔; 오류 후 직접 ROLLBACK |
| SQLSTATE | DB 오류의 분류 코드 | HTTP 상태코드와 다름 |

```sql
-- 변경을 취소하는 실습
BEGIN;
UPDATE knowledge
SET content = '취소할 임시 안내입니다.'
WHERE id = 2;
ROLLBACK;

-- 변경을 확정하는 실습
BEGIN;
UPDATE knowledge
SET content = '실행 전에 가상환경을 확인합니다.'
WHERE id = 2
RETURNING id, content;
COMMIT;
```

**대표 SQLSTATE**

| 코드 | 원인 | 관련 규칙 |
|---|---|---|
| `23502` | NULL 금지 위반 | NOT NULL |
| `23505` | 중복값 위반 | UNIQUE |
| `23514` | 검사 조건 위반 | CHECK |
| `23503` | 존재하지 않는 부모 참조 | FK |
| `25P02` | 오류로 중단된 트랜잭션에서 명령 실행 | ROLLBACK 필요 |

**안전한 변경 순서:** `SELECT로 대상 확인 → BEGIN → 변경 → RETURNING 확인 → COMMIT 또는 ROLLBACK → 다시 SELECT`.

- 실패 시 오류의 이름·코드와 기존 데이터 보존 여부를 함께 확인합니다.
- INSERT를 ROLLBACK하거나 INSERT가 실패해도 IDENTITY 시퀀스 번호는 건너뛸 수 있습니다.
- 번호가 `1, 2, 5`여도 정상일 수 있으므로 `max(id)`를 행 수로 취급하지 않습니다.
- `ROLLBACK`은 DB 전체를 과거 시점으로 되돌리는 명령이 아닙니다.

## 9. 인덱스와 실행계획

| 용어 | 의미 | 주의 |
|---|---|---|
| Index (인덱스) | 행을 찾기 위한 별도 탐색 구조 | 읽기 개선 가능, 쓰기·저장 비용 발생 |
| B-tree 인덱스 | 일반적인 값 기반 검색에 쓰이는 인덱스 구조 | 의미 검색용 벡터 인덱스와 다름 |
| `CREATE INDEX` | 인덱스 생성 | 동일 인덱스 존재 여부 먼저 확인 |
| `pg_indexes` | 현재 인덱스 정보를 조회하는 시스템 뷰 | 테이블 인덱스 확인 |
| `EXPLAIN` | DB의 예상 실행계획 표시 | 실제 실행과 구분 |
| `EXPLAIN ANALYZE` | 쿼리를 실제 실행해 실행 정보 표시 | 수업에서는 SELECT만 측정 |
| `BUFFERS` | 실행계획의 버퍼 관련 정보 표시 옵션 | `EXPLAIN (ANALYZE, BUFFERS)` |
| `Seq Scan` | 순차적으로 테이블을 읽는 접근 방법 | 작은 데이터에서는 정상적인 선택일 수 있음 |
| `rows` | 계획상의 예상/실제 행 수 정보 | 계획·실제 구분 |
| `cost` | 계획의 비용 추정치 | 밀리초 단위가 아님 |

```sql
-- 기존 인덱스 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename = 'knowledge';

-- 중복 생성 여부 확인 후 실행
CREATE INDEX idx_knowledge_category_id
ON knowledge (category_id);

-- 접근 계획 확인
EXPLAIN
SELECT id, title FROM knowledge WHERE category_id = 1;
```

- FK를 만들었다고 자식 테이블의 FK 열에 인덱스가 자동 생성되는 것은 아닙니다.
- 인덱스가 존재해도 항상 인덱스를 사용하는 것은 아닙니다.
- 인덱스 생성 전후 성능 향상이나 특정 계획 이름을 보장하지 않습니다.

## 10. AI·API 연결 개념

| 용어 | 의미 | 오늘의 범위와 관계 |
|---|---|---|
| FastAPI | HTTP 요청·응답을 처리하는 API 계층 | 다음 수업에서 DB 연결 |
| API | 프로그램 사이의 요청·응답 인터페이스 | SQL 결과를 HTTP 응답으로 해석하는 책임은 API에 있음 |
| HTTP 상태코드 | API 요청의 처리 상태를 표현 | `200`, `201`, `404`, `422` 등은 DB가 직접 반환하지 않음 |
| JSON | 데이터를 구조화해 표현하는 형식 | 전날 파일 기반 지식 저장에 사용 |
| Pydantic | API 입력·응답 모델 검증과 관련된 도구 | Pydantic 모델과 DB 테이블 정의는 다름 |
| ORM | 애플리케이션 객체와 DB 테이블을 연결하는 방식 | 후속 연동 수업 |
| Migration | DB 스키마 변경을 관리하는 작업 | 후속 연동 수업 |
| RAG | 검색한 근거를 활용해 답변을 생성하는 구조 | 오늘은 원본 지식 DB만 다룸 |
| LLM | 대규모 언어 모델 | DB 제약조건이 LLM 답변의 사실성을 보장하지 않음 |
| AI Agent | 질문을 이해하고 도구·근거를 활용하는 실행 계층 | DB에 저장된 원본 지식을 활용하는 후속 단계 |
| Embedding | 문서·질문을 벡터 표현으로 변환 | 의미 검색의 후속 단계 |
| Vector Search | 벡터 유사도로 근거 후보를 찾는 검색 | SQL의 일반 B-tree 인덱스와 별개 |
| Metadata | 지식에 딸린 분류·출처 등 부가 정보 | 후속 문서·검색 처리와 연결 |

```text
사용자 요청
   ↓
FastAPI (요청·응답)
   ↓
PostgreSQL (원본 지식·관계·제약·트랜잭션)
   ↓
후속 검색 인덱스 / RAG / Agent / LLM 활용
```

**책임 구분:** DB 제약은 데이터 저장 규칙을 검사합니다. API의 HTTP 응답 결정, AI 답변의 사실성 검증, 벡터 인덱스 갱신은 각각 별도의 작업입니다. 원본 수정·삭제가 벡터 인덱스에 자동 전파되지는 않습니다.

## 11. 헷갈리는 개념 비교

| A | B | 핵심 차이 |
|---|---|---|
| Database | Schema | DB는 데이터·구조의 논리적 공간, Schema는 그 안의 이름 공간 |
| PostgreSQL 서버 | pgAdmin | 서버는 DB 요청 처리, pgAdmin은 접속 클라이언트 |
| Row | Column | 한 건의 데이터 vs 데이터의 공통 항목 |
| PK | FK | 행 자체의 유일 식별 vs 부모 참조 검사 |
| FK | JOIN | 데이터 유효성 제약 vs 조회 시 연결 |
| IDENTITY | PK | 번호 발급 vs 유일성·NULL 금지 |
| `NULL` | `''` | 값 없음 vs 길이 0인 문자열 |
| NOT NULL | CHECK | NULL 방지 vs 조건 검사(예: 길이) |
| UNIQUE | PRIMARY KEY | 중복 방지 vs 행의 기본 식별자 |
| INNER JOIN | LEFT JOIN | 매칭되는 행만 vs 왼쪽 행 모두 보존 |
| `COUNT(*)` | `COUNT(k.id)` | 결과 행 개수 vs NULL 아닌 지식 ID 개수 |
| WHERE | HAVING | 그룹화 전 행 조건 vs 그룹화 후 그룹 조건 |
| COMMIT | ROLLBACK | 미확정 변경 확정 vs 취소 |
| SQLSTATE | HTTP 상태코드 | DB 오류 분류 vs API 응답 상태 |
| SQL INDEX | Vector Index | 일반 조건 기반 접근 지원 vs 벡터 기반 의미 검색 |
| DB 제약 검증 | AI 답변 검증 | 저장 규칙의 적합성 vs 답변의 사실성·품질 |
| SELECT 0행 | SQL 오류 | 정상 실행 후 결과 없음 vs 명령 실행 실패 |

## 12. 실습용 핵심 SQL 모음

### 12.1 현재 접속 확인

```sql
SELECT current_database() AS db_name,
       current_user AS user_name;
SELECT version();
SELECT 1 AS connection_ok;
```

### 12.2 테이블 정의(빈 개인 학습 DB에서 한 번만)

```sql
CREATE TABLE categories (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name VARCHAR(30) NOT NULL UNIQUE,
  CONSTRAINT categories_name_length
    CHECK (char_length(name) BETWEEN 1 AND 30)
);

CREATE TABLE knowledge (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  title VARCHAR(80) NOT NULL,
  content TEXT NOT NULL,
  category_id BIGINT NOT NULL,
  CONSTRAINT knowledge_title_length
    CHECK (char_length(title) BETWEEN 1 AND 80),
  CONSTRAINT knowledge_content_length
    CHECK (char_length(content) BETWEEN 1 AND 2000),
  CONSTRAINT knowledge_category_fk
    FOREIGN KEY (category_id)
    REFERENCES categories(id) ON DELETE RESTRICT
);
```

### 12.3 초기 데이터와 관계 조회

```sql
-- 새로 준비한 빈 테이블에만 한 번 실행
INSERT INTO categories (name)
VALUES ('learning'), ('environment'), ('reference')
RETURNING id, name;

-- 이름으로 실제 분류 ID를 찾아 등록
INSERT INTO knowledge (title, content, category_id)
SELECT '실습 파일 보관',
       '실습 파일은 개인 폴더에 보관합니다.', id
FROM categories WHERE name = 'learning'
RETURNING id, title;

-- 관계 조회
SELECT k.id, k.title, c.name AS category
FROM knowledge AS k
JOIN categories AS c ON c.id = k.category_id
ORDER BY k.id;
```

### 12.4 실패 검증(실제 분류 1이 존재할 때)

```sql
BEGIN;
INSERT INTO knowledge (title, content, category_id)
VALUES ('오류 확인', '', 1);
-- CHECK 위반 오류를 확인한 뒤, 별도로 실행
ROLLBACK;
SELECT COUNT(*) FROM knowledge;
```

### 12.5 psql 전용 명령

```text
\conninfo       현재 접속 정보
\dt             테이블 목록
\d knowledge    테이블 구조
\q              psql 종료
```

> 위 `\` 명령은 SQL이 아니라 **psql 전용 명령**이므로 pgAdmin Query Tool에 입력하지 않습니다.

---

## 최종 암기 포인트

1. **테이블 = 행(Row) + 열(Column)**, 행은 데이터 한 건입니다.
2. **PK는 식별, FK는 참조 검사, JOIN은 조회 연결**입니다.
3. **NOT NULL은 빈 문자열을 막지 않습니다.** 길이 CHECK가 필요합니다.
4. **UPDATE·DELETE는 WHERE로 대상을 제한**하고, RETURNING으로 실제 변경 행을 확인합니다.
5. **LEFT JOIN + COUNT(k.id)**를 쓰면 지식이 없는 분류도 0건으로 집계할 수 있습니다.
6. **COMMIT은 확정, ROLLBACK은 미확정 변경 취소**입니다.
7. **ID는 연속되지 않을 수 있습니다.** 실제 ID는 RETURNING/SELECT로 확인합니다.
8. **INDEX는 항상 빠른 결과를 보장하지 않습니다.** EXPLAIN으로 접근 계획을 관찰합니다.
9. **SQL 결과와 HTTP 상태코드는 다릅니다.** HTTP 응답 정책은 API 계층에서 결정합니다.
10. **PostgreSQL 원본 지식 관리와 RAG 의미 검색은 다른 역할**입니다.

**출처 범위:** 원본 본문 1~8교시, 부록 A01~A17. 설치·제출 평가의 세부 절차보다 개념·용어·대표 SQL을 우선해 정리했습니다.
