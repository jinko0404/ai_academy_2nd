# PostgreSQL SQL 핵심 개념 정리

> PostgreSQL 실습에서 사용한 SQL 문법과 개념을 **중복 없이 주제별로** 정리한 README입니다. 예제의 `public.knowledge`, `public.categories`는 실습 테이블을 기준으로 합니다.

## 목차

1. [접속 정보 확인](#1-접속-정보-확인)
2. [SELECT와 조건 검색](#2-select와-조건-검색)
3. [테이블 생성과 자료형](#3-테이블-생성과-자료형)
4. [키와 제약조건](#4-키와-제약조건)
5. [CRUD와 RETURNING](#5-crud와-returning)
6. [JOIN의 종류](#6-join의-종류)
7. [집계 함수와 GROUP BY](#7-집계-함수와-group-by)
8. [트랜잭션과 데이터 복구](#8-트랜잭션과-데이터-복구)
9. [제약조건 오류와 SQLSTATE](#9-제약조건-오류와-sqlstate)
10. [헷갈리기 쉬운 개념 비교](#10-헷갈리기-쉬운-개념-비교)

---

## 1. 접속 정보 확인

| 표현 | 의미 | 예시 |
|---|---|---|
| `current_database()` | 현재 연결된 데이터베이스 이름 | `SELECT current_database();` |
| `current_user` | 현재 명령을 수행하는 DB 역할 이름 (괄호 없음) | `SELECT current_user;` |
| `version()` | PostgreSQL 서버 버전 및 빌드 정보 | `SELECT version();` |
| `AS` | 결과 열 또는 테이블에 별칭 부여 | `SELECT current_database() AS database_name;` |

```sql
SELECT current_database() AS database_name,
       current_user AS user_name;
SELECT version();
```

## 2. SELECT와 조건 검색

| 문법 | 역할 | 예시 |
|---|---|---|
| `SELECT` | 표시할 열 선택 | `SELECT id, title` |
| `FROM` | 조회할 테이블 지정 | `FROM public.knowledge` |
| `WHERE` | 조건에 맞는 행만 선택 | `WHERE category_id = 1` |
| `AND` | 모든 조건을 만족해야 함 | `WHERE category_id = 1 AND title ILIKE '%실습%'` |
| `LIKE` | 대소문자를 구분하는 문자열 패턴 검색(일반적인 영문 기준) | `title LIKE '%SQL%'` |
| `ILIKE` | 대소문자를 구분하지 않는 패턴 검색 | `title ILIKE '%sql%'` |
| `ORDER BY` | 결과 정렬 | `ORDER BY id DESC` |
| `ASC` / `DESC` | 오름차순(기본값) / 내림차순 | `ORDER BY id ASC` |
| `LIMIT` | 반환할 최대 행 수 | `LIMIT 1` |

**문자열 패턴**

| 패턴 | 뜻 |
|---|---|
| `'%실습%'` | '실습'을 포함 |
| `'실습%'` | '실습'으로 시작 |
| `'%실습'` | '실습'으로 끝남 |

`%`는 길이가 0 이상인 임의의 문자열을 의미합니다. SQL 문자열은 작은따옴표(`'...'`)로 감쌉니다. `LIKE`/`ILIKE`는 **문자열 패턴 검색**이며 의미 기반 검색은 아닙니다.

```sql
SELECT id, title
FROM public.knowledge
WHERE category_id = 1
  AND title ILIKE '%실습%'
ORDER BY id ASC
LIMIT 1;
```

분류 ID가 1이고 제목에 '실습'이 포함된 지식 중, ID가 가장 작은 행의 `id`, `title`을 최대 1행 반환합니다. 일치하는 행이 없으면 정상적으로 0행이 나옵니다.

**SQL 작성 순서:** `SELECT → FROM → WHERE → ORDER BY → LIMIT`

## 3. 테이블 생성과 자료형

| 문법/자료형 | 의미 |
|---|---|
| `CREATE TABLE` | 열, 자료형, 제약조건을 지정해 테이블 생성 |
| `INTEGER` | 정수 자료형 |
| `TEXT` | 선언 시 최대 길이를 지정하지 않는 문자열 자료형 |
| `VARCHAR(n)` | 최대 `n`자의 문자열 자료형 |

```sql
CREATE TABLE public.categories_try (
    id INTEGER PRIMARY KEY,
    name VARCHAR(30) NOT NULL UNIQUE,
    CHECK (char_length(name) BETWEEN 1 AND 30)
);
```

`char_length(text)`는 문자 수를 반환합니다. `BETWEEN 1 AND 30`은 **1 이상 30 이하**로 양쪽 경계를 포함합니다.

## 4. 키와 제약조건

| 제약조건 | 역할 | 주의점 |
|---|---|---|
| `PRIMARY KEY` (PK) | 각 행을 유일하게 식별 | 중복과 `NULL` 금지, **자동 번호 발급 기능 자체는 아님** |
| `REFERENCES` / `FOREIGN KEY` (FK) | 다른 테이블의 키를 참조 | 없는 부모 키를 참조하면 오류 |
| `NOT NULL` | `NULL` 금지 | 빈 문자열 `''`은 막지 못함 |
| `UNIQUE` | 값 중복 방지 | `NULL`도 막으려면 `NOT NULL`을 함께 사용 |
| `CHECK` | 조건이 `FALSE`인 데이터 거부 | `NULL` 금지는 별도 `NOT NULL` 사용 |
| `ON DELETE RESTRICT` | 참조 중인 부모 행 삭제 거부 | 자식 행으로 삭제가 전파되지 않음 |

```sql
CREATE TABLE public.knowledge_try (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category_id INTEGER NOT NULL
        REFERENCES public.categories_try(id)
);
```

- **부모 테이블:** `categories_try` (`id`를 참조받음)
- **자식 테이블:** `knowledge_try` (`category_id`가 부모 `id`를 참조)
- `PRIMARY KEY`는 식별 규칙, `FOREIGN KEY`는 참조 규칙, `JOIN`은 조회 시 행을 연결하는 기능입니다.

## 5. CRUD와 RETURNING

| 작업 | SQL | 의미 |
|---|---|---|
| Create | `INSERT INTO ... VALUES` | 새 행 추가 |
| Read | `SELECT` | 기존 행 조회 |
| Update | `UPDATE ... SET ... WHERE` | 조건에 맞는 행 수정 |
| Delete | `DELETE FROM ... WHERE` | 조건에 맞는 행 삭제 |

### INSERT

```sql
INSERT INTO public.knowledge (title, content, category_id)
VALUES ('임시 안내', '연습 데이터', 1)
RETURNING id, title;
```

새 행의 실제 ID와 제목을 반환합니다. 열 목록을 생략한 `INSERT`는 테이블에 정의된 열 순서대로 값을 제공해야 합니다.

### UPDATE

```sql
UPDATE public.knowledge
SET content = '수정 연습 안내입니다.'
WHERE id = 4
RETURNING id, content;
```

`content`만 수정하며 다른 열은 그대로 둡니다.

### DELETE

```sql
DELETE FROM public.knowledge
WHERE id = 4
RETURNING id, title;
```

해당 행을 삭제하지만 테이블 구조는 유지됩니다.

> **주의:** `UPDATE`와 `DELETE`에서 `WHERE`를 생략하면 모든 행이 대상입니다. 변경 전에 동일한 조건으로 `SELECT`하여 대상을 확인하세요.

`RETURNING`은 INSERT/UPDATE/DELETE가 처리한 행의 값을 보여주는 구문입니다. **결과가 출력되었다고 COMMIT된 것은 아닙니다.**

## 6. JOIN의 종류

예시 데이터:

**categories**

| id | name |
|---:|---|
| 1 | learning |
| 2 | api |
| 3 | reference |

**knowledge**

| id | title | category_id |
|---:|---|---:|
| 10 | SQL | 1 |
| 11 | JOIN | 1 |
| 12 | 서버 | 2 |

| 종류 | 반환하는 행 | 핵심 |
|---|---|---|
| `INNER JOIN` (`JOIN`) | 양쪽에서 조건이 일치하는 행 | 연결되지 않은 분류 제외 |
| `LEFT JOIN` | 왼쪽 전체 + 일치하는 오른쪽 행 | 오른쪽에 없으면 `NULL` |
| `RIGHT JOIN` | 오른쪽 전체 + 일치하는 왼쪽 행 | 왼쪽에 없으면 `NULL` |
| `FULL OUTER JOIN` (`FULL JOIN`) | 양쪽 전체 | 매칭되지 않은 쪽을 `NULL`로 표시 |
| `CROSS JOIN` | 모든 가능한 행 조합 | M행 × N행, `ON` 없음 |
| `SELF JOIN` | 동일 테이블을 서로 다른 별칭으로 연결 | 별도 JOIN 키워드가 아닌 사용 방식 |
| `NATURAL JOIN` | 이름이 같은 모든 열을 자동 조건으로 연결 | 의도치 않은 열로 연결될 수 있음 |
| `JOIN ... USING (열)` | 양쪽에서 이름이 같은 열을 기준으로 연결 | `ON`의 대체 표기, JOIN 종류는 아님 |

### INNER JOIN과 LEFT JOIN

```sql
-- 분류가 연결된 지식만
SELECT c.name, k.title
FROM public.categories AS c
INNER JOIN public.knowledge AS k
    ON c.id = k.category_id;

-- 지식이 없는 분류(reference)도 표시
SELECT c.name, k.title
FROM public.categories AS c
LEFT JOIN public.knowledge AS k
    ON c.id = k.category_id;
```

| 분류 | INNER JOIN 결과 | LEFT JOIN 결과 |
|---|---|---|
| learning | SQL, JOIN | SQL, JOIN |
| api | 서버 | 서버 |
| reference | 없음 | `NULL`인 행 1개 |

### RIGHT / FULL / CROSS JOIN

```sql
-- 오른쪽 categories를 모두 유지
SELECT c.name, k.title
FROM public.knowledge AS k
RIGHT JOIN public.categories AS c ON k.category_id = c.id;

-- 양쪽의 매칭되지 않은 행까지 유지
SELECT c.name, k.title
FROM public.categories AS c
FULL OUTER JOIN public.knowledge AS k ON c.id = k.category_id;

-- 3개 분류 × 3개 지식 = 9행
SELECT c.name, k.title
FROM public.categories AS c
CROSS JOIN public.knowledge AS k;
```

실습의 FK와 `NOT NULL` 규칙이 정상 적용된다면 부모 분류가 없는 지식은 저장되지 않습니다.

### SELF JOIN

```sql
-- employees(id, name, manager_id) 테이블 가정
SELECT e.name AS employee, m.name AS manager
FROM employees AS e
LEFT JOIN employees AS m ON e.manager_id = m.id;
```

같은 테이블을 직원(`e`)과 상사(`m`) 역할로 각각 사용합니다. 상사가 없는 직원도 결과에 남습니다.

### NATURAL JOIN과 USING 주의점

```sql
-- 공통 열 이름 'id'로 연결되어 의도한 FK 관계와 달라질 수 있음
SELECT * FROM categories NATURAL JOIN knowledge;

-- 양쪽 테이블 모두 category_id 열을 가진 경우에 사용
SELECT * FROM table_a JOIN table_b USING (category_id);
```

현재 실습 테이블의 관계는 `categories.id = knowledge.category_id`이므로 명시적인 `ON`이 적절합니다.

### EXISTS / NOT EXISTS (SEMI / ANTI JOIN에 해당하는 조회)

```sql
-- 지식이 하나 이상 있는 분류
SELECT c.id, c.name
FROM public.categories AS c
WHERE EXISTS (
    SELECT 1 FROM public.knowledge AS k
    WHERE k.category_id = c.id
);

-- 지식이 하나도 없는 분류
SELECT c.id, c.name
FROM public.categories AS c
WHERE NOT EXISTS (
    SELECT 1 FROM public.knowledge AS k
    WHERE k.category_id = c.id
);
```

`EXISTS`는 일치 행의 **존재 여부**만 확인하므로 일치하는 지식이 여러 개여도 분류 행이 그만큼 늘어나지 않습니다.

> JOIN은 테이블을 합쳐 저장하지 않습니다. 한쪽 행에 여러 행이 매칭되면 조회 결과 행 수는 증가할 수 있습니다. `ORDER BY`가 없다면 결과 순서는 보장되지 않습니다.

## 7. 집계 함수와 GROUP BY

| 문법 | 역할 |
|---|---|
| `COUNT(*)` | 결과 행 전체 개수 |
| `COUNT(열)` | 해당 열이 `NULL`이 아닌 행 개수 |
| `SUM(열)` | 값의 합계 |
| `MIN(열)` / `MAX(열)` | 최솟값 / 최댓값 |
| `GROUP BY` | 같은 기준값의 행을 그룹으로 묶음 |
| `HAVING` | 그룹별 집계 결과에 조건 적용 |

```sql
SELECT c.name AS category,
       COUNT(k.id) AS knowledge_count
FROM public.categories AS c
LEFT JOIN public.knowledge AS k
    ON k.category_id = c.id
GROUP BY c.id, c.name
HAVING COUNT(k.id) >= 2
ORDER BY c.id;
```

분류별 지식 수를 세고 **지식이 2개 이상인 분류만** 반환합니다. PostgreSQL에서는 이 예제의 `HAVING`에 SELECT 별칭 대신 `COUNT(k.id)`를 사용합니다.

**LEFT JOIN에서 COUNT의 차이**

| 분류 | `COUNT(*)` | `COUNT(k.id)` |
|---|---:|---:|
| learning | 2 | 2 |
| api | 1 | 1 |
| reference | 1 | 0 |

`reference`는 지식이 없어도 LEFT JOIN 결과에 `NULL`이 있는 행 하나가 남습니다. 실제 지식 수는 `COUNT(k.id)`로 세어야 합니다.

```sql
SELECT COUNT(*) AS total_knowledge,
       SUM(char_length(content)) AS total_chars,
       MIN(char_length(content)) AS min_chars,
       MAX(char_length(content)) AS max_chars
FROM public.knowledge;
```

`WHERE`는 **그룹으로 묶기 전의 개별 행**, `HAVING`은 **그룹으로 묶은 후의 집계 결과**를 필터링합니다.

## 8. 트랜잭션과 데이터 복구

| 명령 | 의미 |
|---|---|
| `BEGIN` | 트랜잭션 시작: 여러 작업을 하나의 처리 단위로 묶음 |
| `COMMIT` | 현재 트랜잭션의 변경 확정 |
| `ROLLBACK` | 현재 트랜잭션의 미확정 변경 취소 |

### ROLLBACK 예시

```sql
BEGIN;
UPDATE public.knowledge
SET content = '임시 내용'
WHERE id = 3;
ROLLBACK;
```

수정한 본문이 원래 상태로 돌아갑니다. 같은 트랜잭션에서 여러 행을 수정했다면 함께 취소됩니다.

### COMMIT 예시

```sql
BEGIN;
UPDATE public.knowledge
SET content = '확정할 내용'
WHERE id = 3;
COMMIT;
```

변경이 확정되어 새 연결에서도 확인할 수 있습니다. **이미 COMMIT한 변경은 나중에 ROLLBACK으로 취소할 수 없습니다.**

### 여러 작업을 하나로 묶기

```sql
BEGIN;
UPDATE public.knowledge SET content = '묶음 연습 1' WHERE id = 1;
UPDATE public.knowledge SET content = '묶음 연습 2' WHERE id = 3;
ROLLBACK;  -- 두 수정 모두 취소
```

- 같은 연결에서는 COMMIT 전에도 자신의 변경을 조회할 수 있습니다.
- 다른 연결에서는 일반적으로 미확정 변경이 보이지 않습니다.
- 자동 커밋 상태에서 `BEGIN` 없이 실행한 성공한 UPDATE는 즉시 확정될 수 있습니다.
- ROLLBACK이나 INSERT 실패 후에도 자동 발급 ID의 시퀀스 번호는 건너뛸 수 있습니다.

## 9. 제약조건 오류와 SQLSTATE

`SQLSTATE`는 PostgreSQL 오류 종류를 식별하는 **5자리 코드**입니다. 실습에서는 오류 메시지의 실제 코드와 제약 이름을 확인해야 합니다.

| SQLSTATE | 오류 | 실습 상황 |
|---|---|---|
| `23514` | CHECK 위반 | `content = ''` (빈 본문) |
| `23503` | FOREIGN KEY 위반 | 존재하지 않는 `category_id` 또는 참조 중인 부모 삭제 |
| `23502` | NOT NULL 위반 | `title = NULL` |
| `23505` | UNIQUE 위반 | 기존 `learning` 분류명 중복 |
| `25P02` | 실패 상태인 트랜잭션에서 후속 명령 실행 | 오류 후 ROLLBACK 없이 계속 실행 |

### 오류 재현 예시: 빈 본문

```sql
BEGIN;
INSERT INTO public.knowledge (title, content, category_id)
VALUES ('오류 확인', '', 1);
-- 예상: CHECK 위반 (23514)
```

**오류 메시지를 확인한 다음**, 같은 연결에서 아래 명령을 **별도로** 실행합니다.

```sql
ROLLBACK;
```

다른 테스트 입력 예시(각각 별도의 `BEGIN` 및 오류 후 `ROLLBACK` 사용):

```sql
-- 사전에 category_id = -1이 없는지 확인
INSERT INTO public.knowledge (title, content, category_id)
VALUES ('오류 확인', '정상 본문', -1);  -- 예상: 23503

INSERT INTO public.knowledge (title, content, category_id)
VALUES (NULL, '정상 본문', 1);       -- 예상: 23502

-- learning 분류가 이미 존재한다는 전제
INSERT INTO public.categories (name)
VALUES ('learning');               -- 예상: 23505

-- 분류 id=1을 지식이 참조 중이고 ON DELETE RESTRICT라는 전제
DELETE FROM public.categories WHERE id = 1;  -- 예상: 23503
```

**안전한 검증 순서**

1. 실행 전 지식·분류 목록과 개수를 기록합니다.
2. `BEGIN`을 실행하고 오류를 일으킬 명령 하나만 실행합니다.
3. 실제 SQLSTATE와 제약 이름을 기록합니다.
4. `ROLLBACK`을 별도로 실행합니다.
5. 실행 전후의 행과 값이 유지되는지 비교합니다.

> 위 오류 코드는 **실습 조건에 따른 예상값**이며, 실제 DB에서 실행한 결과가 아닙니다. 실행 도구가 오류 지점에서 멈출 수 있으므로 오류 SQL과 ROLLBACK을 한꺼번에 선택 실행하지 않는 것이 안전합니다.

## 10. 헷갈리기 쉬운 개념 비교

| 구분 | 핵심 차이 |
|---|---|
| `NULL` vs `''` | 값이 없음 vs 길이 0인 문자열. `NOT NULL`은 `''`을 막지 않음 |
| PK vs FK | 행의 고유 식별 vs 다른 테이블의 키 참조 |
| FK vs JOIN | 참조 무결성 검사 vs 조회 결과에서 행 연결 |
| `INNER JOIN` vs `LEFT JOIN` | 매칭되는 행만 vs 왼쪽 행 모두 |
| `COUNT(*)` vs `COUNT(열)` | 모든 결과 행 vs 해당 열이 NULL이 아닌 행 |
| `WHERE` vs `HAVING` | 그룹화 전 행 필터 vs 그룹화 후 집계 필터 |
| `RETURNING` vs `COMMIT` | 변경 결과 반환 vs 변경 확정 |
| `ROLLBACK` vs `COMMIT` | 미확정 변경 취소 vs 변경 확정 |
| `LIKE` vs `ILIKE` | 일반적인 대소문자 구분 패턴 검색 vs 대소문자 무시 패턴 검색 |
| `DELETE` vs `DROP TABLE` | 행 삭제(테이블 유지) vs 테이블 자체 제거 |

---

### 한눈에 보는 SQL 흐름

```sql
-- 조회: 조건 → 연결 → 그룹화 → 그룹 필터 → 정렬 → 개수 제한
SELECT c.name, COUNT(k.id) AS knowledge_count
FROM public.categories AS c
LEFT JOIN public.knowledge AS k ON k.category_id = c.id
WHERE c.name ILIKE '%a%'
GROUP BY c.id, c.name
HAVING COUNT(k.id) >= 1
ORDER BY c.id ASC
LIMIT 10;
```

**실습 원칙:** 실제 ID 확인 → 변경 전 SELECT → BEGIN → 변경 및 RETURNING 확인 → COMMIT 또는 ROLLBACK → 최종 SELECT로 검증.

---

*정리 기준: 사용자가 제공한 SQL 실습 용어·개념 텍스트. 반복된 JOIN, CRUD, 트랜잭션 설명을 통합하고 실습 전제와 예상 오류 코드를 유지했습니다.*
