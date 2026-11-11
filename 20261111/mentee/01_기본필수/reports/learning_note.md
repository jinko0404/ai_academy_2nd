# 교시별 실행 기록
비밀번호는 적지 않습니다. 예상값을 실제 관찰값으로 바꿔 적지 않습니다.

아래는 전달받은 실행 결과와 SQL 검토 내용을 정리한 기록이다. 실행 화면이나 로그를 전달받지 않은 부분은 SQL 기준으로 표시했다.

| 교시 | 실행 파일/구간 | 예상 | 실제 결과·행 수·ID | 오류와 해결·설명 |
|---|---|---|---|---|
| 01 | 01_connect.sql / DB·역할·서버 버전 조회 | 접속 확인 1, 개인 knowledge_lab DB 이름, 서버 정보와 한글 출력 | 전달받은 값: knowledge_lab / PostgreSQL 18.6, x86_64-windows, 64-bit | current_database()는 DB 이름, current_user는 현재 역할, version()은 서버 정보를 반환한다. |
| 02 | 02_select.sql / 전체·상세·분류·빈 결과 조회 | S0 전체 3행, id=2 상세 1행, 분류 1은 id=1·3, id=-1은 0행 | before_knowledge.csv에서 기존 id=1·2·3의 3행 확인. 조건별 결과는 SQL 기준으로 왼쪽과 같다. | 상세 SELECT 끝의 세미콜론 누락을 수정했다. 조건에 맞는 행이 없으면 정상적인 0행 결과다. |
| 03 | 03_keys_erd.sql / 연습 테이블 생성·참조·취소 | 자식 1행 조회 후 ROLLBACK으로 두 테이블 제거 | SQL 기준: knowledge_try에 (1, 안내, 1) 한 행, 취소 후 parent_gone·child_gone은 true | 열에 FK를 선언할 때 REFERENCES를 사용한다. 실제 분류명에는 UNIQUE가 있지만 지식 제목에는 없으며, IDENTITY는 자동 발급이고 PK는 중복·NULL 금지다. |
| 04 | 04_crud.sql / INSERT·UPDATE·DELETE와 각 ROLLBACK | 추가·수정·삭제 결과 반환 후 기존 행 보존 | 기본필수 실행 로그 미전달. 별도 응용 INSERT 화면에서 실제 생성 ID=4 확인. 기본필수는 SQL 기준으로 기존 id=2 수정·id=3 삭제가 각 ROLLBACK으로 취소된다. | INSERT INTO와 RETURNING을 완성했다. RETURNING은 결과 반환이며 확정 명령은 아니다. 자동 발급 번호는 ROLLBACK 후에도 소비될 수 있다. |
| 05 | 05_join.sql / 분류명 연결·분류별 집계 | S0 JOIN 3행, learning 2·environment 1·reference 0 | SQL 기준: 분류별 합계 3개. 본문 총 문자 수·최소·최대 길이의 실제 수치는 미기록 | 제목 대신 COUNT(k.id)로 지식 수를 센다. LEFT JOIN은 빈 분류도 남기고 COUNT(k.id)는 연결되지 않은 NULL을 세지 않는다. |
| 06 | 06_constraints.sql / 빈 본문·없는 분류 실패와 회복 | 빈 본문 23514, 없는 분류 23503, 각각 ROLLBACK 후 기존 행 보존 | SQL·준비 스키마 기준: knowledge_content_length CHECK 위반 / knowledge_category_fk FK 위반 | 오류가 난 트랜잭션은 실패 상태가 되므로 ROLLBACK 구간만 별도로 실행한다. SQL 문법 오류와 의도한 제약 오류를 구분한다. |
| 07 | 07_transaction_index.sql / 취소·확정·재접속·인덱스·실행계획 | 임시 수정 취소, id=2 본문 확정, 인덱스와 실행계획 확인 | SQL 기준 확정 본문: 실행 전에 가상환경을 확인합니다. 실제 재접속·인덱스·실행계획 출력은 미기록 | ROLLBACK과 COMMIT을 구분한다. 인덱스를 만들어도 작은 테이블에서는 순차 탐색이 선택될 수 있으므로 사용 여부와 성능은 실제 계획으로 판단한다. |

S0 → S1의 변경 내용과 인덱스 생성 여부도 기록합니다.
 - SQL 기준 데이터 변경: id=2의 본문만 '실행 전에 가상환경을 확인합니다.'로 확정하며, 기존 지식 수 3개와 ID·제목·분류는 유지한다.
 - 인덱스: category_id에 idx_knowledge_category_id를 생성하는 코드가 작성되어 있다. 실제 생성 여부와 정의는 pg_indexes 조회 결과로 기록할 항목이다.
 - 실행계획: EXPLAIN과 EXPLAIN (ANALYZE, BUFFERS)를 사용한다. 실제 실행계획·시간 출력이 없어 성능 개선을 수치로 단정하지 않는다.
롤백 검증은 행의 ID·내용·수를 대조합니다. 다음 ID 숫자의 불변을 요구하지 않습니다.
07 재접속 때 현재 DB와 새 연결 여부를 확인합니다. DB 서버 종료·장애 시험은 하지 않습니다.
