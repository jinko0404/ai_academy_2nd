# AI Academy 2nd — Learning Archive

> 청년취업사관학교 AI+아카데미 2기 학습 기록 및 실습 저장소

AI 교육 과정에서 학습한 개념과 실습 코드를 날짜별로 정리하는 저장소입니다.

단순한 코드 보관을 넘어, 기본 실습부터 응용 과제까지의 구현 과정과 실험 결과를 기록하고 학습 내용을 복습할 수 있도록 관리합니다.

## 📚 학습 기록

| 날짜 | 주요 학습 주제 | 자료 |
|---|---|---|
| 2026.10.26 | 머신러닝 기초, 분류·회귀 모델 비교 및 평가 | [학습 자료](./20261026/README.md) |
| 2026.10.27 | 결정 트리, Random Forest, XGBoost, 특성 중요도, 교차검증 | [학습 자료](./20261027/README.md) |
| 2026.10.28 | 평가지표(ROC-AUC·임계값), 특성 설계, K-Means, 최종 평가 | [학습 자료](./20261028/README.md) |
| 2026.10.29 | GPU·Tensor 기초, ANN 모델 구조, nn.Module과 nn.Sequential 비교 | [학습 자료](./20261029/README.md) |
| 2026.10.30 | PyTorch ANN 응용 실습 및 모델 성능 비교 | [학습 자료](./20261030/README.md) |
| 2026.11.02 | 이미지 Tensor, CNN·Pooling, 이미지 분류, 전이학습(ResNet18) | [학습 자료](./20261102/README.md) |
| 2026.11.03 | Sequence Tensor·Padding·Mask, RNN·LSTM, Autoencoder | [학습 자료](./20261103/README.md) |
| 2026.11.04 | OpenCV 이미지·영상 전처리, 색공간·필터·형태학, 데이터 증강 | [학습 자료](./20261104/README.md) |
| 2026.11.05 | 데이터셋 분할·누수 검사, 전이학습·Fine-tuning, 클래스 불균형 대응, 오류 분석·Grad-CAM | [학습 자료](./20261105/README.md) |
| 2026.11.06 | 객체 탐지·YOLO, 박스 좌표·IoU·NMS, mAP 평가, ByteTrack·BoT-SORT 추적 및 오류 분석 | [학습 자료](./20261106/README.md) |
| 2026.11.09 | OCR·문서 전처리, 정보 추출·구조화, VLM 문서 질의응답·근거 검증, 품질 평가 및 엣지 배포 | [학습 자료](./20261109/README.md) |
| 2026.11.10 | FastAPI·Uvicorn, HTTP·CRUD, Pydantic 입력 검증, JSON 저장·ID 관리, category 필터 | [학습 자료](./20261110/README.md) |
| 2026.11.11 | PostgreSQL·SQL CRUD, PK·FK·ERD, JOIN·집계, 제약조건, 트랜잭션·인덱스·실행계획 | [학습 자료](./20261111/README.md) |
| 2026.11.12 | FastAPI–PostgreSQL 연동, SQLAlchemy ORM·Session, 지식 CRUD, 오류 처리·트랜잭션 | [학습 자료](./20261112/README.md) |

학습 기록은 교육 진행에 따라 날짜별로 추가합니다.

## 🗂️ 저장소 구조

```text
ai_academy_2nd/
├── README.md
├── 20261026/
│   ├── README.md
│   ├── 01_기본필수/
│   └── 02_응용/
├── ...
├── 20261110/
│   ├── README.md
│   └── mentee/
│       ├── 01_basic/
│       ├── 02_intermediate/
│       └── 03_advanced/
├── 20261111/
│   ├── README_PostgreSQL.md
│   ├── mentee/
│   │   ├── 00_준비/
│   │   ├── 01_기본필수/
│   │   └── 02_응용/
│   └── result/
└── 20261112/
    ├── README.md
    ├── Day19_01/
    │   └── mentee/01_basic/
    │       ├── 01_FastAPI/
    │       └── 02_PostgreSQL/
    └── Day19_02/
        └── mentee/01_기본필수/
            ├── 01_PM01_연동시작/
            ├── 02_PM02_CRUD/
            └── 03_PM03_오류_트랜잭션/
```

- `README.md`: 날짜별 주요 학습 내용과 개념 정리
- `01_기본필수/`: 교육 과정의 기본 실습 및 필수 과제
- `02_응용/`: 기본 개념을 확장한 응용 실습 및 실험
- `03_심화/`: 응용을 넘어 직접 설계·검증하는 심화 실습 및 탐색
- `mentee/01_basic/`, `02_intermediate/`, `03_advanced/`: 11월 10일 실습의 기본·응용·심화 구분
- `Day19_01/`, `Day19_02/`: 11월 12일 오전의 API·DB 개별 실습과 오후의 API–DB 연동 실습
- `data/`: 실습에 사용되는 데이터와 데이터 분할 정보
- `outputs/`: 모델 비교, 예측 결과, 특성 중요도 등 실험 산출물
- `erd/`, `reports/`, `result/`: DB 관계 워크시트, SQL 학습·프로젝트 기록, 조회 결과 스냅샷

※ 날짜별 폴더의 실제 구성은 학습 주제와 과제에 따라 달라질 수 있습니다.

## 🔬 실습 및 결과 기록
 
실습에서는 구현한 코드와 실행 조건, 평가 결과, 오류 원인을 함께 기록합니다. 머신러닝·딥러닝·컴퓨터 비전·문서 처리와 API 개발을 다루며, 주제에 맞는 지표·시각화·요청 응답으로 결과를 확인합니다.

| 실습 영역 | 주요 기록 내용 |
|---|---|
| 머신러닝 | 분류·회귀 성능, 교차검증 결과, 특성 중요도, 파생 특성 효과, 군집 구조 해석 |
| 딥러닝 | ANN·CNN·RNN·LSTM·Autoencoder의 학습 결과, 손실 변화, 학습 설정에 따른 성능 차이 |
| 이미지 분류·전이학습 | 데이터 분할·누수 점검, Fine-tuning 조건, 클래스별 성능, 오분류 사례와 Grad-CAM 해석 |
| 이미지·영상 처리 | 색공간·필터·이진화·형태학·증강 적용 전후의 이미지와 처리 결과 비교 |
| 객체 탐지·추적 | 박스 좌표·IoU·NMS 확인, mAP 평가, 객체 추적 결과와 탐지·추적 오류 분석 |
| OCR·문서 이해 | 인식 원문·좌표, 추출 필드·표 구조, CER·WER·필드 정확도, VLM 응답의 원문 근거와 검토 사유 |
| 배포·성능 측정 | ONNX·TensorRT·Jetson 배포 흐름과 실행 환경, 실제 측정한 지연 시간·처리량, 추가 검증 항목 |
| API·백엔드 | FastAPI 지식 CRUD, Path·Query·Body 구분, Pydantic 검증, HTTP 상태코드, JSON 저장·ID 관리, category 필터와 재시작 후 데이터 유지 |
| 데이터베이스·SQL | PostgreSQL 접속, PK·FK와 1:N 관계, CRUD·RETURNING, JOIN·분류별 집계, 제약 오류와 ROLLBACK, COMMIT·재접속, 인덱스·실행계획 |
| API–DB 연동 | SQLAlchemy ORM 매핑, 요청별 Session, category 문자열과 FK 연결, CRUD 트랜잭션, API 응답 ID와 SQL 행 대조, DB 상태 확인 |

결과를 비교할 때는 데이터 분할, 전처리, 모델 설정과 평가 기준을 함께 명시합니다. 예측값과 정답을 대조하고, 개선 효과뿐 아니라 오류 사례와 실험의 한계도 정리합니다.

실제 실행 결과, 제공된 기록, 모의 응답(mock), 배포 설계를 구분해 표시하며, 실행하거나 측정하지 않은 항목은 미실행·미측정으로 남깁니다. 개별 실습의 코드, 산출물과 상세 해석은 각 날짜별 README에서 확인할 수 있습니다.

### 11월 10일 · 지식 관리 API 실습

FastAPI로 지식을 등록·조회·수정·삭제하는 API를 구현했습니다. 요청의 입력 검증, 처리 함수, 파일 저장의 역할을 구분하고 응답 상태코드와 JSON 파일의 변경을 함께 확인했습니다. 응용에서는 `queries.py`로 조회 계산을 분리하고 `category`가 정확히 일치하는 목록을 반환하도록 확장했습니다.

- [기본필수 통합 프로젝트](./20261110/mentee/01_basic/07_project/README.md) · [실습 기록](./20261110/mentee/01_basic/07_project/실습07_기록.md) · [일일프로젝트 기록](./20261110/mentee/01_basic/07_project/일일프로젝트_기록.md)
- [응용 통합 프로젝트](./20261110/mentee/02_intermediate/07_project/README.md) · [실습 기록](./20261110/mentee/02_intermediate/07_project/실습07_기록.md) · [일일프로젝트 기록](./20261110/mentee/02_intermediate/07_project/일일프로젝트_기록.md)

학습 범위는 원본 지식의 CRUD 관리이며, RAG 검색·LLM 답변 생성·Agent 실행은 후속 기능으로 구분합니다.

### 11월 11일 · PostgreSQL 지식 관리 DB 실습

`categories`와 `knowledge` 두 테이블을 중심으로 PK·FK와 1:N 관계를 정리하고 SQL 조회·추가·수정·삭제를 작성했습니다. JOIN으로 지식에 분류명을 붙이고, LEFT JOIN과 COUNT로 지식이 없는 분류까지 집계하는 흐름을 학습했습니다.

제약조건 위반 시 SQLSTATE를 확인하고 ROLLBACK으로 회복하는 절차, COMMIT으로 변경을 확정하는 과정을 다뤘습니다. 인덱스와 EXPLAIN을 통해 실행계획을 살펴보는 방법도 학습했습니다. 일일프로젝트는 A 유지·B 수정·C 삭제를 나누고 기존 행 스냅샷과 최종 상태를 비교하는 구조로 작성했습니다.

- [기본필수 실습 안내](./20261111/mentee/01_기본필수/README.md) · [학습 기록](./20261111/mentee/01_기본필수/reports/learning_note.md) · [일일프로젝트 기록](./20261111/mentee/01_기본필수/reports/daily_note.md)
- [응용 실습 안내](./20261111/mentee/02_응용/README.md) · [관계 워크시트](./20261111/mentee/02_응용/erd/관계_워크시트.md)
- [기존 지식 스냅샷](./20261111/result/before_knowledge.csv)

### 11월 12일 · FastAPI–PostgreSQL 연동과 트랜잭션 실습

오전에는 JSON 파일을 사용하는 FastAPI 지식 관리 API와 PostgreSQL 테이블 구축·SQL CRUD를 각각 실습했습니다. 오후에는 SQLAlchemy ORM과 Psycopg 드라이버를 통해 FastAPI를 PostgreSQL에 연결하고, `categories`와 `knowledge` 테이블을 사용하는 지식 CRUD를 작성했습니다.

`Engine`과 연결 풀은 재사용하고, `get_db` 의존성으로 요청별 `Session`을 생성·정리하는 구조를 다뤘습니다. API의 `category` 문자열로 분류를 조회하거나 생성한 뒤 `category_id`로 연결하고, `flush()`로 ID를 확보하는 단계와 `commit()`으로 저장을 확정하는 단계를 구분했습니다.

오류·트랜잭션 실습에서는 분류와 지식 변경을 하나의 작업으로 묶고, 예외가 발생하면 `rollback()` 후 다시 예외를 전달하도록 구현했습니다. 없는 지식의 404, 입력 검증의 422, 무결성 오류의 409 처리와 `/health`의 `SELECT 1`·연결 오류 503 처리를 살펴봤습니다. API 응답과 SQL에서 같은 ID의 행을 대조하는 것이 연동 검증 기준이며, 실제 DB 제약 위반 후 rollback·Session 복구 확인은 입력 검증과 별도의 검증 항목으로 구분합니다.

- [11월 12일 개념 정리](./20261112/README.md) · [오전 기본필수 안내](./20261112/Day19_01/mentee/01_basic/README.md)
- [오후 연동 시작](./20261112/Day19_02/mentee/01_기본필수/01_PM01_연동시작/README.md) · [CRUD 실습](./20261112/Day19_02/mentee/01_기본필수/02_PM02_CRUD/README.md) · [오류·트랜잭션 실습](./20261112/Day19_02/mentee/01_기본필수/03_PM03_오류_트랜잭션/README.md)

개념 정리에는 Alembic Migration과 RAG·Agent로의 확장 범위도 포함되어 있습니다. 해당 내용은 현재 실습 코드의 구현·실행 결과와 구분해 참고합니다.

## 🛠️ 개발 환경

- Language: Python, SQL (PostgreSQL)
- Editor: Visual Studio Code
- Version Control: Git & GitHub
- 주요 라이브러리: Pandas, NumPy, scikit-learn
- API 개발·확인: FastAPI, Uvicorn, Pydantic, Postman, Swagger UI(`/docs`)
- 데이터베이스: PostgreSQL, pgAdmin Query Tool, psql
- API–DB 연동: SQLAlchemy, Psycopg, pydantic-settings
- 추가 라이브러리: 실습 주제에 따라 사용

## 📝 학습 기록 원칙

1. 날짜별로 학습 자료를 구분합니다.
2. 기본 필수 과정과 응용 과정을 구분하여 관리합니다.
3. 코드의 목적과 주요 개념을 이해하기 쉽게 기록합니다.
4. 실험 결과뿐 아니라 결과 해석과 한계도 함께 정리합니다.
5. 학습이 진행됨에 따라 새로운 자료와 기록을 지속적으로 추가합니다.

---

**Status:** In Progress  
**Purpose:** AI/ML Learning Archive  
**Last Updated:** 2026-11-12
