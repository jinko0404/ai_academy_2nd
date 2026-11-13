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
| 2026.11.13 | LLM 기초, Token·Embedding·Transformer·Attention, Softmax·출력 선택, 답변 검증 | [학습 자료](./20261113/README.md) |

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
├── 20261112/
│   ├── README.md
│   ├── Day19_01/
│   │   └── mentee/01_basic/
│   │       ├── 01_FastAPI/
│   │       └── 02_PostgreSQL/
│   └── Day19_02/
│       └── mentee/01_기본필수/
│           ├── 01_PM01_연동시작/
│           ├── 02_PM02_CRUD/
│           └── 03_PM03_오류_트랜잭션/
└── 20261113/
    ├── README.md
    └── mentee/01_basic/
        ├── 01_ecosystem/
        ├── 02_tokenization/
        ├── 03_embeddings/
        ├── 04_transformer/
        ├── 05_attention/
        ├── 06_inference/
        ├── 07_limits/
        └── 08_daily_project/
```

- `README.md`: 날짜별 주요 학습 내용과 개념 정리
- `01_기본필수/`: 교육 과정의 기본 실습 및 필수 과제
- `02_응용/`: 기본 개념을 확장한 응용 실습 및 실험
- `03_심화/`: 응용을 넘어 직접 설계·검증하는 심화 실습 및 탐색
- `mentee/01_basic/`, `02_intermediate/`, `03_advanced/`: 날짜별 실습의 기본·응용·심화 구분
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
| LLM 기초 | 교육용 토큰화·패딩·문맥 예산, 벡터 유사도, Transformer shape, Causal Attention, 온도별 확률·Greedy 선택, 답변의 형식·근거·원본 불일치 분류 |

결과를 비교할 때는 데이터 분할, 전처리, 모델 설정과 평가 기준을 함께 명시합니다. 예측값과 정답을 대조하고, 개선 효과뿐 아니라 오류 사례와 실험의 한계도 정리합니다.

실제 실행 결과, 제공된 기록, 모의 응답(mock), 배포 설계를 구분해 표시하며, 실행하거나 측정하지 않은 항목은 미실행·미측정으로 남깁니다. 개별 실습의 코드, 산출물과 상세 해석은 각 날짜별 README에서 확인할 수 있습니다.

## 🛠️ 개발 환경

- Language: Python, SQL (PostgreSQL)
- Editor: Visual Studio Code
- Version Control: Git & GitHub
- 주요 라이브러리: Pandas, NumPy, scikit-learn
- 딥러닝·LLM 기초 계산 및 시각화: PyTorch, Matplotlib
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
**Last Updated:** 2026-11-13
