# Day28 · Windows 11 / VS Code / CUDA / PostgreSQL 환경설정

이 지침은 `mentee` 또는 `mentor` 폴더가 VS Code의 **열린 폴더**일 때를 기준으로 합니다. Windows 11에서 PostgreSQL을 직접 설치한 기존 환경을 사용하며 Docker/WSL을 필수로 요구하지 않습니다.

## 0. 확인할 환경

- Python **3.12** 권장 (`py -0p`, `py -3.12 --version`)
- VS Code → **Terminal → New Terminal** (Git Bash 권장)
- NVIDIA GPU와 최신 호환 Driver (`nvidia-smi`에서 GPU 이름과 Driver 버전 확인)
- 기존 PostgreSQL과 pgAdmin; `knowledge_lab` DB, pgvector 서버 Extension
- OpenAI API를 사용할 경우 유효한 API Key와 실제 접근 가능한 모델

## 1. 가상환경 생성 (Git Bash)

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python --version
python -m pip --version
```

VS Code `Ctrl+Shift+P` → **Python: Select Interpreter** → 현재 폴더의 `.venv\Scripts\python.exe` 선택. 다른 `.venv`와 혼동하지 마세요.

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

PowerShell 실행정책에 막히면 Git Bash를 이용하거나 정확한 `.venv\Scripts\python.exe`를 지정하세요.

## 2. PyTorch CUDA Build 먼저 설치

`https://pytorch.org/get-started/locally/`에서 Windows / Pip / 현재 Python / GPU·Driver에 맞는 CUDA 옵션을 선택하고 **사이트가 제공하는 설치 명령을 그대로 실행**하세요.

```bash
nvidia-smi
python -c "import torch; print('torch:',torch.__version__); print('CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"
```

`CUDA available: True`가 아니면 실제 실습(06~08 `--live`)을 실행하기 전 GPU 환경부터 해결합니다. E5와 Cross-Encoder에 **CPU 자동 Fallback을 두지 않았습니다.**

## 3. 강의 패키지 설치

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements_test.txt
python -m pip install -r requirements_live.txt
python -m pip check
python tools/check_environment.py
python tools/validate_dataset.py
```

- `requirements.txt`: `langchain-core`, `langchain-openai`, FastAPI, Pydantic, dotenv 등
- `requirements_live.txt`: GPU 모델 경로에서 필요한 sentence-transformers, NumPy, psycopg, pgvector Python Adapter
- `pip install pgvector`는 Python Adapter이며, **PostgreSQL 서버의 `vector` Extension을 설치하는 명령이 아닙니다.**
- 회사 환경에서 Hugging Face 모델 다운로드/외부 API 호출이 차단되면 인터넷 허용·보안 정책을 확인하세요.

## 4. 비밀 설정 (`.env`를 Git에 올리지 않음)

```bash
cp .env.example .env
```

`.env`에서 `OPENAI_API_KEY`, `LLM_MODEL`, `PGPASSWORD`만 실제 값으로 변경합니다. `RAG_SCHEMA=d27_rag`는 전날의 기존 Corpus를 이어받기 위한 설계입니다.

```dotenv
OPENAI_API_KEY=CHANGE_ME
LLM_MODEL=CHANGE_ME
PGHOST=localhost
PGPORT=5432
PGDATABASE=knowledge_lab
PGUSER=postgres
PGPASSWORD=CHANGE_ME
RAG_SCHEMA=d27_rag
LANGSMITH_TRACING=false
```

실제 값은 README·실행 로그·Issue·GitLab Merge Request에 붙이지 마세요. `LANGSMITH_TRACING`은 기본 꺼짐입니다. 선택적으로 추적할 경우 내부 데이터 반출 정책과 비밀정보 마스킹을 먼저 확인해야 합니다.

## 5. pgAdmin에서 pgvector 서버 Extension 확인

pgAdmin → Servers → Databases → `knowledge_lab` → Query Tool:

```sql
SELECT version();
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
SELECT name, default_version, installed_version
FROM pg_available_extensions WHERE name = 'vector';
```

확장이 설치되어 있고 권한을 가진 경우:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

`extension "vector" is not available`이면 현재 PostgreSQL 버전에 맞는 **Windows 서버 확장**을 별도로 설치해야 합니다. Python 패키지 재설치로는 해결되지 않습니다.

## 6. Day27 평가용 Corpus 준비 (실제 GPU Embedding + DB)

```bash
# 현재 DB는 건드리지 않는 계획 출력
python tools/setup_db.py

# Day27과 동일한 합성 Corpus 24건을 현재 d27_rag Schema에 준비
# CUDA E5 인코딩과 PostgreSQL 연결이 실제 실행되며, 환경에 따라 시간이 소요됩니다.
python tools/setup_db.py --apply
```

- `intfloat/multilingual-e5-small`, 384차원, `query: `와 `passage: ` Prefix 및 정규화 계약을 Day27과 동일하게 유지합니다.
- 전날 DB를 이용할 경우 동일 모델·Revision·Schema·문서 버전을 먼저 확인합니다.
- `d27_rag`는 기존 실습 전용 Schema입니다. `public`의 다른 테이블이나 학생 데이터베이스를 삭제하지 마세요.
- 이 작업은 교육용 Corpus 업로드이지 Gold Label을 검색 데이터에 삽입하는 일이 아닙니다.

## 7. 실습 실행 (Git Bash, 역할 폴더 기준)

```bash
# 기본은 무료·오프라인 계산 연습
python 01_basic/01_failure_diagnosis/main.py --offline
python 01_basic/03_retrieval_metrics/main.py --offline
python 01_basic/06_langchain_runner/main.py --offline
python 01_basic/08_agent_quality_report/main.py --offline

# 실제 모델·DB·GPU·유료 LLM 연결 (실습 06~08만)
python tools/check_environment.py --live
python 01_basic/06_langchain_runner/main.py --live
python 01_basic/08_agent_quality_report/main.py --live
```

**중요:** `--live`는 비용이 발생할 수 있습니다. Q01 등 몇 개의 질문을 실행하고 실제 응답이 로컬 Corpus를 참조하는지 확인하세요. 실행 모드가 다른 결과의 평균을 함께 계산하지 마세요. Python 코드는 실제 GPU에서 테스트하려면 해당 PC에서 실행해야 합니다.

`outputs/result.json`에 실행 모드와 질문·출처·평가값이 기록됩니다. 실습 08 Intermediate/Advanced는 `outputs/report.md`도 생성합니다.

## 8. 전체 검증 및 리포트 CSV

```bash
python -m pytest -q
python tools/run_all.py --level 01_basic --tests
python tools/export_report_csv.py 01_basic/08_agent_quality_report/outputs/result.json
```

- 수강생용은 TODO가 미완성이므로 해당 실습 테스트가 **실패하는 것이 정상**입니다. TODO 완성 후 통과시킵니다.
- `pytest.ini`의 `--import-mode=importlib` 설정은 같은 이름의 `tests/test_exercise.py`끼리 충돌하지 않도록 합니다.
- **CSV의 실측 근거**는 실행 모드/환경/Question ID로 추적하며, Fixture 결과는 성능 보고서로 제출하면 안 됩니다.

## 9. 대표적인 오류

| 현상 | 원인/해결 |
|---|---|
| `ModuleNotFoundError: langchain_core` | 현재 Python Interpreter·`.venv`를 확인하고 `requirements.txt` 설치 |
| `torch.cuda.is_available() == False` | Driver·PyTorch CUDA Wheel·환경 가상환경 확인. CPU 자동 전환 금지 |
| `CUDA out of memory` | E5/Reranker 중복 로드·모델/Batch/입력 크기·VRAM 점검 |
| `psycopg.OperationalError` | Windows PostgreSQL 서버/포트/비밀번호/DB 이름 점검 |
| `extension vector is not available` | pgvector **서버** Extension 설치/버전 일치 확인 |
| `AuthenticationError`, `token_invalidated` | 유효한 API Key 발급·환경변수·모델 사용 권한 점검 |
| HTTP 429 | 호출 한도·요금제·일시 요청 속도 점검 |
| `Gold leakage` | Target 함수에 Gold/Reference Answer 전달 금지 |
| `unjudged relevance` | 관련성 0과 미판정을 분리하고 판단 레이블 검수 |
| Trace 필드 누락 | 실행 대상이 `candidate_ids`, `ranked_ids`, `included_ids`를 반환하는지 검사 |
| No-answer Recall=None | **의도된 정책**. No-answer 성능은 별도 평가 |

## 10. 안전 및 다음 날 Agent 준비

고정 교육용 점수가 높아도 실제 Agent 서비스 배포 승인이 아닙니다. 데이터 접근제어, Tool 호출 권한, Grounding, Human Review, 오류·실행시간·비용을 별도로 검증해야 합니다.

## 11. 선택적 LLM-as-a-Judge (실습 05 심화 확장)

실제 모델이 제공된 Context·Claim을 평가하는 경험이 필요할 때만 다음을 실행합니다.

```bash
python tools/run_judge_live.py --case-id G01 --confirm-paid-api
```

`G01`~`G03`은 **합성 교육용 사례**입니다. 사람의 `supported` 판정은 Judge 모델로 전달하지 않습니다. 모델은 `JudgeReview` Pydantic Structured Output을 반환하며 `tools/outputs/judge_G01.json`에 저장합니다. 유료 LLM API를 실제 호출하므로 기본 실습에 자동 포함하지 않습니다. **LLM Judge는 사람의 최종 검수를 대신하지 않습니다.**
