# Windows 11 · 기존 DB와 GPU 준비

1. VS Code에서 11/20 가상환경을 선택. Git Bash에서 `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`로 CUDA 확인.
2. GPU가 False이면 CPU로 진행하지 않습니다. 드라이버/venv의 CUDA 버전을 확인하고 강사와 맞춥니다.
3. `pip install -r requirements.txt` 실행 시 기존 CUDA PyTorch가 CPU 빌드로 교체되지 않게 주의합니다.
4. 같은 `report_agent_lab`의 비밀번호를 `.env`에 입력. `.env`는 버전관리에서 제외.
5. pgAdmin에서 `report_agent_lab`을 열고 `sql/check_existing_db_readonly.sql`의 SELECT만 실행.
6. `report_embeddings`가 0행이면 11/20 실습의 `scripts/index_overviews.py`가 완료되었는지 확인. 새 DB·새 Schema·새 Seed 생성 금지.
7. 보고서별 `ready`·`pending` 상태는 DB의 현재값입니다. 초기 데이터는 180개이며 이후 CRUD 때문에 숫자는 달라질 수 있습니다.
8. `LLM_MODEL`에 본인 API 키로 호출 가능한 Tool Calling 지원 모델을 입력합니다. 인증 오류 401은 토큰 상태 문제이므로 다른 API 요청으로 확인합니다.
9. Basic 01→08 순서로 개념·실제 DB·Tool/Agent를 연결합니다. `--check`는 구조 확인이며 실제 연동 검증이 아닙니다.
10. FastAPI는 로컬 루프백 127.0.0.1에서만 실습합니다. DB에 사용자별 소유권 정보가 없으므로 외부 배포하지 않습니다.
