"""Day27 06_langchain_pipeline 실행과 핵심 결과 확인 (네트워크 없이 --offline 테스트)."""
from pathlib import Path
import subprocess,sys,json
import pytest

# 이 테스트는 실습을 오프라인으로 한 번 실행해 정상 종료하고 결과 JSON을 쓰는지 확인합니다.
# 테스트 파일은 tests 폴더 안에 있으므로 parents[1]로 한 단계 더 올라가 실습 폴더를 찾습니다.
HERE=Path(__file__).resolve().parents[1]
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())

@pytest.mark.parametrize('run_idx',[1])
def test_offline_contract(run_idx):
    # 1. langchain_core가 설치된 환경에서 실행합니다. 패키지가 없으면 pytest가 이 테스트를 건너뜁니다.
    pytest.importorskip("langchain_core")

    # 2. 현재 테스트와 같은 Python(sys.executable)으로 main.py --offline을 별도 실행합니다.
    #    종료 코드 0이면 성공입니다. 실패 시 원인을 보도록 터미널 출력과 오류를 함께 표시합니다.
    res=subprocess.run([sys.executable,str(HERE/'main.py'),'--offline'],
                       cwd=ROLE_ROOT,capture_output=True,text=True,encoding='utf-8')
    assert res.returncode==0,res.stdout+'\n'+res.stderr

    # 3. 저장된 JSON을 열어 결과 묶음(row), 실행 모드, 원래 질문이 기록됐는지 확인합니다.
    path=HERE/'outputs'/'day27_06_fixture.json'
    assert path.exists(),path
    data=json.loads(path.read_text(encoding='utf-8'))
    assert 'row' in data
    assert data['execution_mode'] in ('fixed_diagnostic_cases','offline_fixture')
    assert data['row']['trace']['original_question']
