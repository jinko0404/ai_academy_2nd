"""Day27 07_before_after 실행과 핵심 결과 확인 (네트워크 없이 --offline 테스트)."""
from pathlib import Path
import subprocess,sys,json
import pytest

# tests 폴더에서 한 단계 더 올라가 main.py가 들어 있는 실습 폴더를 찾습니다.
HERE=Path(__file__).resolve().parents[1]
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())

@pytest.mark.parametrize('run_idx',[1])
def test_offline_contract(run_idx):
    # 1. langchain_core가 설치됐는지 확인한 뒤, 테스트와 같은 Python으로 main.py --offline을 실행합니다.
    pytest.importorskip("langchain_core")
    res=subprocess.run([sys.executable,str(HERE/'main.py'),'--offline'],
                       cwd=ROLE_ROOT,capture_output=True,text=True,encoding='utf-8')
    assert res.returncode==0,res.stdout+'\n'+res.stderr

    # 2. 실행이 정상 종료되면 저장된 JSON을 열어 실행 모드와 비교 범위(k=2)를 확인합니다.
    path=HERE/'outputs'/'day27_07_fixture.json'
    assert path.exists(),path
    data=json.loads(path.read_text(encoding='utf-8'))
    assert 'row' in data
    assert data['execution_mode'] in ('fixed_diagnostic_cases','offline_fixture')
    assert data['row']['comparison']['k'] == 2
