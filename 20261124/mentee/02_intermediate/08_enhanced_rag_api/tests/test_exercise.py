"""Day27 08_enhanced_rag_api 실행과 핵심 결과 확인 (네트워크 없이 --offline 테스트)."""
from pathlib import Path
import subprocess,sys,json
import pytest

# tests 폴더에서 한 단계 더 올라가 main.py가 들어 있는 실습 폴더를 찾습니다.
HERE=Path(__file__).resolve().parents[1]
ROLE_ROOT=next(p for p in HERE.parents if (p/'coursekit').is_dir())

@pytest.mark.parametrize('run_idx',[1])
def test_offline_contract(run_idx):
    # 1. langchain_core가 설치됐는지 확인하고, 같은 Python으로 main.py --offline을 별도 실행합니다.
    pytest.importorskip("langchain_core")
    res=subprocess.run([sys.executable,str(HERE/'main.py'),'--offline'],
                       cwd=ROLE_ROOT,capture_output=True,text=True,encoding='utf-8')
    assert res.returncode==0,res.stdout+'\n'+res.stderr

    # 2. 저장된 JSON을 읽고 네 요청의 HTTP 상태 코드가 순서대로 맞는지 봅니다.
    #    Q01은 답변 200, Q06은 근거 부족 응답 200, 잘못된 후보 수는 422, 재정렬 실패 예시는 502입니다.
    path=HERE/'outputs'/'day27_08_fixture.json'
    assert path.exists(),path
    data=json.loads(path.read_text(encoding='utf-8'))
    assert 'rows' in data
    assert data['execution_mode'] in ('fixed_diagnostic_cases','offline_fixture')
    assert [r['http_status'] for r in data['rows']] == [200,200,422,502]
