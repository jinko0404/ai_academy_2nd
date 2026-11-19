"""정상/경계 사례를 검증합니다. mentee는 TODO를 채우기 전 실패가 정상입니다."""
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import DATA,load_json
from coursekit.documents import by_doc,parse_one
import tasks as m


# 세 파싱 상태의 건수를 확인하고, HTML 표에서 열 이름과 값이 함께 남는지 검사합니다.
def test_contract():
    rows=load_json(DATA/'manifest.json'); results=[m.parse_dispatch(r) for r in rows]
    assert [r['status'] for r in results].count('parsed')==4
    assert [r['status'] for r in results].count('needs_ocr')==1
    assert [r['status'] for r in results].count('unsupported')==1
    html=next(r for r in results if r['doc']['doc_id']=='DOC-FACILITY')
    assert '시설: 실습실' in html['units'][0]['raw_text']
