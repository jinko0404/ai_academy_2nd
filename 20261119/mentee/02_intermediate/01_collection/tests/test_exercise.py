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


# 원문 목록의 6건이 모두 집계되고, 문서 ID를 일부러 중복하면 ValueError가 발생하는지 검사합니다.
def test_contract():
    rows=load_json(DATA/'manifest.json');r=m.collect_all(rows)
    assert r['total']==6 and r['counts']=={'collected':5,'unsupported':1,'read_error':0}
    with pytest.raises(ValueError):m.sort_rows(rows+[rows[0]])
