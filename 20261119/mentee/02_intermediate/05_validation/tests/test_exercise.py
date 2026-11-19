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


# 검증 목표 — 누락은 커버리지를 낮추고, 정상 겹침은 통과하고, 같은 ID가 나오면 오류로 기록되는지 확인합니다.
def test_contract():
    # (0, 8)과 (10, 20)은 인덱스 8과 9를 건너뛰므로 18/20이 기대값입니다.
    assert m.coverage(20,[(0,8),(10,20)])==.9
    # 청크 사이에 겹친 곳이 있어도 누락과 ID 중복이 없다면 valid는 True입니다.
    assert m.make_report(20,[(0,8),(6,14),(12,20)],['A','B','C'])['valid']
    # ID A가 두 번 들어간 예제에서 오류 목록에 A가 한 번 기록되는지 확인합니다.
    assert m.make_report(20,[(0,8),(0,8),(8,20)],['A','A','C'])['errors']['duplicate_ids']==['A']
