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


# 검증 목표 — 기본 실행에서 준비한 청크에 원문 ID와 E5 입력 형식이 제대로 담겼는지 확인합니다.
# 테스트는 prepare()만 호출하므로 E5 모델과 DB 연결이 필요 없습니다.
def test_contract():
    # 과제 안내 문서와 그 문서에서 만들어진 청크 목록을 함께 받습니다.
    doc,chunks=m.prepare('DOC-ASSIGN')
    # 청크 목록이 비어 있으면 실패하고, all()로 모든 청크가 이 문서의 ID를 갖는지 확인합니다.
    assert chunks and all(c['doc_id']==doc['doc_id'] for c in chunks)
    # E5에 넣을 문서용 문자열에는 후행 공백까지 포함한 'passage: ' 접두어가 붙어야 합니다.
    assert all(c['embedding_text'].startswith('passage: ') for c in chunks)
