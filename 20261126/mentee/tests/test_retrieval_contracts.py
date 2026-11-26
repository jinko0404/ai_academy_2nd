"""실제 DB 대신 테스트 더블로 검사하는 I/O 계약. 실제 연동 성공으로 오해하지 마세요."""
from contextlib import contextmanager
from types import SimpleNamespace
import pytest


def test_select_query_uses_only_existing_report_tables_and_parameters(monkeypatch):
    from coursekit import db
    seen={}
    class FakeRows:
        def mappings(self):return self
        def all(self):return []
    class FakeConn:
        def execute(self,statement,params):
            seen['sql']=str(statement)
            seen['params']=params
            return FakeRows()
    @contextmanager
    def conn():
        yield FakeConn()
    monkeypatch.setattr(db,'readonly_connection',conn)
    monkeypatch.setattr(db,'settings',lambda:SimpleNamespace(
        embedding_model='intfloat/multilingual-e5-small',
        allowed_categories=('RESEARCH_PLAN','MEETING')))
    assert db.select_semantic_candidates([0.0]*384,'ALL',3)==[]
    assert 'report_read_view' in seen['sql'] and 'report_embeddings' in seen['sql']
    assert 'category_code IN' in seen['sql']
    assert 'indexed_revision=v.revision' in seen['sql']
    assert 'sha256' in seen['sql']
    assert seen['params']['allowed_categories']==('RESEARCH_PLAN','MEETING')
    assert 'CREATE' not in seen['sql'].upper() and 'DELETE' not in seen['sql'].upper()


def test_rag_pipeline_separates_candidates_ranked_and_context(monkeypatch):
    from coursekit import rag
    def fake_rows(vec,category,k):
        return [
            {'report_id':63,'category_code':'RESEARCH_PLAN','title':'RAG 평가 계획',
             'overview':'실무 원문', 'content':'근거 문서 A', 'cosine_distance':0.15},
            {'report_id':66,'category_code':'RESEARCH_PLAN','title':'Agent 안전성 연구',
             'overview':'실무 원문', 'content':'근거 문서 B', 'cosine_distance':0.18},
        ]
    monkeypatch.setattr(rag,'settings',lambda:SimpleNamespace(
        allowed_categories=('RESEARCH_PLAN',),min_similarity=0.5))
    monkeypatch.setattr(rag,'encode_query',lambda query:[0]*384)
    monkeypatch.setattr(rag,'select_semantic_candidates',fake_rows)
    monkeypatch.setattr(rag,'rerank',lambda query,rows:list(reversed(rows)))
    out=rag.retrieve_reports('RAG 근거 검색','RESEARCH_PLAN',candidate_k=3,context_k=1)
    assert out['candidate_ids']==[63,66]
    assert out['ranked_ids']==[66,63]
    assert out['included_ids']==[66]
    assert out['sources'][0]['source_id']=='RPT-66'
    assert '[RPT-66]' in out['context'] and '[RPT-63]' not in out['context']


def test_rag_rejects_category_outside_server_policy(monkeypatch):
    from coursekit import rag
    monkeypatch.setattr(rag,'settings',lambda:SimpleNamespace(
        allowed_categories=('RESEARCH_PLAN',),min_similarity=0.5))
    with pytest.raises(PermissionError):
        rag.retrieve_reports('보고서 검색','MEETING')


def test_no_ready_vector_returns_clear_error_not_fake_search(monkeypatch):
    from coursekit import rag,db
    monkeypatch.setattr(rag,'settings',lambda:SimpleNamespace(
        allowed_categories=('RESEARCH_PLAN',),min_similarity=0.5))
    monkeypatch.setattr(rag,'encode_query',lambda q:[0]*384)
    monkeypatch.setattr(rag,'select_semantic_candidates',lambda *args:[])
    monkeypatch.setattr(db,'preflight',lambda:{'fresh_vectors':0})
    with pytest.raises(RuntimeError,match='유효한 E5 벡터가 0건'):
        rag.retrieve_reports('RAG','RESEARCH_PLAN')
