"""실행 없이 확인할 수 있는 계약만 검사하는 테스트.
실제 report_agent_lab·GPU·LLM API를 가짜 호출로 통과했다고 주장하지 않습니다.
"""
from pathlib import Path
import ast
import pytest

from coursekit import contracts
from coursekit.settings import CATEGORIES


@pytest.mark.parametrize('value', [1, 63, '63', 180, 1000])
def test_report_id_valid(value):
    assert contracts.report_id_from(value) > 0


@pytest.mark.parametrize('value', [True, False, 0, -1, None, 1.1, '01', '1;DROP DATABASE', 'hello', ''])
def test_report_id_rejects_invalid(value):
    with pytest.raises(ValueError):
        contracts.report_id_from(value)


@pytest.mark.parametrize('category', ['ALL', *CATEGORIES])
def test_category_codes_match_11_20_schema(category):
    q, cat=contracts.search_request('RAG 인용 근거',category)
    assert q=='RAG 인용 근거' and cat==category


def test_tool_call_model_request_is_not_execution():
    req=contracts.tool_call_request('lookup_status', {'report_id':63}, 'call_abc')
    assert 'content' not in req
    result=contracts.tool_result_message(req, {'status':'ok','report_id':63})
    assert result['tool_call_id']=='call_abc' and result['content']['status']=='ok'


def test_unregistered_tool_and_blank_id_fail():
    with pytest.raises(ValueError):contracts.tool_call_request('delete_report',{},'call_01')
    with pytest.raises(ValueError):contracts.tool_call_request('lookup_status',{},' ')


def test_source_contract_validates_server_evidence_ids():
    good={'status':'answered','sources':[{'source_id':'RPT-63'}]}
    assert contracts.validate_source_contract(good,{'RPT-63'})
    with pytest.raises(ValueError):contracts.validate_source_contract(good,{'RPT-64'})
    with pytest.raises(ValueError):contracts.validate_source_contract(
        {'status':'insufficient_context','sources':[{'source_id':'RPT-63'}]}, {'RPT-63'})
    with pytest.raises(ValueError):contracts.validate_source_contract(
        {'status':'answered','sources':[{'source_id':'RPT-63'},{'source_id':'RPT-63'}]}, {'RPT-63'})


def test_messages_requested_vs_returned_are_distinct_counts():
    msgs=[{'tool_calls':[{'name':'lookup_status','args':{'report_id':63},'id':'call_1'}]},
          {'type':'ToolMessage','tool_call_id':'call_1','name':'lookup_status','content':{'status':'ok'}}]
    assert contracts.count_tool_calls(msgs)=={'requested':1,'executed_messages':1,'unmatched':[]}


def test_audit_does_not_emit_question_or_key():
    got=contracts.sanitize_event({'event':'invoke','report_id':63,'api_key':'secret','question':'sensitive','elapsed_ms':2.0})
    assert got=={'event':'invoke','report_id':63,'elapsed_ms':2.0}


def test_package_has_only_readonly_diagnostic_sql():
    root=Path(__file__).resolve().parents[1]
    sqls=list((root/'sql').glob('*.sql'))
    assert len(sqls)==1
    sql=sqls[0].read_text(encoding='utf8')
    # 주석은 설명일 뿐, SQL 실행문에 CREATE/DROP 등 DDL/DML이 없어야 합니다.
    statements='\n'.join(line for line in sql.splitlines() if not line.lstrip().startswith('--')).upper()
    assert 'CREATE DATABASE' not in statements
    assert 'DROP ' not in statements
    assert 'UPDATE ' not in statements
    assert 'INSERT ' not in statements
    assert 'DELETE ' not in statements
    assert 'REPORT_READ_VIEW' in statements


def test_all_labs_and_python_compile():
    root=Path(__file__).resolve().parents[1]
    for level in ('01_basic','02_intermediate','03_advanced'):
        labs=sorted((root/level).glob('0*_*'))
        assert len(labs)==8
        for folder in labs:
            assert (folder/'main.py').is_file() and (folder/'tasks.py').is_file() and (folder/'README.md').is_file()
    for path in root.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf8'))


def test_cuda_is_fail_fast_not_cpu_fallback(monkeypatch):
    import torch
    from coursekit.gpu import require_cuda
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CPU 폴백 금지'):
        require_cuda()


def test_api_must_reject_unverified_citation():
    root=Path(__file__).resolve().parents[1]
    import importlib.util, sys
    path=root/'01_basic'/'08_end_to_end'/'tasks.py'
    spec=importlib.util.spec_from_file_location('day29_api_validation',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bad={'status':'completed','answer':'[RPT-999]','tool_trace':[],
         'citation_ids_valid':False,'verified_evidence_ids':['RPT-63']}
    if 'NotImplementedError' in (root/'01_basic'/'08_end_to_end'/'tasks.py').read_text(encoding='utf8'):
        pytest.skip('수강생 TODO를 아직 채우지 않았습니다.')
    with pytest.raises(RuntimeError,match='UNVERIFIED_CITATION'):
        module.api_response(bad)


def test_advanced_status_tool_is_category_scoped_without_user_acl():
    root=Path(__file__).resolve().parents[1]
    import importlib.util,sys
    folder=root/'03_advanced'/'05_db_status_tool'
    if 'NotImplementedError' in (folder/'policy_05.py').read_text(encoding='utf8'):
        pytest.skip('수강생 심화 TODO 작업 전입니다.')
    sys.path.insert(0,str(folder))
    try:
        spec=importlib.util.spec_from_file_location('status_adv_test',folder/'tasks.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        r={'status':'ok','report_id':63,'category_code':'RESEARCH_PLAN',
           'revision':1,'index_status':'pending','vector_fresh':False}
        assert m.enforce_read_policy(r,{'RESEARCH_PLAN'})['report_id']==63
        assert m.enforce_read_policy(r,{'MEETING'})['status']=='not_found_or_forbidden'
    finally:sys.path.pop(0)
