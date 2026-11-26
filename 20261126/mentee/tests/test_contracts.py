"""정답 로직 자체의 오프라인 단위 검증. 실제 GPU/DB/LLM 성공과 구분합니다."""
from pathlib import Path
import importlib.util
import pytest
ROOT=Path(__file__).resolve().parents[1]

def _module(path):
    spec=importlib.util.spec_from_file_location('verify_'+path.parents[0].name+'_'+path.parents[1].name,path)
    import sys
    sys.path.insert(0,str(path.parent))
    try:
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    finally:sys.path.pop(0)

def test_01_basic_01_agent_vs_rag():
    t=_module(ROOT/'01_basic'/'01_agent_vs_rag'/'tasks.py')
    assert t.classify_request('보고서 인덱싱 상태')=='agent' or t.classify_request('인덱싱 상태')=='db'
    p=t.plan_for_request('근거 보고서 찾아서 63번 인덱싱 상태 알려줘')
    assert p['strategy']=='agent' and set(p['tools'])=={'rag_search','lookup_status'}

def test_02_intermediate_01_agent_vs_rag():
    t=_module(ROOT/'02_intermediate'/'01_agent_vs_rag'/'tasks.py')
    assert t.classify_request('보고서 인덱싱 상태')=='agent' or t.classify_request('인덱싱 상태')=='db'
    p=t.plan_for_request('근거 보고서 찾아서 63번 인덱싱 상태 알려줘')
    assert p['strategy']=='agent' and set(p['tools'])=={'rag_search','lookup_status'}

def test_03_advanced_01_agent_vs_rag():
    t=_module(ROOT/'03_advanced'/'01_agent_vs_rag'/'tasks.py')
    assert t.classify_request('보고서 인덱싱 상태')=='agent' or t.classify_request('인덱싱 상태')=='db'
    p=t.plan_for_request('근거 보고서 찾아서 63번 인덱싱 상태 알려줘')
    assert p['strategy']=='agent' and set(p['tools'])=={'rag_search','lookup_status'}

def test_01_basic_02_tool_message():
    t=_module(ROOT/'01_basic'/'02_tool_message'/'tasks.py')
    call=t.make_request('lookup_status',{'report_id':63},'call_001')
    msg=t.match_result(call,{'status':'ok','report_id':63})
    assert msg['tool_call_id']=='call_001' and msg['type']=='ToolMessage'

def test_02_intermediate_02_tool_message():
    t=_module(ROOT/'02_intermediate'/'02_tool_message'/'tasks.py')
    call=t.make_request('lookup_status',{'report_id':63},'call_001')
    msg=t.match_result(call,{'status':'ok','report_id':63})
    assert msg['tool_call_id']=='call_001' and msg['type']=='ToolMessage'

def test_03_advanced_02_tool_message():
    t=_module(ROOT/'03_advanced'/'02_tool_message'/'tasks.py')
    call=t.make_request('lookup_status',{'report_id':63},'call_001')
    msg=t.match_result(call,{'status':'ok','report_id':63})
    assert msg['tool_call_id']=='call_001' and msg['type']=='ToolMessage'

def test_01_basic_03_tool_schema():
    t=_module(ROOT/'01_basic'/'03_tool_schema'/'tasks.py')
    assert t.validate_id(63)==63
    with pytest.raises(ValueError):t.validate_id(-1)

def test_02_intermediate_03_tool_schema():
    t=_module(ROOT/'02_intermediate'/'03_tool_schema'/'tasks.py')
    assert t.validate_id(63)==63
    with pytest.raises(ValueError):t.validate_id(-1)

def test_03_advanced_03_tool_schema():
    t=_module(ROOT/'03_advanced'/'03_tool_schema'/'tasks.py')
    assert t.validate_id(63)==63
    with pytest.raises(ValueError):t.validate_id(-1)

def test_01_basic_04_rag_tool():
    t=_module(ROOT/'01_basic'/'04_rag_tool'/'tasks.py')
    assert t.prepare_search('보고서 찾아줘','ALL')['category']=='ALL'
    with pytest.raises(ValueError):t.prepare_search(' ', 'ALL')

def test_02_intermediate_04_rag_tool():
    t=_module(ROOT/'02_intermediate'/'04_rag_tool'/'tasks.py')
    assert t.prepare_search('보고서 찾아줘','ALL')['category']=='ALL'
    with pytest.raises(ValueError):t.prepare_search(' ', 'ALL')

def test_03_advanced_04_rag_tool():
    t=_module(ROOT/'03_advanced'/'04_rag_tool'/'tasks.py')
    assert t.prepare_search('보고서 찾아줘','ALL')['category']=='ALL'
    with pytest.raises(ValueError):t.prepare_search(' ', 'ALL')

def test_01_basic_05_db_status_tool():
    t=_module(ROOT/'01_basic'/'05_db_status_tool'/'tasks.py')
    assert t.parse_report_id('63')==63
    a=t.status_view({'status':'ok','report_id':63,'revision':1,'vector_fresh':False,'index_status':'pending'})
    assert a['index_status']=='pending'

def test_02_intermediate_05_db_status_tool():
    t=_module(ROOT/'02_intermediate'/'05_db_status_tool'/'tasks.py')
    assert t.parse_report_id('63')==63
    a=t.status_view({'status':'ok','report_id':63,'revision':1,'vector_fresh':False,'index_status':'pending'})
    assert a['index_status']=='pending'

def test_03_advanced_05_db_status_tool():
    t=_module(ROOT/'03_advanced'/'05_db_status_tool'/'tasks.py')
    assert t.parse_report_id('63')==63
    a=t.status_view({'status':'ok','report_id':63,'revision':1,'vector_fresh':False,'index_status':'pending'})
    assert a['index_status']=='pending'

def test_01_basic_06_create_agent():
    t=_module(ROOT/'01_basic'/'06_create_agent'/'tasks.py')
    assert '63' in t.question_for_agent(63)
    with pytest.raises(ValueError):t.question_for_agent(0)

def test_02_intermediate_06_create_agent():
    t=_module(ROOT/'02_intermediate'/'06_create_agent'/'tasks.py')
    assert '63' in t.question_for_agent(63)
    with pytest.raises(ValueError):t.question_for_agent(0)

def test_03_advanced_06_create_agent():
    t=_module(ROOT/'03_advanced'/'06_create_agent'/'tasks.py')
    assert '63' in t.question_for_agent(63)
    with pytest.raises(ValueError):t.question_for_agent(0)

def test_01_basic_07_reliability():
    t=_module(ROOT/'01_basic'/'07_reliability'/'tasks.py')
    assert t.validate_report_id('63')==63
    with pytest.raises(ValueError):t.validate_report_id(-1)

def test_02_intermediate_07_reliability():
    t=_module(ROOT/'02_intermediate'/'07_reliability'/'tasks.py')
    assert t.validate_report_id('63')==63
    with pytest.raises(ValueError):t.validate_report_id(-1)

def test_03_advanced_07_reliability():
    t=_module(ROOT/'03_advanced'/'07_reliability'/'tasks.py')
    assert t.validate_report_id('63')==63
    with pytest.raises(ValueError):t.validate_report_id(-1)

def test_01_basic_08_end_to_end():
    t=_module(ROOT/'01_basic'/'08_end_to_end'/'tasks.py')
    assert '보고서' in t.default_question()
    r={'status':'completed','answer':'완료','tool_trace':[],'citation_ids_valid':True,'verified_evidence_ids':[]}
    assert t.api_response(r)['status']=='completed'

def test_02_intermediate_08_end_to_end():
    t=_module(ROOT/'02_intermediate'/'08_end_to_end'/'tasks.py')
    assert '보고서' in t.default_question()
    r={'status':'completed','answer':'완료','tool_trace':[],'citation_ids_valid':True,'verified_evidence_ids':[]}
    assert t.api_response(r)['status']=='completed'

def test_03_advanced_08_end_to_end():
    t=_module(ROOT/'03_advanced'/'08_end_to_end'/'tasks.py')
    assert '보고서' in t.default_question()
    r={'status':'completed','answer':'완료','tool_trace':[],'citation_ids_valid':True,'verified_evidence_ids':[]}
    assert t.api_response(r)['status']=='completed'
