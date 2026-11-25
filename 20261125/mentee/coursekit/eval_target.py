"""LangChain Target Runner. Gold Label이 Runner로 유입되지 않도록 분리.

Fixture 모드에서 읽는 랭킹/Latency는 시험용 고정 값이지 GPU·DB 실측이 아닙니다.
Live 모드에서는 Day27 RAG의 E5(cuda), pgvector, CrossEncoder(cuda), ChatOpenAI를 호출합니다.
"""
from functools import lru_cache
from time import perf_counter
from coursekit.eval_dataset import assert_no_gold_leakage,read_json,target_inputs

ALLOWED_VARIANTS = {'baseline':(False,False),'rewrite_only':(True,False),
                    'rerank_only':(False,True),'both':(True,True)}


def target_fixture(request,variant='baseline'):
    assert_no_gold_leakage(request)
    if variant not in ALLOWED_VARIANTS:raise ValueError('unknown experiment variant')
    qid=request['question_id']
    matches=[r for r in read_json('fixture_runs.json') if r['question_id']==qid and r['variant']==variant]
    if len(matches)!=1:raise ValueError('fixture record missing/duplicate: '+qid)
    record=dict(matches[0])
    # 평가 레코드의 Gold/Reference는 포함하지 않으며, 코드도 이 파일을 읽지 않습니다.
    return record


@lru_cache(maxsize=1)
def _live_chain():
    from coursekit.runtime import create_demo_chain
    return create_demo_chain(offline=False)


def target_live(request,variant='baseline'):
    assert_no_gold_leakage(request)
    if variant not in ALLOWED_VARIANTS:raise ValueError('unknown experiment variant')
    rewrite,rerank=ALLOWED_VARIANTS[variant]
    qid=request['question_id']
    from coursekit.config import require_cuda
    require_cuda()  # CPU로 조용히 전환하지 않는 과정 공통 기준
    start=perf_counter()
    chain=_live_chain()
    result=chain.invoke({'query':request['question'],'history':request.get('history',[]),
                        'category':request['category'],'candidate_k':5,'context_k':2,
                        'use_rewrite':rewrite,'use_rerank':rerank,'use_multiquery':False})
    trace=result.get('trace')
    if not isinstance(trace,dict) or not all(key in trace for key in ('candidate_ids','ranked_ids','included_ids','source_mode')):
        raise RuntimeError('Day27 RAG에서 평가 Trace 필수 필드 누락. 빈 문서로 대체하지 않습니다.')
    if trace['source_mode']!='live':raise RuntimeError('실제 실행이라고 보고했으나 fixture Trace입니다.')
    response=result.get('result')
    if not isinstance(response,dict):raise RuntimeError('LLM 응답 누락')
    return {'question_id':qid,'variant':variant,
            'candidate_ids':trace['candidate_ids'],'ranked_ids':trace['ranked_ids'],
            'included_ids':trace['included_ids'],
            'answer':response['answer'],'status':result['status'],
            'sources':result.get('sources',[]),
            'timing_ms':trace.get('timing_ms',{}),'mode':'live',
            'total_elapsed_ms':round((perf_counter()-start)*1000,3),
            'data_version':'d28-eval-v1'}


def as_runnable(target_function):
    """실제 LangChain RunnableLambda; 설치 안 되어 있으면 정상 에러로 안내."""
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(target_function)
