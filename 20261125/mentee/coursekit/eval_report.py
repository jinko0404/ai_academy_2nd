"""Run / Gold / Metrics를 결합한 질문별 보고서와 최소 Release Gate."""
from pathlib import Path
import json
from coursekit.eval_dataset import target_inputs,read_json
from coursekit.eval_metrics import evaluate_retrieval,summarize_retrieval


def source_integrity(run):
    # Source IDs and locators are checked against the original indexed corpus.
    corpus={d['chunk_id']:d for d in read_json('corpus.json')}
    included=set(run.get('included_ids',[]))
    for s in run.get('sources',[]):
        if s.get('chunk_id') not in included:return False
        doc=corpus.get(s.get('chunk_id'))
        if doc is None or not doc['ready'] or not doc['is_current'] or doc['access_level']!='public':return False
        if s.get('source_locator')!=doc['source_locator'] or type(s.get('citation_id')) is not int:return False
        # citation_id can differ from fixture ID in a live DB after migration.
        if run.get('mode')=='offline_fixture' and s['citation_id']!=doc['fixture_citation_id']:return False
    if run.get('status')=='answered' and not run.get('sources'):return False
    return True


def evaluate_record(example,run,gold,grades,k=3):
    if run['question_id']!=example['question_id']:raise ValueError('question_id 일치 필요')
    if run['mode'] not in ('offline_fixture','live'):raise ValueError('실행 모드를 확인하세요')
    if not all(field in run for field in ('ranked_ids','candidate_ids','included_ids','answer','status','sources')):
        raise ValueError('Target Trace 필수 항목 누락 - 평가 불가능')
    metrics=evaluate_retrieval(run['ranked_ids'],gold,grades,k)
    return {'question_id':example['question_id'],'variant':run['variant'],
            'mode':run['mode'],'data_version':run['data_version'],'metrics':metrics,
            'ranked_ids':run['ranked_ids'],'included_ids':run['included_ids'],
            'status':run['status'],'sources_valid':source_integrity(run),
            'no_answer_result':('correct_abstention' if run['status']=='insufficient_context' else 'unsupported_answer') if not example['answerable'] else 'not_no_answer',
            'latency_ms':run.get('total_elapsed_ms'),
            'timing_ms':run.get('timing_ms',{}),'answer_preview':run['answer'][:120]}


def report_summary(rows):
    if not rows:raise ValueError('실행 결과 없음')
    combos={(r['mode'],r['data_version'],r['variant']) for r in rows}
    if len(combos)!=1:raise ValueError('한 보고서에 서로 다른 모드/버전/변형을 혼합할 수 없습니다.')
    agg=summarize_retrieval([r['metrics'] for r in rows])
    agg['source_integrity_failures']=sum(not r['sources_valid'] for r in rows)
    agg['insufficient_context']=sum(r['status']=='insufficient_context' for r in rows)
    agg['status_failures']=sum(r['status'] not in ('answered','insufficient_context') for r in rows)
    agg['no_answer_correct_abstentions']=sum(r.get('no_answer_result')=='correct_abstention' for r in rows)
    agg['no_answer_unsupported_answers']=sum(r.get('no_answer_result')=='unsupported_answer' for r in rows)
    agg.update(zip(('mode','data_version','variant'),next(iter(combos))))
    return agg


def write_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')
    return p


def release_gate(summary,*,min_recall=.70,min_mrr=.60):
    """교육용 1차 Quality Gate. 실제 서비스 배포 승인 전체를 뜻하지 않습니다."""
    checks={
     'retrieval_recall':summary.get('mean_recall_at_k') is not None and summary['mean_recall_at_k']>=min_recall,
     'retrieval_mrr':summary.get('mrr_at_k') is not None and summary['mrr_at_k']>=min_mrr,
     'source_integrity':summary.get('source_integrity_failures',1)==0,
     'execution_status':summary.get('status_failures',1)==0,
     'human_review':summary.get('human_review_passed',False),
     'access_control_tests':summary.get('access_control_passed',False)
    }
    return {'ready_for_agent_review':all(checks.values()),'checks':checks,
            'scope':'교육용 1차 Gate. Agent의 Tool 권한·안전성·운영 배포 승인 별도 필요'}
