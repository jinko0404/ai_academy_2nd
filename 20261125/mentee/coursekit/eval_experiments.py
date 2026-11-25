"""Offline A/B(동일 Dataset의 버전 비교)와 실패 위치 분류."""
from coursekit.eval_metrics import evaluate_retrieval,mean_valid


def compare_pair(question_id,gold,grades,baseline,variant,k=3):
    if baseline['question_id']!=question_id or variant['question_id']!=question_id:
        raise ValueError('question_id 불일치')
    if baseline['mode']!=variant['mode'] or baseline['data_version']!=variant['data_version']:
        raise ValueError('실행 모드/데이터 버전이 달라 직접 비교할 수 없음')
    a=evaluate_retrieval(baseline['ranked_ids'],gold,grades,k)
    b=evaluate_retrieval(variant['ranked_ids'],gold,grades,k)
    return {'question_id':question_id,'before':a,'after':b,
            'recall_delta':(b['recall_at_k']-a['recall_at_k']) if b['recall_at_k'] is not None else None,
            'rr_delta':(b['rr_at_k']-a['rr_at_k']) if b['rr_at_k'] is not None else None,
            'mode':baseline['mode']}


def diagnose(run,gold):
    candidates=run.get('candidate_ids',[])
    ranking=run.get('ranked_ids',[])
    included=run.get('included_ids',[])
    if run.get('status') not in ('answered','insufficient_context'):
        return 'system_failure'
    if not gold:return 'no_answer_case'
    if not set(candidates)&set(gold):return 'candidate_miss'
    if not set(ranking[:2])&set(gold):return 'ranking_miss'
    if not set(included)&set(gold):return 'context_miss'
    if run.get('status')!='answered':return 'generation_or_validation'
    return 'inspect_answer_grounding'  # 출처 ID만으로 사실 충실성을 인증할 수 없음
