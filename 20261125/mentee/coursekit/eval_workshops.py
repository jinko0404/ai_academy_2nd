"""Day28 practice dispatcher: all completed levels share the same input/output contracts.

The fixed fixtures are for teaching calculation, not evidence of real model quality.
Exercise 06-08 --live requires CUDA E5, pgvector, CrossEncoder, ChatOpenAI.
"""
from pathlib import Path
import json
from coursekit.eval_dataset import read_json,read_questions,read_gold,target_inputs
from coursekit.eval_target import target_fixture,target_live,as_runnable
from coursekit.eval_report import write_json,report_summary
from coursekit.eval_metrics import mean_valid


def _base():
    q=read_questions();gold,grades,version=read_gold()
    return q,gold,grades,version


def execute(number, hooks, *, live=False, level='01_basic'):
    if live and number<6:
        raise ValueError('실습 01~05는 제공 레이블/Fixture 기반 계산입니다. 06~08에서 --live로 실제 RAG 평가를 수행하세요.')
    questions,gold,grades,version=_base()
    qby={q['question_id']:q for q in questions}
    cases=read_json('failure_cases.json')
    result={'exercise':number,'level':level,'dataset_version':version,
            'mode':'live' if live else 'offline_fixture'}

    if number==1:
        rows=[]
        for case in cases:
            stage=hooks['classify_case'](case)
            evidence=hooks['describe_evidence'](case,stage)
            row={'case_id':case['case_id'],'stage':stage,'evidence':evidence}
            if 'recommend_fix' in hooks:row['recommendation']=hooks['recommend_fix'](stage)
            if 'audit_decision' in hooks:row['audit']=hooks['audit_decision'](case,stage,evidence)
            rows.append(row)
        result['cases']=rows

    elif number==2:
        corpus={c['chunk_id']:c for c in read_json('corpus.json')}
        rows=[]
        for qid in ('Q01','Q05','Q06','Q07'):
            ids=hooks['select_gold'](qid,gold)
            ok=hooks['validate_gold_ids'](ids,corpus)
            row={'question_id':qid,'gold_chunk_ids':ids,'valid_ids':ok}
            if 'label_no_answer' in hooks:row['no_answer']=hooks['label_no_answer'](qid,gold)
            if 'audit_gold' in hooks:row['audit']=hooks['audit_gold'](qid,ids,version)
            rows.append(row)
        result['annotations']=rows
        result['target_example']=target_inputs(qby['Q01'])  # no Gold leakage

    elif number==3:
        rows=[]
        values=[]
        for qid in ('Q01','Q07','Q06'):
            run=target_fixture(target_inputs(qby[qid]),'baseline')
            ids=run['ranked_ids']; g=gold[qid]; k=3
            recall=hooks['calculate_recall'](ids,g,k)
            hits=hooks['count_gold_hits'](ids,g,k)
            row={'question_id':qid,'ranked_ids':ids,'gold_count':len(g),
                 'hits_at_3':hits,'recall_at_3':recall}
            if 'precision_and_hit' in hooks:
                row.update(hooks['precision_and_hit'](ids,g,k))
            rows.append(row);values.append(recall)
        result['metrics']=rows
        if 'summarize_valid_recall' in hooks:result['aggregation']=hooks['summarize_valid_recall'](values)

    elif number==4:
        rows=[];rr_vals=[]
        for qid in ('Q01','Q07','Q06'):
            run=target_fixture(target_inputs(qby[qid]),'baseline'); ids=run['ranked_ids'];g=gold[qid]
            rank=hooks['first_relevant_rank'](ids,g,3)
            # No-answer RR is excluded from MRR even though a missing gold hit has 0 when answerable.
            rr=hooks['calculate_rr'](rank) if g else None
            row={'question_id':qid,'first_rank':rank,'rr_at_3':rr}
            if 'calculate_dcg' in hooks:row['dcg_at_3']=hooks['calculate_dcg'](ids,grades[qid],3)
            if 'calculate_ndcg' in hooks:row['ndcg_at_3']=hooks['calculate_ndcg'](ids,grades[qid],3)
            rows.append(row);rr_vals.append(rr)
        result['ranking']=rows;result['mrr_at_3']=mean_valid(rr_vals)

    elif number==5:
        cases=read_json('generation_claims.json'); rows=[]
        for case in cases:
            f=hooks['claim_faithfulness'](case['claims'])
            notes=hooks['describe_claim_result'](case['claims'])
            row={'case_id':case['case_id'],'faithfulness':f,'reason':notes,'label_source':'human_fixture'}
            if 'label_answer_quality' in hooks:
                row.update(hooks['label_answer_quality'](case['relevance'],case['correctness']))
            if 'prepare_human_review' in hooks:
                row['human_review']=hooks['prepare_human_review'](case['case_id'],f,'교육용 검수자')
            rows.append(row)
        result['generation_reviews']=rows

    elif number==6:
        target=as_runnable(lambda req:target_live(req,'both') if live else target_fixture(req,'both'))
        rows=[]
        for example in questions[:3]:
            req=hooks['safe_target_input'](example)
            run=hooks['invoke_target'](target,req)
            row={'question_id':example['question_id'],'source_mode':run['mode'],
                 'ranked_ids':run['ranked_ids'],'included_ids':run['included_ids']}
            if 'evaluate_target_run' in hooks:
                row['evaluation']=hooks['evaluate_target_run'](example,run,gold[example['question_id']],grades[example['question_id']])
            if 'audit_runner' in hooks:row['audit']=hooks['audit_runner'](req,run,version)
            rows.append(row)
        result['runner_results']=rows;result['langchain_runnable']=True

    elif number==7:
        qids=('Q01','Q07','Q06') if not live else ('Q01',) # live variants cost real tokens
        rows=[]
        for qid in qids:
            req=target_inputs(qby[qid]);g=gold[qid]; gr=grades[qid]
            load=lambda v:target_live(req,v) if live else target_fixture(req,v)
            a=load('baseline'); b=load('rerank_only')
            from coursekit.eval_metrics import evaluate_retrieval
            m1=evaluate_retrieval(a['ranked_ids'],g,gr,3)
            m2=evaluate_retrieval(b['ranked_ids'],g,gr,3)
            row={'question_id':qid,'baseline_recall':m1['recall_at_k'],
                 'rerank_recall':m2['recall_at_k'],
                 'delta_recall':hooks['score_delta'](m1['recall_at_k'],m2['recall_at_k']),
                 'diagnosis':hooks['find_problem_stage'](a,g)}
            if 'compare_four_variants' in hooks:
                variants=['baseline','rerank_only'] if live else ['baseline','rewrite_only','rerank_only','both']
                runs={v:(a if v=='baseline' else b if v=='rerank_only' else load(v)) for v in variants}
                row['matrix']=hooks['compare_four_variants'](runs,g,gr)
            if 'regression_decision' in hooks:
                row['regression']=hooks['regression_decision'](m1['recall_at_k'],m2['recall_at_k'],0)
            rows.append(row)
        result['comparisons']=rows
        result['comparison_kind']='offline_version_comparison_not_online_AB'

    elif number==8:
        rows=[]
        for example in questions[:3] if live else questions:
            qid=example['question_id'];req=target_inputs(example)
            run=target_live(req,'both') if live else target_fixture(req,'both')
            rows.append(hooks['evaluate_one'](example,run,gold[qid],grades[qid]))
        summary=hooks['summarize_all'](rows)
        result['summary']=summary;result['questions']=rows
        if 'write_summary_report' in hooks:
            result['markdown_report']=hooks['write_summary_report'](summary,rows)
        if 'agent_release_gate' in hooks:
            result['agent_gate']=hooks['agent_release_gate'](summary,False,False)
    else:raise ValueError('unknown Day28 task index')
    return result
