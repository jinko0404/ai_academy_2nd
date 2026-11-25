"""고정 평가셋의 수학·Gold 누출·No-answer·출처 보존 핵심 계약 검사."""
from pathlib import Path
import sys,math
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coursekit.eval_dataset import read_questions,read_gold,read_json,target_inputs,assert_no_gold_leakage,validate_dataset
from coursekit.eval_metrics import recall_at_k,precision_at_k,hit_at_k,rr_at_k,ndcg_at_k
from coursekit.eval_target import target_fixture
from coursekit.eval_report import evaluate_record,release_gate


def test_dataset_matches_corpus_and_gold_separate():
    q=read_questions();g,grade,ver=read_gold()
    assert validate_dataset(q,g,grade,read_json('corpus.json'))['no_answer']==1
    assert 'gold_chunk_ids' not in target_inputs(q[0])
    with pytest.raises(ValueError):assert_no_gold_leakage({'gold_chunk_ids':['FAC-102-C1']})


def test_retrieval_math_with_noanswer():
    ranked=['FAC-101-C1','FAC-107-C1','FAC-102-C1']
    assert recall_at_k(ranked,['FAC-102-C1'],3)==1.0
    assert precision_at_k(ranked,['FAC-102-C1'],3)==pytest.approx(1/3)
    assert hit_at_k(ranked,['FAC-102-C1'],3)==1
    assert rr_at_k(ranked,['FAC-102-C1'],3)==pytest.approx(1/3)
    assert recall_at_k([],[],3) is None
    assert rr_at_k([],[],3) is None


def test_unjudged_is_not_grade_zero():
    grades={'FAC-102-C1':3,'FAC-101-C1':0}
    with pytest.raises(ValueError,match='미판정'):
        ndcg_at_k(['FAC-999-C1'],grades,1)


def test_fake_source_rejected():
    ex=read_questions()[0];g,grades,v=read_gold()
    run=target_fixture(target_inputs(ex),'both')
    clean=evaluate_record(ex,run,g[ex['question_id']],grades[ex['question_id']])
    assert clean['sources_valid'] is True
    run['sources'][0]['source_locator']='forged-fake-source.md'
    forged=evaluate_record(ex,run,g[ex['question_id']],grades[ex['question_id']])
    assert forged['sources_valid'] is False


def test_human_access_required_for_agent_gate():
    good={'mean_recall_at_k':1.0,'mrr_at_k':1.0,'source_integrity_failures':0,
          'status_failures':0,'human_review_passed':False,'access_control_passed':False}
    assert release_gate(good)['ready_for_agent_review'] is False
