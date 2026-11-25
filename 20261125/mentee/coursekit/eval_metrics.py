"""Retrieval 지표: No-answer는 평가불가(None), 실패 0과 구분.

Precision@k의 분모는 지정된 k로 고정(반환이 k 미만이어도 k 사용).
MRR은 질문별 Reciprocal Rank의 평균; nDCG는 등급 있는 Gold에 한해 계산.
"""
from math import log2, isfinite


def validate_k(k):
    if type(k) is not int or k<1:raise ValueError('k는 1 이상의 정수')


def validate_ranked(ranked):
    if not isinstance(ranked,list) or any(not isinstance(i,str) or not i for i in ranked):
        raise ValueError('ranked_ids는 비어있지 않은 문자열의 list')
    if len(ranked)!=len(set(ranked)):
        raise ValueError('검색 결과의 중복 ID를 제거한 뒤 평가하세요.')


def recall_at_k(ranked, gold, k):
    validate_k(k);validate_ranked(ranked)
    if not gold:return None  # Gold 0개: 검색에 성공했다는 0점/1점과 다름
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


def precision_at_k(ranked, gold, k):
    validate_k(k);validate_ranked(ranked)
    if not gold:return None  # 정보 부족 질문은 no-answer 별도 평가
    return len(set(ranked[:k]) & set(gold)) / k


def hit_at_k(ranked, gold, k):
    validate_k(k);validate_ranked(ranked)
    if not gold:return None
    return int(bool(set(ranked[:k]) & set(gold)))


def rr_at_k(ranked, gold, k):
    validate_k(k);validate_ranked(ranked)
    if not gold:return None
    for position,cid in enumerate(ranked[:k],1):
        if cid in gold:return 1.0/position
    return 0.0


def mean_valid(values):
    valid=[float(v) for v in values if v is not None]
    return sum(valid)/len(valid) if valid else None


def dcg_at_k(ranked, relevance, k):
    validate_k(k);validate_ranked(ranked)
    if any(cid not in relevance for cid in ranked[:k]):
        raise ValueError('미판정 문서의 등급을 0으로 처리할 수 없습니다. 먼저 관련성 레이블을 검수하세요.')
    return sum((2**relevance[cid]-1)/log2(position+1)
               for position,cid in enumerate(ranked[:k],1))


def ndcg_at_k(ranked, relevance, k):
    validate_k(k);validate_ranked(ranked)
    if not relevance:return None
    if any(type(v) is not int or v<0 for v in relevance.values()):
        raise ValueError('등급은 0 이상의 정수')
    ideal=sorted(relevance,key=lambda cid:(-relevance[cid],cid))
    ideal_score=dcg_at_k(ideal,relevance,k)
    return dcg_at_k(ranked,relevance,k)/ideal_score if ideal_score>0 else None


def evaluate_retrieval(ranked, gold, grades, k=3):
    return {'recall_at_k':recall_at_k(ranked,gold,k),
            'precision_at_k':precision_at_k(ranked,gold,k),
            'hit_at_k':hit_at_k(ranked,gold,k),
            'rr_at_k':rr_at_k(ranked,gold,k),
            'dcg_at_k':dcg_at_k(ranked,grades,k) if gold else None,
            'ndcg_at_k':ndcg_at_k(ranked,grades,k),
            'k':k,'gold_count':len(gold),'retrieved_count':min(k,len(ranked))}


def summarize_retrieval(items):
    """평가 가능한 질문만 평균; No-answer와 실패를 별도 카운트."""
    keys=('recall_at_k','precision_at_k','hit_at_k','rr_at_k','ndcg_at_k')
    summary={('mean_'+key):mean_valid(row.get(key) for row in items) for key in keys}
    summary['mrr_at_k']=summary['mean_rr_at_k']
    summary['evaluated_questions']=sum(row.get('recall_at_k') is not None for row in items)
    summary['no_answer_excluded']=sum(row.get('recall_at_k') is None for row in items)
    return summary
