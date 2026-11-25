"""Generation 평가에서 Claim·Question·Reference를 구분합니다.

실제 LLM Judge는 별도 파일에서만 opt-in. 여기서는 **사람/교육용 레이블**로 계산합니다.
"""
from math import isfinite


def faithfulness_from_labels(claims):
    if not claims:return None  # 주장 자체가 없으면 N/A; 참인 비율을 1로 만들지 않음
    for claim in claims:
        if not isinstance(claim.get('supported'),bool):
            raise ValueError('Claim.supported는 사람 검토 bool')
    return sum(int(c['supported']) for c in claims)/len(claims)


def response_review(question,context,answer,claims,*,relevance,correctness,reviewer):
    """Relevance/Correctness는 제공된 사람 평가 레이블; 자동 추론 결과가 아님."""
    if not question.strip() or not isinstance(context,str) or not isinstance(answer,str):
        raise ValueError('입력 자료형 오류')
    if relevance not in ('높음','보통','낮음','평가 보류'):
        raise ValueError('relevance 등급 오류')
    if correctness not in ('정답','부분정답','오답','판정 불가'):
        raise ValueError('correctness 등급 오류')
    return {'faithfulness':faithfulness_from_labels(claims),
            'relevance':relevance,'correctness':correctness,
            'claim_count':len(claims),'reviewer':reviewer,
            'review_mode':'human_or_fixture_label'}
