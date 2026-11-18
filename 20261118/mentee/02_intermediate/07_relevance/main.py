"""Day23 실습 07 · 검색 결과 정성평가 (응용, mentee).

실습 목표는 검색된 문서의 본문을 읽고 질문에 답할 근거가 있는지 판단하는
방법을 익히는 것입니다. 벡터 거리는 문서가 얼마나 가까운지 나타냅니다.
문서가 실제로 질문에 답하는지는 본문을 읽어 확인합니다.

실행 흐름
1. 평가할 질문 3개를 준비하고 질문마다 가까운 문서를 최대 3건 찾습니다.
2. 검색 상태와 문서별 검토 기록을 만듭니다.
3. 같은 질문에서 분류 필터를 적용하기 전후의 검색 결과를 비교합니다.
4. 결과를 터미널에 표시하고 outputs/exercise_07.json에 저장합니다.

python main.py --mode fixture는 로컬 모의 검색으로 흐름을 확인합니다.
python main.py --mode live는 실제 E5와 PostgreSQL을 사용합니다.
live에서는 실습 05에서 문서 벡터를 저장한 DB에 접속해야 합니다.
"""
import argparse
import sys
from pathlib import Path
# mentee/common을 모듈 검색 경로에 추가해 공용 day23kit을 불러옵니다.
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.fixtures import fixture_search
from day23kit.embedding import get_encoder, encode_question
from day23kit.db import connect_db, exact_topk, check_data_ready
from day23kit.io import save_result

from tasks import make_casebook, judge_training_reference, make_review_row, classify_search_state, review_filter_effect



def run(mode,level,args):
    """질문별 검색 결과를 사람이 검토할 수 있는 기록으로 정리합니다."""
    cases=make_casebook()
    # reviews에는 문서 후보별 기록을, search_state에는 질문별 검색 상태를 모읍니다.
    reviews=[]; search_state=[]
    for case in cases:
        # 검색 자체는 fixture의 수학 예시 또는 실제 E5/pgvector.
        if mode=='fixture':hits=fixture_search(case['question'],k=3)
        else:
            # 첫 질문에서만 E5와 DB 연결을 준비하고, 나머지 질문에서도 그대로 사용합니다.
            if '_live_cache' not in locals():
                encoder=get_encoder('live')
                conn=connect_db(Path(__file__).with_name('.env'))
                check_data_ready(conn)
                _live_cache=True
            qvec=encode_question(encoder,case['question'])
            hits=exact_topk(conn,qvec,3)
        # 검색이 정상적으로 끝났다면 후보가 있는지, 0건인지 기록합니다.
        search_state.append({'case_id':case['case_id'],'state':classify_search_state(hits) if level>=2 else ('candidates' if hits else 'empty')})
        for hit in hits:
            row=make_review_row(case,hit)
            # 나중에 검토할 때 어떤 버전의 문서였는지 추적할 수 있게 남깁니다.
            row['source_version']=hit['source_version']
            reviews.append(row)
    # 사람이 원문 근거를 직접 확인했는지는 아직 판단하지 않았으므로 None으로 둡니다.
    result=dict(exercise=7,level=level,source_mode='simulated' if mode=='fixture' else 'pgvector_exact',
                review_rows=reviews,search_states=search_state,
                human_evidence_ok=None)
    if level>=2:
        # 질문과 k=3은 그대로 두고, development 필터 유무에 따라 반환된 ID를 비교합니다.
        if mode=='live':
            qvec=encode_question(encoder,'서버는 어디에서 실행하나요?')
            baseline=[h['id'] for h in exact_topk(conn,qvec,k=3)]
            filtered=[h['id'] for h in exact_topk(conn,qvec,k=3,category='development')]
        else:
            baseline=[h['id'] for h in fixture_search('서버는 어디에서 실행하나요?',k=3)]
            filtered=[h['id'] for h in fixture_search('서버는 어디에서 실행하나요?',k=3,category='development')]
        result['filter_effect']=review_filter_effect(baseline,filtered)
        # 같은 빈 목록이어도 오류 정보가 주어지면 error로 분류하는지 예시로 확인합니다.
        result['empty_is_not_error']=classify_search_state([])=='empty' and classify_search_state([],error='DB 연결 오류')=='error'
    if level>=3:result['evaluation_integrity']=check_evaluation_integrity(reviews)
    if mode=='live':conn.close()
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 07 · 검색 결과 정성평가')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    # 중급 난이도 2로 실행하고 이 파일 옆 outputs 폴더에 JSON을 저장합니다.
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_07.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
