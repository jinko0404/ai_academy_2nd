"""Day23 실습 06 · Exact·HNSW·EXPLAIN (응용, mentee).

실습 목표는 벡터 검색의 실행 계획을 읽고, 정확 검색(Exact)과
근사 최근접 검색(ANN)의 결과를 비교하는 방법을 익히는 것입니다.
정확 검색은 모든 후보의 거리를 비교합니다. HNSW는 그래프 인덱스로
가까운 후보를 탐색하므로 정확 검색과 다른 결과가 나올 수 있습니다.

실행 흐름
1. 인덱스 생성 SQL과 실행 계획을 확인할 검색 SQL을 준비합니다.
2. fixture는 예시 실행 계획을 읽습니다. live는 질문을 E5 벡터로 바꾸고
   실습 05에서 저장한 문서를 대상으로 PostgreSQL 실행 계획을 확인합니다.
3. 중급 단계에서는 코사인 거리와 인덱스 설정이 맞는지 확인하고,
   예시 결과 목록이 얼마나 겹치는지 계산합니다.
4. 결과를 터미널에 표시하고 outputs/exercise_06.json에 저장합니다.

기본 실행은 python main.py --mode fixture입니다.
live 실행은 실습 05의 live 저장과 같은 DB 접속 설정이 필요합니다.
인덱스를 생성하는 명령까지 실행하려면 live에서 --create-index를 지정합니다.
"""
import argparse
import sys
from pathlib import Path
# mentee/common을 모듈 검색 경로에 추가해 day23kit을 불러옵니다.
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.data import E5_MODEL_ID
from day23kit.embedding import get_encoder, encode_question
from day23kit.db import connect_db, exact_topk, check_data_ready
from day23kit.io import save_result

from tasks import create_index_sql, explain_sql, read_scan_type, check_operator_compatibility, compare_topk



def run(mode,level,args):
    """실행 계획의 스캔 방식과 중급 점검 결과를 딕셔너리로 반환합니다."""
    # 이 함수 호출들은 SQL 문자열을 준비합니다. 실행은 아래 live 분기에서 합니다.
    sql_index=create_index_sql()
    sql_plan=explain_sql()
    result=dict(exercise=6,level=level,source_mode='simulated' if mode=='fixture' else 'pgvector_live',
                index_sql=sql_index,explain_sql=sql_plan)
    if mode=='fixture':
        # 직접 준비한 계획 예시를 사용하므로 결과에 모의 계획임을 표시합니다.
        sample=['Seq Scan on day23_search_docs (cost=...)','Sort Key: (embedding <=> ...)']
        result['scan_type']=read_scan_type(sample)
        result['plan_is_simulated']=True
    else:
        # 실습 05에서 문서에 사용한 것과 같은 E5 모델로 질문 벡터를 만듭니다.
        encoder=get_encoder('live')
        qvec=encode_question(encoder,'서버는 어디에서 실행하나요?')
        with connect_db(Path(__file__).with_name('.env')) as conn:
            # 연결된 DB에서 05가 만든 검색 테이블과 문서 벡터가 준비됐는지 확인합니다.
            check_data_ready(conn)
            if getattr(args,'create_index',False):
                # 인덱스 생성 SQL을 실행하고, PostgreSQL이 실행 계획에 쓸 통계를 갱신합니다.
                conn.execute(sql_index)
                conn.execute('ANALYZE day23_search_docs')
            # SQL의 %s 두 곳에 모델 ID와 질문 벡터를 순서대로 전달합니다.
            # EXPLAIN 결과는 행마다 텍스트가 첫 번째 열(r[0])에 있습니다.
            plan_lines=[str(r[0]) for r in conn.execute(sql_plan,(E5_MODEL_ID,qvec)).fetchall()]
            result['scan_type']=read_scan_type(plan_lines)
            result['explain_lines']=plan_lines
            if level>=3:
                # 고급 live 단계의 비교입니다. 두 번 모두 같은 질문 벡터로
                # development 문서 중 코사인 거리가 가까운 문서를 최대 3건 찾습니다.
                # SET LOCAL로 바꾼 설정은 현재 트랜잭션이 끝나면 원래 값으로 돌아갑니다.

                # 1. 정확 검색 결과를 기준으로 만듭니다.
                # 인덱스 스캔과 비트맵 스캔을 피하도록 설정해 순차 스캔을 유도합니다.
                # 순차 스캔은 테이블을 읽으며 조건에 맞는 모든 문서의 거리를 비교합니다.
                conn.execute('SET LOCAL enable_indexscan=off')
                conn.execute('SET LOCAL enable_bitmapscan=off')
                # 검색 결과의 각 행 h에서 문서 ID만 꺼내 순서대로 모읍니다.
                exact_ids=[h['id'] for h in exact_topk(conn,qvec,k=3,category='development')]

                # 2. 인덱스를 쓰기 쉬운 조건에서 다시 검색합니다.
                # 인덱스와 비트맵 스캔을 허용하고, 순차 스캔은 되도록 피하게 합니다.
                # 이 설정은 실행 계획 선택에 영향을 주며 실제 스캔 방식은 EXPLAIN에서 봅니다.
                conn.execute('SET LOCAL enable_indexscan=on')
                conn.execute('SET LOCAL enable_bitmapscan=on')
                conn.execute('SET LOCAL enable_seqscan=off')
                # fetchall()로 EXPLAIN 결과를 가져오고, 각 행의 첫 번째 열에서 계획 텍스트를 꺼냅니다.
                # sql_plan에는 category 조건이 없어서 아래 development 검색과는 별개의 실행 계획입니다.
                indexed_plan=[str(r[0]) for r in conn.execute(sql_plan,(E5_MODEL_ID,qvec)).fetchall()]
                # 같은 검색 함수를 호출해도 PostgreSQL은 현재 설정을 바탕으로 계획을 선택합니다.
                indexed_ids=[h['id'] for h in exact_topk(conn,qvec,k=3,category='development')]
                # 비교가 끝났으므로 이후 쿼리에서 순차 스캔을 다시 허용합니다.
                conn.execute('SET LOCAL enable_seqscan=on')

                # 3. 두 ID 목록과 실행 계획에서 읽은 스캔 방식을 저장합니다.
                # 여기서 확인하는 것은 계획에 Index Scan이 있는지입니다.
                # HNSW를 실제로 사용했는지 확인하려면 같은 필터의 EXPLAIN에서 인덱스 이름을 봐야 합니다.
                confirmed_index=(read_scan_type(indexed_plan)=='index')
                result['comparison']={
                    'category':'development','k':3,'exact_ids':exact_ids,
                    'forced_plan_ids':indexed_ids,
                    'forced_plan_scan':read_scan_type(indexed_plan),
                    'hnsw_index_observed':confirmed_index,
                    # 예를 들어 기준 ID 3건 중 2건이 겹치면 2/3입니다. 결과 순서는 비교하지 않습니다.
                    # 위 계획이 index로 분류된 경우에만 비율을 기록하고, 그 외에는 JSON에 null로 저장합니다.
                    'exact_vs_ann_overlap':compare_topk(exact_ids,indexed_ids) if confirmed_index else None,
                    'note':'Index Scan이 관측됐을 때만 ANN 비교라 기록합니다. 9건에서 속도 향상 주장 금지.',
                }
    if level>=2:
        # 두 예시 목록에서 102와 107이 겹치므로 일치율은 2/3입니다.
        result['operator_compatible']=check_operator_compatibility(sql_plan,'vector_cosine_ops')
        result['example_overlap']=compare_topk([102,106,107],[102,107,101])
    if level>=3:
        result['ann_filtered_example']=filtered_ann_warning(k=3,returned=1,exact_count=4)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 06 · Exact·HNSW·EXPLAIN')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')

    parser.add_argument('--create-index',action='store_true',help='live에서만 HNSW 인덱스를 실제 생성')
    args=parser.parse_args()
    # 이 파일은 중급 난이도 2를 실행합니다. level >= 3 분기는 고급 단계용입니다.
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_06.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
