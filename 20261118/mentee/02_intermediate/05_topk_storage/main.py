"""Day23 실습 05 · 원문·벡터 저장과 Top-k (응용, mentee).

실습 목적은 문서 원문과 E5 벡터를 같은 테이블에 저장하고, 질문 벡터에
가장 가까운 문서를 찾는 것입니다. Top-k는 가까운 순서대로 최대 k건을
가져오는 검색이며, 중급 실습에서는 development 분류에서 최대 3건을 찾습니다.

실행 흐름
1. 로컬 지식 9건을 읽고 문서 입력과 검색에 사용할 SQL을 준비합니다.
2. 검색 개수와 문서 분류를 검사합니다.
3. fixture에서는 모의 검색을 실행합니다. live에서는 문서와 질문을 E5로
   벡터화하고, 문서 원문과 벡터를 PostgreSQL에 저장한 뒤 pgvector로 검색합니다.
4. 결과를 터미널에 표시하고 outputs/exercise_05.json에 저장합니다.

기본 실행은 python main.py --mode fixture입니다.
실제 저장과 검색은 DB 설정을 준비한 뒤 python main.py --mode live로 실행합니다.
"""
import argparse
import sys
from pathlib import Path
# mentee/common의 공용 day23kit 패키지를 불러올 수 있게 경로를 추가합니다.
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.data import load_knowledge, E5_MODEL_ID, TABLE_NAME
from day23kit.fixtures import fixture_search
from day23kit.embedding import get_encoder, encode_passages, encode_question, validate_vectors
from day23kit.db import connect_db, ensure_table, upsert_rows, exact_topk
from day23kit.io import save_result

from tasks import schema_sql, prepare_document_rows, topk_sql, filtered_sql, check_search_options



def run(mode,level,args):
    """문서 입력과 SQL을 준비하고, 검색 결과를 JSON에 담을 딕셔너리로 반환합니다."""
    rows=load_knowledge()
    sql_schema=schema_sql()
    inputs=prepare_document_rows(rows)
    sql_query=topk_sql()
    question='서버는 어디에서 실행하나요?'
    # SQL 문자열과 첫 번째 문서 입력을 결과에 남겨 실행 전에 준비한 내용을 확인합니다.
    result=dict(exercise=5,level=level,source_mode='simulated' if mode=='fixture' else 'pgvector_exact',
                table=TABLE_NAME,model=E5_MODEL_ID,prepared_count=len(inputs),
                first_passage=inputs[0]['passage_input'],
                sql_schema=sql_schema,sql_topk=sql_query)
    category=None if level==1 else 'development'
    if level>=2:
        # 중급에서는 development 문서만 검색합니다. k=3은 결과 개수의 상한입니다.
        result['sql_filtered']=filtered_sql()
        result['options']=check_search_options(3,category)
    if mode=='fixture':
        # 로컬 모의 검색으로 후보를 구합니다. E5 모델이나 DB 연결 없이 실행됩니다.
        hits=fixture_search(question,3,category=category)
    else:
        # 같은 E5 모델로 문서 벡터와 질문 벡터를 만들어 서로 비교할 수 있게 합니다.
        # encode_passages는 rows에서 passage 입력 문자열을 만들어 인코딩합니다.
        encoder=get_encoder('live')
        vectors=encode_passages(encoder,rows)
        if level>=3:check_id_vector_integrity(rows,vectors)
        else:validate_vectors(vectors,len(rows))
        qvec=encode_question(encoder,question)
        with connect_db(Path(__file__).with_name('.env')) as conn:
            # 검색 테이블을 준비하고, 문서와 벡터를 같은 순서로 짝지어 저장합니다.
            # 같은 ID가 이미 저장되어 있으면 해당 행을 갱신합니다.
            ensure_table(conn)
            upsert_rows(conn,rows,vectors,E5_MODEL_ID)
            # 코사인 거리가 작은 문서부터 최대 3건을 가져옵니다.
            hits=exact_topk(conn,qvec,k=3,category=category)
    # 최종 JSON의 hits에는 문서 ID, 본문, 소수점 여섯 자리로 반올림한 거리만 담습니다.
    result['hits']=[{'id':h['id'],'content':h['content'],'distance':round(h['distance'],6)} for h in hits]
    if level>=3:
        # 이전 단계의 벡터를 재사용합니다. live에서는 같은 모델을 다시 다운로드/로드하지 않습니다.
        if mode=='fixture':
            encoder=get_encoder('fixture')
            vectors=encode_passages(encoder,rows)
        result['id_vector_integrity']=check_id_vector_integrity(rows,vectors)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 05 · 원문·벡터 저장과 Top-k')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    # 중급 난이도 2로 실행하고 main.py 옆 outputs 폴더에 결과를 저장합니다.
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_05.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
