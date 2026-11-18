"""Day23 실습 03 · E5 문장 Embedding (응용, mentee).

이전 실습에서 벡터끼리 유사도를 비교했다면, 이번에는 실제 검색에 쓸
문서와 질문을 벡터로 바꾸는 과정을 연습합니다. 임베딩은 텍스트를
숫자 목록으로 표현한 것이며, 이 실습에서는 입력 하나당 숫자 384개를 얻습니다.

지식 읽기 → E5 입력 문자열 만들기 → 벡터 변환 → ID와 벡터의 대응 및
배열 검증 → 결과 요약을 outputs/exercise_03.json에 저장하는 순서입니다.

fixture 모드는 오프라인 모의 인코더로 흐름을 확인합니다.
live 모드는 준비된 환경에서 실제 E5 모델로 벡터를 만듭니다.
"""
import argparse
import sys
from pathlib import Path
# mentee/common에 있는 공용 day23kit 패키지를 찾을 수 있도록 경로를 추가합니다.
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.data import load_knowledge, E5_MODEL_ID
from day23kit.embedding import get_encoder
from day23kit.io import save_result

from tasks import make_passages, make_queries, encode_both, verify_alignment, check_shapes_vectors



def run(mode,level,args):
    """임베딩을 만들고 검증한 뒤 JSON에 담을 결과 딕셔너리를 반환합니다."""
    # 기초 단계는 처음 3건을 사용하고, 중급 단계는 지식 전체를 사용합니다.
    rows=load_knowledge()[:3] if level==1 else load_knowledge()
    encoder=get_encoder(mode)
    # 검색 대상은 passage:로, 검색할 질문은 query:로 시작하는 문자열로 준비합니다.
    passages=make_passages(rows)
    questions=make_queries(['서버는 어디에서 실행하나요?'])
    docs,queries=encode_both(encoder,passages,questions)
    # JSON에는 벡터 배열의 크기와 입력 예시를 기록합니다.
    # shape의 첫 숫자는 입력 개수, 두 번째 숫자는 벡터의 차원입니다.
    result=dict(exercise=3,level=level,source_mode='e5_live' if mode=='live' else 'simulated',
                model_id=E5_MODEL_ID if mode=='live' else encoder.model_id,
                document_shape=list(docs.shape),query_shape=list(queries.shape),
                example_passage=passages[0],example_query=questions[0])
    if level>=2:
        # 문서 ID가 가리키는 벡터 위치를 기록하고 문서·질문 배열을 각각 검증합니다.
        result['alignment']=verify_alignment(rows,docs)
        result['vectors_valid']=check_shapes_vectors(docs,queries,len(rows))
    if level>=3:result['model_contract']=make_model_stamp(encoder,mode)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 03 · E5 문장 Embedding')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    # 중급 난이도 2로 실행하고 결과를 터미널과 JSON에서 확인할 수 있게 합니다.
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_03.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
