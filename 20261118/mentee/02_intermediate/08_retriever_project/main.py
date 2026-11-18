"""Day23 실습 08 · AI Agent용 Retriever (응용, mentee).

실습 목표는 질문을 입력받아 답변의 근거가 될 문서를 찾아 주는 검색기
(Retriever)를 만드는 것입니다. 검색기는 문서의 원문과 거리를 반환합니다.
찾은 문서를 바탕으로 LLM이 답변을 작성하는 단계는 Day25에서 다룹니다.

실행 흐름
1. 질문의 앞뒤 공백을 제거하고 검색 개수와 분류를 검사합니다.
2. 질문과 가까운 문서를 최대 k건 찾고 검색에 걸린 시간을 측정합니다.
3. 문서의 원문, ID, 버전, 거리와 실행 요약을 JSON으로 저장합니다.

python main.py --mode fixture는 로컬 모의 검색으로 흐름을 확인합니다.
python main.py --mode live는 E5와 PostgreSQL을 사용합니다.
live에서는 실습 05에서 문서 벡터를 저장한 DB의 접속 정보를 .env에 설정합니다.
"""
import argparse
import sys
from pathlib import Path
# mentee/common을 검색 경로에 추가해 공용 day23kit을 불러옵니다.
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from time import perf_counter
from day23kit.io import save_result

from tasks import validate_question, retrieve_candidates, format_candidates, validate_options, make_run_report



def run(mode,level,args):
    """질문과 검색 조건을 확인하고, 검색 결과를 저장할 딕셔너리를 만듭니다."""
    # 중급 실습은 명령줄에서 받은 k와 category를 검색에 사용합니다.
    question=validate_question(getattr(args,'question','서버는 어디에서 실행하나요?'))
    k=3 if level==1 else args.k
    category=None if level==1 else args.category
    if level>=2:options=validate_options(k,category)
    else:options={'k':k,'category':category}
    # 검색 함수 호출 직전과 직후의 차이를 잽니다. live에서는 모델 로드와 DB 연결도 포함됩니다.
    t0=perf_counter()
    hits=retrieve_candidates(question,mode,options['k'],options['category'],Path(__file__).with_name('.env'))
    elapsed=(perf_counter()-t0)*1000  # 초를 밀리초(ms)로 바꿉니다.
    formatted=format_candidates(hits)
    # retrieved에는 후보 문서를 담고, 사람이 근거를 확인할 때까지 human_evidence_ok는 None으로 둡니다.
    result=dict(exercise=8,level=level,source_mode='simulated' if mode=='fixture' else 'pgvector_exact',question=question,options=options,
                retrieved=formatted,retrieved_count=len(formatted),human_evidence_ok=None,
                next_step='검토 후 Context 후보로 전달; LLM 답변 생성은 Day25 RAG에서 수행')
    if level>=2:result['run_report']=make_run_report(question,hits,mode,elapsed)
    if level>=3:result['contract_check']=check_model_and_versions(hits,mode)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 08 · AI Agent용 Retriever')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')

    parser.add_argument('--question',default='서버는 어디에서 실행하나요?',help='검색 질문')
    parser.add_argument('--k',type=int,default=3,help='반환할 최대 검색 후보 (1~9)')
    parser.add_argument('--category',default=None,choices=['course','development','support'],help='Metadata 분류 필터')
    args=parser.parse_args()
    # 중급 난이도 2로 실행하고, 터미널과 outputs/exercise_08.json에 결과를 남깁니다.
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_08.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
