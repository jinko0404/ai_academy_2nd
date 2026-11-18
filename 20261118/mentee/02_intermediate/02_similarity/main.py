"""Day23 실습 02 · Vector·Similarity·Distance (응용, mentee).

벡터 검색에서 질문과 문서가 얼마나 가까운지 수치로 비교하는 실습입니다.
내적, 코사인 유사도, L2 거리의 차이를 확인하고, 벡터의 길이가
내적과 코사인 유사도로 정렬한 검색 순위에 어떤 영향을 주는지 살펴봅니다.

실행: python main.py --mode fixture
이 단계에서는 코드에 정의한 2차원 벡터로 계산합니다.
E5 임베딩 생성과 PostgreSQL 검색은 이후 실습에서 다룹니다.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
import numpy as np
from day23kit.mathops import cosine_distance
from day23kit.io import save_result

from tasks import dot, cosine, l2, normalize, rank_dot_vs_cosine



def run(mode,level,args):
    """예제 벡터의 점수와 후보 순위를 계산해 결과 딕셔너리로 반환합니다."""
    # q는 질문, a는 점수 계산용 문서, u와 v는 순위 비교용 문서입니다.
    q=np.array([1.,0.]);a=np.array([.8,.6]);u=np.array([1.,0.]);v=np.array([10.,10.])
    # 유사도는 클수록, 거리는 작을수록 가깝습니다. 코사인 거리는 1 - 유사도입니다.
    result=dict(exercise=2,level=level,source_mode='local_math',
                dot=round(dot(q,a),6),cosine=round(cosine(q,a),6),l2=round(l2(q,a),6),
                cosine_distance=round(1-cosine(q,a),6))
    if level>=2:
        # 정규화한 v는 출력용입니다. 순위를 계산할 때는 원래 u와 v를 사용합니다.
        # 내적에서는 길이가 큰 V가, 코사인 유사도에서는 q와 방향이 같은 U가 1위입니다.
        result['normalized_v']=normalize(v).round(6).tolist()
        result['rankings']=rank_dot_vs_cosine(q,{'U':u,'V':v})
    if level>=3:
        # 고급 단계의 분기에서는 차원 불일치, NaN, 영벡터의 검증 오류를 수집합니다.
        result['validated']=validate_pair(q,a)
        errors={}
        for label,x,y in [('dimension_mismatch',q,np.array([1.,0.,2.])),
                           ('nonfinite',q,np.array([np.nan,1.])),
                           ('zero_vector',q,np.array([0.,0.]))]:
            try:validate_pair(x,y)
            except ValueError as e:errors[label]=str(e)
        result['invalid_inputs']=errors
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 02 · Vector·Similarity·Distance')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    # 중급 난이도 2로 계산한 결과를 이 파일 옆 outputs 폴더의 JSON에 저장합니다.
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_02.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
