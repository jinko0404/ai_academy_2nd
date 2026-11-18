"""Day23 실습 01 · 검색 대상과 ID (응용, mentee).

Day23 전체 목표는 질문과 관련된 지식을 찾아 ID, 제목, 본문, 거리를 반환하는
검색기를 만드는 것입니다. E5로 텍스트를 벡터로 바꾸고 pgvector로 검색합니다.

실습 01에서는 검색에 쓸 문서를 준비합니다. 로컬 지식 9건에서 기존 문서
101~103을 확인하고, 각 문서의 원문·ID·버전과 E5 입력 문자열을 함께 기록합니다.
ID 중복과 필수 필드를 검사한 뒤 결과를 outputs/exercise_01.json에 저장합니다.

실행: python main.py --mode fixture
이 실습은 두 모드 모두 로컬 데이터를 사용합니다. 실제 임베딩 생성과
PostgreSQL 검색은 이후 실습에서 다룹니다.
"""
import argparse
import sys
from pathlib import Path
# 현재 파일의 상위 mentee 폴더에 있는 공용 모듈 경로를 추가합니다.
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.data import load_knowledge
from day23kit.io import save_result

from tasks import select_original, make_passage_input, make_mapping, reject_duplicate_ids, reject_missing_fields



def run(mode,level,args):
    """로컬 지식을 읽어 문서 입력과 검증 결과를 하나의 딕셔너리로 모읍니다."""
    rows=load_knowledge()
    selected=select_original(rows)
    # 리스트 인덱스는 0부터 시작하므로 selected[1]은 두 번째 문서입니다.
    passage=make_passage_input(selected[1])
    # 기초는 선택한 문서만, 중급은 새 문서 104~109를 포함한 전체를 매핑합니다.
    mapping=make_mapping(selected if level==1 else rows)
    result=dict(exercise=1,level=level,source_mode='local_snapshot',
                selected_ids=[r['id'] for r in selected],example_passage=passage,
                total_mapped=len(mapping),mapping=mapping)
    if level>=2:
        # 중복 검사는 ID 집합을 반환하므로 len으로 고유 ID 개수를 구합니다.
        # 필수 필드 검사는 모든 행이 유효하면 True를 반환합니다.
        result['unique_ids']=len(reject_duplicate_ids(rows))
        result['all_fields_valid']=reject_missing_fields(rows)
    if level>=3:
        result['versions']=audit_versions(rows)
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 01 · 검색 대상과 ID')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    # 이 파일은 중급 실습이므로 난이도 2로 실행합니다.
    result=run(args.mode,2,args)
    # 터미널 위치와 관계없이 이 파일 옆 outputs 폴더에 저장합니다.
    path=save_result(Path(__file__).resolve().parent,'exercise_01.json',result)
    import json
    # 한글을 그대로 표시하고, 들여쓰기 두 칸으로 터미널에 출력합니다.
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
