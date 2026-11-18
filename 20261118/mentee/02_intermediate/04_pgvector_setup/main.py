"""Day23 실습 04 · 서버 확장과 Python 연결 (응용, mentee).

이전 실습에서 텍스트를 벡터로 바꿨습니다. 이번에는 PostgreSQL에서
pgvector를 쓸 수 있는지 확인하고, Python으로 접속해 간단한 거리 계산을 시험합니다.
pgvector는 PostgreSQL에서 벡터를 저장하고 비교할 수 있게 해 주는 확장입니다.

실행 순서
1. pgAdmin이나 Python에서 사용할 점검 SQL을 준비합니다.
2. 중급 단계에서는 예시 조회 결과와 오류 메시지의 분류를 확인합니다.
3. live 모드에서는 실제 서버에 연결해 확장 상태를 확인하고, 활성화된 경우 거리를 계산합니다.
4. 결과를 터미널에 표시하고 outputs/exercise_04.json에 저장합니다.

python main.py --mode fixture는 DB에 연결하지 않고 예시만 확인합니다.
python main.py --mode live는 pgvector 사전 설정과 main.py 옆 .env의 접속 정보를 사용합니다.
"""
import argparse
import sys
from pathlib import Path
# mentee/common을 검색 경로에 추가해 공용 day23kit 패키지를 불러옵니다.
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'common'))
from day23kit.db import connect_db
from day23kit.io import save_result

from tasks import sql_server_info, sql_extension_state, sql_distance_probe, check_extension_row, classify_install_error



def run(mode,level,args):
    """점검에 사용할 SQL과 모드별 확인 결과를 딕셔너리로 모읍니다."""
    # tasks의 세 함수는 SQL 문자열을 만듭니다. 이 줄에서는 SQL을 실행하지 않습니다.
    sqls={'server':sql_server_info(),'extension':sql_extension_state(),'distance':sql_distance_probe()}
    result=dict(exercise=4,level=level,source_mode='not_run' if mode=='fixture' else 'pgvector_live',
                sql_for_pgadmin=sqls,db_check='not_run')
    if level>=2:
        # 세 번째 값이 None인 예시는 서버에 확장이 있고 현재 DB에는 미설치인 경우입니다.
        # 두 예시 모두 미리 정한 값이며 실제 DB 조회나 설치 로그에서 얻은 값은 아닙니다.
        result['illustrated_extension_state']=check_extension_row(('vector','0.8.7',None))
        result['example_error_stage']=classify_install_error('postgres.h not found')
    if level>=3:
        result['manual_pending']=audit_install_steps({})
    if mode=='live':
        # live만 실제 DB 연결. Windows 빌드 및 CREATE EXTENSION은 사전 수동 절차.
        with connect_db(Path(__file__).with_name('.env')) as conn:
            # fetchone()은 결과 한 행을 꺼냅니다. [0]은 그 행의 첫 번째 열입니다.
            ver=conn.execute('SELECT version()').fetchone()[0]
            dbname=conn.execute('SELECT current_database()').fetchone()[0]
            ext=conn.execute("SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name='vector'").fetchone()
            result['db_check']={'server_version':ver,'database':dbname,'extension':check_extension_row(ext)} if level>=2 else {'server_version':ver,'database':dbname,'extension_provided':ext is not None}
            # ext[2]는 현재 DB에 설치된 버전입니다. 버전이 있을 때만 vector 타입으로 계산합니다.
            if ext is not None and ext[2]:
                result['db_check']['distance_probe']=float(conn.execute(sql_distance_probe()).fetchone()[0])
            else:
                result['db_check']['distance_probe']='not_run_extension_not_activated'
    return result


def main():
    parser=argparse.ArgumentParser(description='Day23 실습 04 · 서버 확장과 Python 연결')
    parser.add_argument('--mode',choices=['fixture','live'],default='fixture',
        help='fixture=모의 벡터/오프라인, live=실제 E5/DB가 필요한 단계')
    args=parser.parse_args()
    # 중급 난이도인 2로 실행하고 main.py가 있는 폴더를 기준으로 결과를 저장합니다.
    result=run(args.mode,2,args)
    path=save_result(Path(__file__).resolve().parent,'exercise_04.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('결과 저장:',path)

if __name__=='__main__':
    main()
