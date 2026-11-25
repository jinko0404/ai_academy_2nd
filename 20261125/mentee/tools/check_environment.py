"""환경 확인; 기본 실행은 API/DB/GPU를 열지 않습니다."""
from pathlib import Path
import argparse,importlib.util,sys,os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true',help='CUDA·DB 연결까지 실제 검사')
    args=parser.parse_args()
    print('Python:',sys.version.split()[0])
    for name in ('langchain_core','langchain_openai','pydantic','fastapi','pytest','torch','sentence_transformers','psycopg','pgvector'):
        print(name, '설치됨' if importlib.util.find_spec(name) else '없음')
    if args.live:
        from coursekit.config import require_cuda
        print('CUDA device:',require_cuda())
        # API Key와 Password를 화면에 표시하지 않습니다.
        from coursekit.config import require_setting
        print('API Key 설정:',bool(require_setting('OPENAI_API_KEY')))
        print('PostgreSQL Password 설정:',bool(require_setting('PGPASSWORD')))
        from coursekit.live import open_db
        with open_db() as c:
            print('PostgreSQL SELECT 1:',c.execute('SELECT 1').fetchone()[0])
if __name__=='__main__':main()
