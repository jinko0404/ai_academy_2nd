"""선택한 난이도의 8개 실습 실행. 완료하지 않은 TODO는 정상적으로 실패합니다."""
from pathlib import Path
import sys,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--level',choices=['01_basic','02_intermediate','03_advanced'],default='01_basic')
    p.add_argument('--tests',action='store_true',help='pytest 테스트도 실행')
    args=p.parse_args()
    folders=sorted((ROOT/args.level).iterdir())
    failed=[]
    for folder in folders:
        if not (folder/'main.py').exists():continue
        cmd=[sys.executable,str(folder/'main.py'),'--offline']
        proc=subprocess.run(cmd,cwd=folder)
        if proc.returncode:failed.append(folder.name)
        if args.tests:
            t=subprocess.run([sys.executable,'-m','pytest','-q',str(folder/'tests')],cwd=ROOT)
            if t.returncode and folder.name not in failed:failed.append(folder.name)
    print('실습 완료:',len(folders)-len(failed),'/',len(folders),'실패:',failed)
    return int(bool(failed))
if __name__=='__main__':raise SystemExit(main())
