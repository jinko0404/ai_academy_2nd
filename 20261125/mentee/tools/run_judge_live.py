"""명시적으로 한 건의 합성 Claim에 LLM-as-a-Judge 실행; 비용 발생 가능."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from coursekit.eval_dataset import read_json
from coursekit.eval_report import write_json
from coursekit.eval_judge import judge_case_live

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case-id',choices=['G01','G02','G03'],required=True)
    p.add_argument('--confirm-paid-api',action='store_true',help='실제 LLM API 호출/과금을 명시적으로 승인')
    args=p.parse_args()
    if not args.confirm_paid_api:
        p.error('--confirm-paid-api 옵션을 지정해야 실제 Judge를 실행합니다.')
    case=next((c for c in read_json('generation_claims.json') if c['case_id']==args.case_id),None)
    if case is None:raise ValueError('알 수 없는 Case ID')
    from coursekit.config import require_setting
    require_setting('OPENAI_API_KEY')
    result=judge_case_live(case,reference='교육용 케이스: 사람 검토가 최종 승인 필요')
    out={'case_id':args.case_id,'mode':'live_llm_judge_on_synthetic_case',
         'review':result.model_dump(),'limitations':'보조 판정. 사람 검토 후 승인 필요'}
    file=ROOT/'tools'/'outputs'/f'judge_{args.case_id}.json'
    write_json(file,out)
    print('LLM Judge 보조 평가 저장:',file)
if __name__=='__main__':main()
