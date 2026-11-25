"""Day28 통합 평가 JSON을 엑셀/CSV에서 볼 질문별 표로 변환합니다."""
import csv,json,argparse
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('input',type=Path,help='08 실습의 outputs/result.json')
p.add_argument('--output',type=Path,default=None)
args=p.parse_args()
d=json.loads(args.input.read_text(encoding='utf-8'))
rows=d.get('questions')
if not isinstance(rows,list):raise SystemExit('실습 08의 result.json을 지정하세요.')
out=args.output or args.input.with_suffix('.csv')
out.parent.mkdir(parents=True,exist_ok=True)
columns=['question_id','mode','variant','recall_at_k','precision_at_k','hit_at_k','rr_at_k','ndcg_at_k','status','sources_valid','no_answer_result']
with out.open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
 for r in rows:
  m=r['metrics'];writer.writerow({
    'question_id':r['question_id'],'mode':d['mode'],'variant':r['variant'],
    'recall_at_k':m['recall_at_k'],'precision_at_k':m['precision_at_k'],
    'hit_at_k':m['hit_at_k'],'rr_at_k':m['rr_at_k'],'ndcg_at_k':m['ndcg_at_k'],
    'status':r['status'],'sources_valid':r['sources_valid'],
    'no_answer_result':r['no_answer_result']})
print('CSV 저장:',out)
