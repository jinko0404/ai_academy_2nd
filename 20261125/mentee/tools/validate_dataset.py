"""Gold/QID/Metadata의 정확성 검사. LLM 호출 없음."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from coursekit.eval_dataset import read_json,read_questions,read_gold,validate_dataset
if __name__=='__main__':
    gold,grades,version=read_gold()
    summary=validate_dataset(read_questions(),gold,grades,read_json('corpus.json'))
    print('DATASET VERIFIED',version,summary)
