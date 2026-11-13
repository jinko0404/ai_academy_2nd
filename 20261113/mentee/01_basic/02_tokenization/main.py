# 전체 흐름: 문장 준비 → 토큰과 번호로 변환 → 길이 맞추기 → 예산 계산 → JSON 저장.
# 이 파일은 실제 LLM을 호출하지 않고, 공통 파일의 학습용 도구로 계산한다.

# Path는 파일과 폴더의 경로를 다루는 도구이다.
from pathlib import Path
# sys는 파이썬 실행 환경을 다루는 도구이다.
import sys

# __file__: 현재 main.py의 경로. resolve(): 절대 경로로 변환.
# parent: 해당 파일이 들어 있는 폴더. 여기서는 02_tokenization 폴더이다.
HERE = Path(__file__).resolve().parent
# 한 단계 위인 01_basic 폴더이며, 그 안에 common 폴더가 있다.
LEVEL_ROOT = HERE.parent
# import할 모듈을 찾는 폴더 목록 맨 앞(0번)에 01_basic 경로를 추가한다.
# str()은 Path 객체를 경로 문자열로 바꾼다.
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# DATA: A/B/C 문장 정보가 들어 있는 리스트.
# ToyTokenizer: 문장을 쪼개고 토큰마다 번호를 붙이는 학습용 클래스.
# context_budget: 입력과 출력 예약량을 빼고 남는 토큰 예산을 계산하는 함수.
# write_json: 파이썬 데이터를 JSON 파일로 저장하는 함수.
from common.support import DATA, ToyTokenizer, context_budget, write_json

# 주의: 여기의 ToyTokenizer는 규칙 기반 학습용이며 실제 BERT 서브워드 출력과 다릅니다.
# def는 함수를 정의한다. 아래 들여쓴 코드는 맨 아래에서 main()을 호출할 때 실행된다.
def main():
    # DATA는 [{"id": "A", "content": "실습 파일은 ...", ...}, ...] 형태이다.
    # item에는 딕셔너리 하나씩 들어가고, item["content"]로 문장만 꺼낸다.
    # [ ... for ... ]는 꺼낸 결과를 새로운 리스트로 모으는 문법이다.
    # 풀어 쓰면:
    # texts = []
    # for item in DATA:
    #     texts.append(item["content"])
    texts = [item["content"] for item in DATA]

    # 리스트 + 리스트는 두 리스트의 항목들을 이어 붙인다.
    # A/B/C 문장에 예시 문장 3개를 추가해서 토크나이저를 만든다.
    # ToyTokenizer(...)로 만든 객체를 tok라는 변수에 저장한다.
    # 내부에서는 문장들을 규칙으로 쪼개고, 중복을 없애고, 정렬해 어휘표를 만든다.
    # 어휘표는 {토큰 문자열: 정수 번호} 형태이다. 여기서 LLM 학습을 하는 것은 아니다.
    # [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3이고 나머지 토큰에도 번호가 붙는다.
    tok = ToyTokenizer(texts + ["AI", "FastAPI와 PostgreSQL", "print(x) // code"])

    # TODO D20-02-B01: 제공된 지식 A/B/C 세 문장을 학습용 Token·Token ID로 변환하세요.
    # 입력·대상: 공통 A/B/C 문장을 쓰고 특수 토큰과 PAD를 구분합니다.
    # 완료 확인: outputs/token_report.json 의 tokens·ids·count·mask·budget 값을 확인합니다.
    # 주의: ToyTokenizer는 실제 서브워드 토크나이저가 아니며 임의 모델의 Token 수를 대표하지 않습니다.
    # BEGIN D20-02-B01
    # 문장별 분석 결과를 하나씩 담을 빈 리스트이다.
    rows = []
    # DATA의 A/B/C 항목을 하나씩 꺼내 item에 넣고 아래 코드를 반복한다.
    for item in DATA:
        # encode는 문장을 토큰으로 나누고, 앞뒤에 [BOS]/[EOS]를 붙인 다음
        # 각 토큰을 어휘표의 정수 ID로 바꾼다. 결과는 딕셔너리로 돌려준다.
        # 예: tok.encode("AI")의 주요 결과는 다음과 같다.
        # {"tokens": ["[BOS]", "AI", "[EOS]"], "input_ids": [2, AI의 ID, 3],
        #  "valid_length": 3, "attention_mask": [1, 1, 1], ...}
        # 위 'AI의 ID' 자리에는 실제 정수가 들어가며, 준비 문장에 따라 달라질 수 있다.
        record = tok.encode(item["content"])  # TODO: item의 content 문자열로 바꾸세요.
        # { ... }로 이번 문장의 결과 딕셔너리를 만들고 append로 rows 끝에 추가한다.
        # "이름": 값에서 왼쪽은 결과 항목 이름, 오른쪽은 그 항목에 넣을 값이다.
        rows.append({
            # 원본 항목의 식별자와 문장을 그대로 보관한다.
            "id": item["id"],
            "text": item["content"],
            # record["tokens"]는 딕셔너리에서 "tokens"라는 키의 값을 꺼내는 문법이다.
            # 꺼낸 값이 이미 리스트이므로 바깥에 []를 한 번 더 씌우지 않는다.
            # record["tokens"]   → ["[BOS]", "AI", "[EOS]"]
            # [record["tokens"]] → [["[BOS]", "AI", "[EOS]"]] (리스트가 한 겹 늘어남)
            "tokens": record["tokens"],    # TODO: record에서 Token 목록 가져오기
            # 토큰 문자열 대신 정수 번호들을 담은 리스트. tokens와 순서가 대응한다.
            "ids": record["input_ids"],       # TODO: record에서 input_ids 가져오기
            # PAD를 제외한 개수이며, BOS/EOS는 포함한다. A/B/C는 각각 8, 7, 9이다.
            "count": record["valid_length"],      # TODO: PAD 제외 유효 Token 개수
            # 이 실습에서는 문장 시작 BOS와 문장 끝 EOS를 하나씩 붙이므로 2개이다.
            "special_count": 2,  # TODO: 이 ToyTokenizer의 BOS/EOS 개수
        })
    # END D20-02-B01

    # TODO D20-02-B02: 길이가 다른 두 입력을 Padding하고 attention_mask의 0/1을 확인하세요.
    # 입력·대상: 공통 A/B/C 문장을 쓰고 특수 토큰과 PAD를 구분합니다.
    # 완료 확인: outputs/token_report.json 의 tokens·ids·count·mask·budget 값을 확인합니다.
    # 주의: ToyTokenizer는 실제 서브워드 토크나이저가 아니며 임의 모델의 Token 수를 대표하지 않습니다.
    # BEGIN D20-02-B02
    # batch는 여러 문장을 함께 처리하고 가장 긴 문장에 맞춰 길이를 맞춘다.
    # 짧은 문장 뒤에 [PAD]를 붙이는 것을 패딩(Padding)이라고 한다.
    # "AI" → ["[BOS]", "AI", "[EOS]", "[PAD]", "[PAD]"]
    # "FastAPI와 PostgreSQL" → ["[BOS]", "FastAPI", "와", "PostgreSQL", "[EOS]"]
    # 첫 문장의 유효 토큰 수는 3이지만, 패딩 후 리스트 길이는 5이다.
    # batch의 결과는 [첫 문장 결과 딕셔너리, 두 번째 문장 결과 딕셔너리]이다.
    batch = tok.batch(["AI", "FastAPI와 PostgreSQL"])  # TODO: "AI", "FastAPI와 PostgreSQL" 입력
    # TODO: batch[0]과 batch[1]의 input_ids 길이를 비교하는 assert
    # batch[0]은 첫 번째 결과, batch[1]은 두 번째 결과이다. 인덱스는 0부터 시작한다.
    # len()은 항목 개수를 구하고, ==는 두 값이 같은지 비교한다.
    # assert 조건은 조건이 참이면 넘어가고, 거짓이면 AssertionError로 중단한다.
    # 여기서는 패딩한 두 ID 리스트의 길이가 모두 5인지 서로 비교한다.
    assert len(batch[0]["input_ids"]) == len(batch[1]["input_ids"])
    # TODO: attention_mask와 input_ids의 길이를 비교하는 assert
    # attention_mask는 각 위치가 실제 토큰이면 1, PAD이면 0인 리스트이다.
    # 첫 문장은 [1, 1, 1, 0, 0], 두 번째 문장은 [1, 1, 1, 1, 1]이다.
    # for b in batch로 각 결과를 꺼내 ID와 마스크 길이를 비교한다.
    # all(...)은 비교 결과가 모두 참일 때만 True이다.
    # 풀어 쓰면 다음과 같다.
    # for b in batch:
    #     assert len(b["input_ids"]) == len(b["attention_mask"])
    # 이 검사는 길이만 확인한다. 마스크 값이 올바른지는 결과를 별도로 살펴봐야 한다.
    assert all(len(b["input_ids"]) == len(b["attention_mask"]) for b in batch)
    # END D20-02-B02

    # TODO D20-02-B03: PPT 문맥 예시(300+1200+2000+150, 출력 예약 850)의 합과 남는 예산을 계산하세요.
    # 입력·대상: 공통 A/B/C 문장을 쓰고 특수 토큰과 PAD를 구분합니다.
    # 완료 확인: outputs/token_report.json 의 tokens·ids·count·mask·budget 값을 확인합니다.
    # 주의: ToyTokenizer는 실제 서브워드 토크나이저가 아니며 임의 모델의 Token 수를 대표하지 않습니다.
    # BEGIN D20-02-B03
    # 문맥 창(Context Window)은 입력과 생성할 출력을 담는 토큰 공간이다.
    # parts는 입력의 각 부분에 쓰는 토큰 수를 기록한 딕셔너리이다.
    # system: 지시문, history: 대화 기록, evidence: 참고 자료, user: 사용자 질문.
    # 실제 문장을 세어 얻은 수치가 아니라, 수업에서 주어진 예시 수치이다.
    parts = {"system": 300, "history": 1200, "evidence": 2000, "user": 150}  # TODO: PPT 수치 입력
    # 인자 순서: 전체 공간 5000, 입력별 사용량 parts, 출력에 남겨 둘 공간 850.
    # 함수 안에서 sum(parts.values())로 값들을 더하면 입력 사용량은 3650이다.
    # remaining = 5000 - 3650 - 850 = 500.
    # 반환값은 {"window": 5000, "input_used": 3650, "output_reserved": 850,
    #           "remaining": 500, "fits": True}이다.
    # fits는 입력 사용량 + 출력 예약량이 전체 공간 안에 들어가는지를 나타낸다.
    # 이 함수는 예산만 계산하며, 문장을 실제로 자르거나 출력을 생성하지 않는다.
    budget = context_budget(5000, parts, 850)  # TODO: 출력 예약 850 Token
    # 힌트: budget["remaining"]은 입력+출력 예약 뒤 남는 Token입니다.
    # END D20-02-B03
    # 아래 항목들은 후속 실습용 자리이며, 기본 단계에서는 비워 둔다.
    # {}는 빈 딕셔너리, []는 빈 리스트, None은 값이 없음을 나타낸다.
    # 잘림(truncation)은 길이 제한 때문에 실제 토큰 일부를 제거하는 처리이다.
    # 예산 계산과 달리, 이 파일에서는 잘림 사례를 실행하지 않는다.
    trunc_cases = {}
    # 배치의 추가 분석 결과를 담을 자리.
    batch_report = []
    # 언어별 토큰화 비교 결과를 담을 자리.
    mixed_rows = []
    # 작은 문맥 창에서 예산을 초과하는 예시를 담을 자리.
    small_budget = None
    # 결과 파일에 넣을 설명 문자열. 이 문장 자체가 어떤 처리를 수행하지는 않는다.
    policy = "예산과 잘림의 차이를 설명"

    # TODO D20-02-B04: Token·Mask·문맥 예산 결과를 outputs/token_report.json에 저장하세요.
    # 입력·대상: 공통 A/B/C 문장을 쓰고 특수 토큰과 PAD를 구분합니다.
    # 완료 확인: outputs/token_report.json 의 tokens·ids·count·mask·budget 값을 확인합니다.
    # 주의: ToyTokenizer는 실제 서브워드 토크나이저가 아니며 임의 모델의 Token 수를 대표하지 않습니다.
    # BEGIN D20-02-B04
    # 지금까지 만든 결과들을 딕셔너리 하나에 모아 저장할 준비를 한다.
    # 예: "records": rows는 rows의 값을 "records"라는 항목 이름으로 담는다.
    # 리스트, 딕셔너리, 숫자 등을 여기서 문자열로 직접 바꿀 필요는 없다.
    result = {
        "tokenizer_kind": "교육용 규칙 기반 (실제 사전학습 모델 아님)",
        "records": rows,          # TODO: 위에서 만든 rows
        "batch": batch,            # TODO: Padding 결과
        "budget": budget,           # TODO: Context Window 예산
        "truncation": trunc_cases,
        "batch_report": batch_report,
        "language_comparison": mixed_rows,
        "overflow_budget": small_budget,
        "policy": policy,
    }
    # Path 객체의 /는 경로를 이어 붙인다. 여기서는 숫자 나눗셈이 아니다.
    # 저장 위치: 현재 main.py 폴더 아래 outputs/token_report.json.
    # write_json은 폴더가 없으면 만들고, result를 JSON 문자열로 변환해 저장한다.
    # 한글을 유지하고 들여쓰기를 넣어 UTF-8 파일로 쓴다. None은 JSON의 null이 된다.
    write_json(HERE / "outputs" / "token_report.json", result)  # 결과를 파일에 저장
    # 파일에 저장하는 것과 별도로, 터미널 화면에 간단한 요약을 출력한다.
    # r에는 rows의 결과 딕셔너리가 하나씩 들어간다.
    # (r["id"], r["count"])는 식별자와 개수를 튜플 하나로 묶는다.
    # 리스트 컴프리헨션 결과: [("A", 8), ("B", 7), ("C", 9)].
    print("[02] Token 분석", [(r["id"], r["count"]) for r in rows])
    # END D20-02-B04

# 이 파일을 직접 실행하면 __name__이 "__main__"이 되어 main()을 실행한다.
# 다른 파일에서 이 파일을 import할 때는 main()을 자동으로 실행하지 않는다.
if __name__=="__main__":
    main()
