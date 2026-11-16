"""Day21 실습 03: 작업 분해와 정보 부족. 실행: python main.py"""
# 전체 흐름
# 1. main()이 JSON 파일에서 판단 카드 목록을 읽습니다.
# 2. 첫 두 카드를 하나씩 analyze_card()에 전달합니다.
# 3. analyze_card()는 필요한 사실 중 근거에 없는 사실을 찾습니다.
# 4. 부족한 사실이나 충돌이 있으면 정보 부족, 모두 없으면 답변 가능으로 판단합니다.
# 5. 카드별 결과와 최종 출력 지시문을 outputs 폴더에 저장합니다.
# 판단 카드는 사람이 미리 정리한 데이터이며, 이 프로그램은 카드의 항목을 비교합니다.

# Path는 파일 경로를 다루고, sys는 공통 모듈을 찾을 경로를 설정하는 데 사용합니다.
from pathlib import Path
import sys

# 현재 main.py가 들어 있는 폴더의 절대 경로를 구합니다.
HERE = Path(__file__).resolve().parent
# parents[0]은 01_basic, parents[1]은 mentee 폴더입니다.
ROLE_ROOT = HERE.parents[1]
# mentee 안의 coursekit 패키지를 가져올 수 있도록 검색 경로에 추가합니다.
sys.path.insert(0, str(ROLE_ROOT))
# DATA는 mentee/data 경로입니다. 나머지는 파일 읽기·저장·실행을 돕는 함수입니다.
from coursekit.io import DATA, read_json, write_json, write_text, run

# 기준: Day21 PPT 21~27장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.

def analyze_card(card):
    # card는 질문 하나에 대한 판단 재료가 들어 있는 딕셔너리입니다.
    # required에는 답변에 필요한 사실 이름들이 들어 있습니다.
    # supported는 {사실 이름: 출처 ID 목록} 형태로 근거가 있는 사실을 담습니다.
    # conflicts에는 서로 충돌하는 내용이 미리 정리되어 있습니다.
    # TODO D21-03-B01: 질문이 요구하는 사실과, 제공 자료가 지원하는 사실을 구분하세요.
    # 입력·대상: 교사가 작성한 required/supported/conflicts 카드
    # 완료 기준: C01 missing=[], C02 missing=[보관 기간]
    # 핵심: 이 코드는 자유로운 질문의 의미를 자동 이해하지 않습니다. 명시된 판단 카드를 처리합니다.
    # BEGIN D21-03-B01
    # 필요한 사실 목록을 복사합니다. 예를 들면 ["보관 장소", "보관 기간"]입니다.
    required = list(card["required"])   # TODO: card["required"]
    supported = card["supported"]   # TODO: card["supported"]
    # 부족한 사실을 모을 빈 목록을 준비합니다.
    missing = []
    # 필요한 사실을 하나씩 꺼내 근거가 있는지 확인합니다.
    for fact in required:
        # 딕셔너리에서 'in'은 키를 검사합니다.
        # 예를 들어 "보관 기간"이라는 키가 supported에 없으면 부족한 사실로 추가합니다.
        if fact not in supported:
            missing.append(fact)    # TODO: supported에 없는 fact인지
    # 카드에 기록된 충돌 목록을 복사합니다.
    conflicts = list(card["conflicts"])   # TODO: card["conflicts"]
    # END D21-03-B01
    # TODO D21-03-B02: 하나라도 근거가 부족하면 수업의 정보 부족 상태를 선택하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: README의 예상 결과와 출력 파일을 대조합니다.
    # 핵심: 오늘 계약은 부분 답변을 새 상태로 추가하지 않습니다. 충분한 근거가 없으면 명시적으로 멈춥니다.
    # BEGIN D21-03-B02
    # TODO: missing 또는 conflicts가 있으면 부족, 아니면 answered
    # 빈 목록은 조건식에서 False, 항목이 있는 목록은 True로 취급됩니다.
    # missing 또는 conflicts 중 하나라도 차 있으면 insufficient_context가 됩니다.
    # 둘 다 비어 있으면 answered가 됩니다. 아래 한 줄은 if/else를 값으로 쓰는 표현입니다.
    status = "insufficient_context" if missing or conflicts else "answered"
    # 정보 부족 상태에서는 출처가 빈 목록으로 유지됩니다.
    source_ids = []
    # 답변 가능한 경우에만 필요한 사실들의 출처를 모읍니다.
    if status=="answered":
            for fact in required:
                # supported[fact]는 해당 사실을 뒷받침하는 출처 ID 목록입니다.
                # extend([1, 2])는 목록의 각 값을 추가합니다. 결과는 [1, 2] 형태입니다.
                source_ids.extend(supported[fact])
    # 같은 출처가 여러 사실을 지원할 수 있으므로 중복 ID를 제거합니다.
    # dict.fromkeys()는 처음 나온 순서대로 고유한 키를 만들고, list()는 키를 목록으로 바꿉니다.
    source_ids=list(dict.fromkeys(source_ids))
    # 카드 ID, 필요한 사실, 지원되는 사실, 부족·충돌 내용, 상태와 출처를 결과에 담습니다.
    # list(supported)는 딕셔너리의 키인 '지원되는 사실 이름'을 목록으로 만듭니다.
    # basis는 이 판단이 교사가 작성한 사실 카드를 바탕으로 했다는 기록입니다.
    result={"case_id":card["case_id"],"required":required,"supported_facts":list(supported),
            "missing":missing,"conflicts":conflicts,"status":status,"source_ids":source_ids,
            "basis":"teacher_authored_fact_card"}
    # END D21-03-B02
    # 카드 하나의 분석 결과를 main()으로 돌려줍니다.
    return result

def main():
    # JSON 파일을 읽어 파이썬 목록과 딕셔너리로 변환합니다.
    # 경로의 /는 폴더 경로와 파일 이름을 연결합니다.
    cards=read_json(DATA/"decomposition_cards.json")
    # TODO D21-03-B03: 정상·정보 부족의 판단과 최종 출력 지시를 저장하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 2개 결과와 작업 지시를 저장합니다. 모델 추론 관찰 기록이 아닙니다.
    # BEGIN D21-03-B03
    # 카드별 분석 결과를 모을 빈 목록입니다.
    results = []
    # [:2]는 목록의 처음 두 항목을 선택합니다.
    for card in cards[:2]:
        # 카드 하나를 분석한 뒤, 반환된 결과 딕셔너리를 results에 추가합니다.
        results.append(analyze_card(card))  # TODO: analyze_card(card)
    # 근거 대조, 정보 부족 시 처리, 최종 응답 형식을 안내할 지시문을 작성합니다.
    # 이 문자열은 아래에서 파일로 저장되며, 여기서 모델에 전송되지는 않습니다.
    instruction = "질문이 요구하는 사실과 근거를 대조하세요. 부족하면 추측하지 말고 필요한 정보를 적으세요. 내부 사고 기록 대신 최종 JSON만 반환하세요."  # TODO: 사실 대조·부족 처리·최종 JSON 요구(긴 내부 사고 요구 X)
    # 결과 목록은 JSON 파일로, 지시문은 텍스트 파일로 저장합니다.
    # 공통 저장 함수가 outputs 폴더를 만들고 UTF-8로 기록합니다.
    # 현재 코드에는 두 결과의 예상 상태를 확인하는 assert 검사가 아직 없습니다.
    write_json(HERE/"outputs"/"task_cards.json", results)
    write_text(HERE/"outputs"/"task_instruction.txt", instruction)
    # END D21-03-B03
    print("[03] 판단 카드 완료. 근거 대조는 사람의 판단이 포함됩니다.")

if __name__ == "__main__":
    # 이 파일을 직접 실행하면 run()이 main()을 호출합니다.
    # 다른 파일에서 import할 때는 main()을 자동으로 실행하지 않습니다.
    # run()은 미완성 TODO나 파일·데이터 관련 일부 오류를 안내하는 공통 함수입니다.
    run(main)
