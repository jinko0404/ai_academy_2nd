"""Day21 실습 04: Context 선택·예산. 실행: python main.py"""
# 전체 흐름
# 1. main()에서 전체 지식 파일을 읽습니다.
# 2. select_context()에서 선택한 ID의 지식을 골라 JSON 문자열로 바꿉니다.
# 3. calculate_budget()에서 근거 문맥에 쓸 수 있는 가상 token 예산을 계산합니다.
# 4. 포함한 ID와 제외한 ID·이유를 정리합니다.
# 5. 선택한 지식 원문과 선택·예산 기록을 outputs 폴더에 저장합니다.
# context는 모델이 답변할 때 참고할 자료를 뜻합니다.

# Path는 파일 경로, sys는 모듈 검색 경로, json은 JSON 문자열 변환에 사용합니다.
from pathlib import Path
import sys
import json

# 현재 main.py가 들어 있는 폴더의 절대 경로입니다.
HERE = Path(__file__).resolve().parent
# parents[0]은 01_basic, parents[1]은 mentee 폴더입니다.
ROLE_ROOT = HERE.parents[1]
# mentee 안의 coursekit 패키지를 가져올 수 있도록 검색 경로에 추가합니다.
sys.path.insert(0, str(ROLE_ROOT))
# DATA는 mentee/data 경로입니다. 파일 읽기·저장·실행 함수도 함께 가져옵니다.
from coursekit.io import DATA, read_json, write_json, write_text, run

# 기준: Day21 PPT 28~36장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.

def select_context(rows, selected_ids):
    # rows는 전체 지식 딕셔너리들의 목록, selected_ids는 사용할 ID 목록입니다.
    # 예를 들어 [102]는 ID가 102인 지식 하나를 선택하겠다는 뜻입니다.
    # TODO D21-04-B01: 선택 ID의 순서를 보존해 지식을 골라 JSON으로 직렬화하세요.
    # 입력·대상: knowledge_v1 목록, selected_ids=[102]
    # 완료 기준: 102만 포함한 JSON. 없는 ID는 ValueError. 순서가 바뀌면 문맥도 달라집니다.
    # 핵심: ID 선택과 직렬화일 뿐 의미 검색·RAG를 구현한 것이 아닙니다.
    # BEGIN D21-04-B01
    # {ID: 지식 딕셔너리}로 정리합니다. by_id[102]로 102번 지식을 바로 찾을 수 있습니다.
    by_id = {row["id"]:row for row in rows}
    # 딕셔너리는 같은 키를 하나만 보관합니다. 개수가 줄었다면 원본에 중복 ID가 있습니다.
    if len(by_id) != len(rows):
        raise ValueError("중복 ID를 확인하세요")
    # 중복 ID를 제거하면서 처음 나온 순서를 유지합니다.
    # 예를 들어 [102, 101, 102]를 넣으면 [102, 101]이 됩니다.
    ids = list(dict.fromkeys(selected_ids))  # 중복 선택은 첫 순서만 남김
    # 딕셔너리의 'in'은 키를 검사합니다. any()는 하나라도 True이면 True입니다.
    # 선택한 ID 중 원본에 없는 ID가 있으면 오류를 발생시킵니다.
    if any(i not in by_id for i in ids):
        raise ValueError("선택한 ID가 없습니다")
    # 선택한 지식을 담을 빈 목록을 만들고, 선택 ID의 순서대로 지식을 추가합니다.
    chosen = []
    for knowledge_id in ids:
        # 숫자 ID로 찾은 지식 딕셔너리 전체를 넣어 제목과 본문도 함께 보존합니다.
        chosen.append(by_id[knowledge_id])  # TODO: by_id[knowledge_id]
    # 파이썬 목록을 JSON 문자열로 변환합니다. ensure_ascii=False는 한글을 그대로 표시합니다.
    context_json = json.dumps(chosen, ensure_ascii=False)  # TODO: chosen
    # END D21-04-B01
    # 선택한 지식 목록과 JSON 문자열 두 값을 함께 돌려줍니다.
    return chosen, context_json

def calculate_budget():
    # 전체 가상 한도에서 다른 용도로 예약한 양을 빼 문맥에 남는 공간을 구합니다.
    # TODO D21-04-B02: PPT의 가상 token 예산에서 근거 문맥에 남는 공간을 구하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: available_for_context=4936
    # 핵심: 실제 모델 최대 길이나 측정 결과가 아닙니다. 입력 형식을 포함한 실제 token 측정은 별도입니다.
    # BEGIN D21-04-B02
    # window는 전체 한도, instructions_question은 지시문·질문에 쓸 양입니다.
    # examples는 예제, output_reserve는 답변용 예약량, margin은 여유분입니다.
    parts = {"window":8192,"instructions_question":800,"examples":600,"output_reserve":1600,"margin":256}
    # 미완성 TODO: parts의 값으로 8192 - 800 - 600 - 1600 - 256을 계산해야 합니다.
    available = parts["window"]-parts["instructions_question"]-parts["examples"]-parts["output_reserve"]-parts["margin"]  # TODO: window에서 지시·질문, 예제, 출력 예약, 여유를 뺍니다.
    # 계산 항목과 남은 예산을 묶습니다. source는 가상 계산이라는 기록입니다.
    result = {"source":"hypothetical_budget_not_model_measurement", "parts":parts, "available_for_context":available}
    # TODO: 결과 4936과 비교. len(text)를 token 수라고 부르지 마세요.
    # 예상값과 다르면 AssertionError로 중단됩니다. 현재 available=0이면 여기서 멈춥니다.
    # token은 모델이 텍스트를 나누는 단위이며, 문자열의 글자 수와 별도로 측정해야 합니다.
    assert available==4936
    # END D21-04-B02
    return result

def main():
    # 1단계. JSON 파일을 파이썬 목록으로 읽습니다. 경로의 /는 폴더와 파일 이름을 연결합니다.
    rows=read_json(DATA/"knowledge_v1.json")
    # 2단계. ID 102를 선택하고, 반환된 목록과 JSON 문자열을 각각 변수에 받습니다.
    chosen, context_json=select_context(rows,[102])
    # 3단계. 문맥에 쓸 가상 예산을 계산합니다.
    budget=calculate_budget()
    # TODO D21-04-B03: 포함·제외 ID와 원문·예산을 저장하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 포함 [102], 제외 101·103. 실제 입력 원문을 보존합니다.
    # BEGIN D21-04-B03
    # 4단계. 선택한 지식 각각에서 ID를 꺼내 목록으로 만들어야 합니다.
    # [row["id"] for row in chosen]으로 각 지식의 ID를 모을 수 있습니다.
    included_ids = [row["id"] for row in chosen]  # TODO: chosen의 id 목록
    # 전체 지식을 돌면서 포함 목록에 없는 ID와 제외 이유를 기록합니다.
    excluded = []
    for row in rows:
        if row["id"] not in included_ids:
            excluded.append({"id":row["id"], "reason":"이번 서버 실행 질문에서 선택하지 않음"})  # TODO: 제외 이유
    # manifest는 지식 버전, 포함·제외 목록, 예산을 정리한 기록입니다.
    manifest = {"context_version":"knowledge_v1", "included_ids":included_ids, "excluded":excluded, "budget":budget}
    # 5단계. context.json에는 이미 만든 JSON 문자열을 그대로 기록합니다.
    # context_manifest.json에는 딕셔너리를 JSON으로 변환해 기록합니다.
    # 저장 함수는 outputs 폴더를 생성하고 UTF-8로 저장합니다.
    write_text(HERE/"outputs"/"context.json", context_json)  # TODO: context_json
    write_json(HERE/"outputs"/"context_manifest.json", manifest)
    # END D21-04-B03
    print("[04] Context와 가상 예산 저장 완료. token 실측·검색 결과가 아닙니다.")

if __name__ == "__main__":
    # 이 파일을 직접 실행하면 run()이 main()을 호출합니다.
    # 다른 파일에서 import할 때는 main()이 자동 실행되지 않습니다.
    # run()은 파일 누락과 ValueError 등 일부 오류를 안내합니다.
    # 위의 AssertionError와 TypeError는 그대로 전파됩니다.
    run(main)
