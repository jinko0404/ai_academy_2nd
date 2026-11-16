"""Day21 실습 02: Zero/Few-shot·버전. 실행: python main.py"""
# 전체 실행 흐름
# 1. 공통 모듈을 가져오고 데이터·출력 파일의 기준 경로를 설정합니다.
# 2. build_examples()에서 답변 가능한 경우와 정보가 부족한 경우의 예제를 만듭니다.
# 3. 같은 질문에 대해 예제가 없는 Zero-shot과 예제 두 개가 있는 Few-shot을 준비합니다.
# 4. 각 묶음을 프롬프트 문자열로 변환하고 예제 ID 포함 여부를 확인합니다.
# 5. 예제, 실험 설계 기록, 두 프롬프트를 outputs 폴더에 저장합니다.
# 이 실습에서는 프롬프트를 설계하며, 실제 모델 호출이나 응답 품질 평가는 수행하지 않습니다.

# Path는 파일 경로를 다루고, sys는 공통 모듈의 검색 경로를 설정하는 데 사용합니다.
from pathlib import Path
import sys

# 현재 파일이 있는 02_examples_versions 폴더의 절대 경로입니다.
HERE = Path(__file__).resolve().parent
# HERE.parents[0]은 01_basic, parents[1]은 mentee 폴더입니다.
ROLE_ROOT = HERE.parents[1]
# mentee 안의 coursekit 패키지를 import할 수 있도록 검색 경로에 추가합니다.
sys.path.insert(0, str(ROLE_ROOT))
# DATA는 mentee/data 경로입니다. 읽기·저장 함수와 공통 실행 함수도 가져옵니다.
from coursekit.io import DATA, read_json, write_json, write_text, run
# make_bundle()은 지식 스냅샷, 템플릿, 출력 규칙, 예제 등을 딕셔너리로 묶습니다.
from coursekit.prompting import make_bundle

# 기준: Day21 PPT 14~20장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.
# render_prompt()는 묶음과 질문 사례를 받아 실제 프롬프트 문자열을 만듭니다.
from coursekit.prompting import render_prompt

def build_examples():
    # Few-shot에 넣을 시범 답변 두 개를 만듭니다.
    # 각 예제는 question(질문), context(근거), answer(정답 형식)으로 구성됩니다.
    # TODO D21-02-B01: 예제 ID 901을 사용해 정상 답과 정보 부족 답을 완성하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 정상·부족 예제 2개. 평가 질문 D/H를 Few-shot에 사용하지 않습니다.
    # 핵심: 입력 예제를 늘려도 모델 가중치가 바뀌는 것은 아닙니다.
    # BEGIN D21-02-B01
    # 예제 전용 지식입니다. 장소 정보만 있으므로 대여료 질문에는 답할 근거가 없습니다.
    # 901은 이 예제 안의 출처 ID이며, 실제 질문의 출처로 그대로 복사하면 안 됩니다.
    example_context = [{"id":901,"content":"대여 장소는 안내 데스크입니다."}]
    examples = [
        # 첫 번째 예제는 근거로 답할 수 있는 경우입니다.
        # answered 상태에서는 출처를 기록하고 부족한 정보는 빈 문자열로 둡니다.
        {"question":"어디서 대여하나요?","context":example_context,
         "answer":{
            "status":"answered",
            "answer":"안내 데스크에서 대여합니다.",
            "source_ids":[901], 
            "missing_information":""
            }
        },
        # 두 번째 예제는 같은 근거에 질문의 답이 없는 경우입니다.
        # insufficient_context 상태에서는 정해진 안내 문구, 빈 출처 목록,
        # 부족한 사실의 이름을 기록해 추측하지 않는 응답 방식을 보여 줍니다.
        {"question":"대여료는 얼마인가요?","context":example_context,
         "answer":{
            "status":"insufficient_context", 
            "answer":"제공된 지식으로는 답할 수 없습니다.", 
            "source_ids":[], 
            "missing_information":"대여료"
            }
        },
    ]
    # TODO: 정상 답에는 [901], 정보 부족에는 []와 부족한 사실 이름을 사용합니다.
    # END D21-02-B01
    # 두 예제가 담긴 목록을 main()에 반환합니다.
    return examples

def main():
    # 1단계. Few-shot 프롬프트에 넣을 시범 예제를 준비합니다.
    examples=build_examples()
    # TODO D21-02-B02: 같은 질문에 Zero-shot과 Few-shot Prompt를 각각 만드세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 두 Prompt가 생성됩니다. 이는 05의 근거 규칙 A/B와 별도 설계 실험입니다.
    # BEGIN D21-02-B02
    # 2단계. 개발용 질문 목록을 읽고 [0]으로 첫 번째 사례를 선택합니다.
    # case에는 질문과 사용할 지식 ID가 들어 있으며, 두 프롬프트에 같은 사례를 씁니다.
    case = read_json(DATA/"dev_v1.json")[0]
    # 3단계. 예제 목록만 다르게 전달해 비교할 두 묶음을 만듭니다.
    # Zero-shot은 예제 0개, Few-shot은 위에서 작성한 예제 2개를 포함합니다.
    zero_bundle = make_bundle(examples=[])     # 예제 없는 입력
    few_bundle = make_bundle(examples=examples)      # TODO: examples 사용
    # 4단계. 두 묶음 모두 같은 "A" 근거 규칙을 선택해 예제 유무만 비교합니다.
    # render_prompt()는 case의 지식 ID에 해당하는 실제 지식을 골라 템플릿에 넣습니다.
    # 반환값은 (프롬프트 문자열, 지식 JSON 문자열)입니다.
    # 여기서는 첫 번째 값만 필요하므로 두 번째 값은 관례적으로 _에 받습니다.
    zero_prompt, _ = render_prompt(zero_bundle, case, "A")
    few_prompt, _ = render_prompt(few_bundle, case, "A")
    # 예제 ID가 Zero-shot에는 없고 Few-shot에는 있는지 검사합니다.
    # 조건이 거짓이면 AssertionError로 중단됩니다. 이 검사는 모델의 답변 품질을 평가하지 않습니다.
    assert "901" not in zero_prompt and "901" in few_prompt
    # TODO: 예제 ID 901의 포함 여부를 확인
    # END D21-02-B02
    # TODO D21-02-B03: 변경점·가설·고정 조건과 두 Prompt를 저장하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: 가설을 실제 성능 결과처럼 기록하지 않습니다.
    # BEGIN D21-02-B03
    # 5단계. 무엇을 바꾸고 무엇을 고정했는지 딕셔너리로 기록합니다.
    # base_version/new_version은 비교할 설계의 이름이고, changed는 변경한 조건입니다.
    # hypothesis는 앞으로 확인할 예상이며, fixed는 두 프롬프트에 공통으로 적용한 조건입니다.
    # source_mode의 design_only는 현재 산출물이 프롬프트 설계 단계임을 나타냅니다.
    experiment={
        "experiment":"examples_design_only",
        "base_version":"examples_zero_v1",
        "new_version":"examples_few_v1",
        "changed":"예제 유무",
        "hypothesis":"정상과 정보 부족의 출력 조합을 잘 따를지 관찰한다.",
        "fixed":["질문","knowledge_v1","answer_schema_v1","근거 규칙"],
        "source_mode":"design_only"
    }
    # TODO: 예제 유무를 바꿨다는 기록과 고정 조건을 작성
    # 6단계. / 연산자로 경로를 연결하고 공통 저장 함수로 UTF-8 파일을 기록합니다.
    # outputs 폴더가 없으면 저장 함수가 생성합니다.
    # examples.json은 시범 예제, experiment.json은 실험 설계 기록입니다.
    # zero_prompt.txt와 few_prompt.txt를 열면 모델에 전달할 입력을 직접 비교할 수 있습니다.
    write_json(HERE/"outputs"/"examples.json", examples)  # TODO: examples
    write_json(HERE/"outputs"/"experiment.json", experiment)
    write_text(HERE/"outputs"/"zero_prompt.txt", zero_prompt)  # TODO: zero_prompt
    write_text(HERE/"outputs"/"few_prompt.txt", few_prompt)   # TODO: few_prompt
    # END D21-02-B03
    print("[02] 예제·버전 설계 완료. 05는 예제 고정, 근거 규칙만 비교합니다.")

if __name__ == "__main__":
    # 이 파일을 직접 실행할 때 run()을 통해 main()을 호출합니다.
    # 다른 파일에서 import하면 함수 정의만 불러오고 main()은 자동 실행하지 않습니다.
    # run()은 미완성 TODO, 파일 누락, 잘못된 값·키 등의 오류를 안내합니다.
    run(main)
