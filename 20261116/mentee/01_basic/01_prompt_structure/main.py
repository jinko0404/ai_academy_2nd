"""Day21 실습 01: Prompt 구조와 요구사항. 실행: python main.py"""
# 전체 흐름
# 1. 공통 모듈을 불러올 경로를 설정합니다.
# 2. build_requirements()에서 모델에게 전달할 여섯 가지 요구사항을 만듭니다.
# 3. main()에서 지식 파일의 첫 항목을 읽고 질문과 함께 build_prompt()에 전달합니다.
# 4. build_prompt()에서 요구사항, 질문, 지식, 출력 규칙을 하나의 문자열로 조립합니다.
# 5. main()에서 요구사항을 확인하고 outputs 폴더에 JSON과 텍스트로 저장합니다.
# 이 파일의 실행 범위는 프롬프트 작성과 저장까지이며, 모델 호출은 포함하지 않습니다.

# Path는 파일 경로, sys는 모듈 검색 경로, json은 JSON 직렬화에 사용합니다.
# Template은 문자열 안의 $변수 자리에 실제 값을 넣는 도구입니다.
from pathlib import Path
import sys
import json
from string import Template

# __file__은 현재 파일의 경로입니다. resolve()로 절대 경로를 얻고
# parent로 main.py가 들어 있는 01_prompt_structure 폴더를 구합니다.
HERE = Path(__file__).resolve().parent
# parents[0]은 01_basic, parents[1]은 mentee 폴더입니다.
ROLE_ROOT = HERE.parents[1]
# mentee를 모듈 검색 경로 맨 앞에 추가해 그 안의 coursekit을 가져옵니다.
sys.path.insert(0, str(ROLE_ROOT))
# DATA는 mentee/data 경로이며, 나머지는 공통 파일 읽기·저장·실행 함수입니다.
from coursekit.io import DATA, read_json, write_json, write_text, run
# CONTRACT_TEXT에는 응답 JSON의 필드, 자료형, 근거 부족 시 처리 규칙이 들어 있습니다.
from coursekit.prompting import CONTRACT_TEXT

# 기준: Day21 PPT 7~13장. 실제 모델 호출/실습 데이터 수정은 하지 않습니다.
# outputs는 실행 시 생성됩니다. 05/08의 모델 응답은 직접 수집한 원문만 등록합니다.

def build_requirements():
    # 요구사항을 딕셔너리로 묶습니다. 각 키는 아래 Template의 $변수와 연결됩니다.
    # role은 작성자의 역할, task는 수행할 작업, input은 사용할 입력입니다.
    # constraints는 지킬 제약, failure는 근거 부족 시 처리, output은 응답 형식입니다.
    # TODO D21-01-B01: 역할·작업·입력·제약·실패·출력 여섯 칸을 완성하세요.
    # 입력·대상: PPT 8장의 여섯 요소
    # 완료 기준: 6개 키가 모두 있고 각 문장이 비어 있지 않습니다.
    # 핵심: 역할을 전문가로 꾸미는 것과 정확한 근거를 제공하는 것은 다릅니다.
    # BEGIN D21-01-B01
    requirements = {
        "role": "초보 수강생의 실습 도움말 작성자",         # TODO: 누구를 돕는 작성자인가?
        "task": "제공된 지식으로 질문에 한국어 2문장 이내로 답합니다.",         # TODO: 어떤 질문에, 어떤 언어·길이로 답할까?
        "input": "사용자 질문과 선택한 지식 목록",        # TODO: 질문과 선택 지식
        "constraints": "제공된 근거만 사용하고 예제 ID를 답에 복사하지 않습니다.",  # TODO: 근거 범위와 예제 ID 사용 금지
        "failure": "답을 뒷받침하는 사실이 없으면 정보 부족을 명시합니다.",      # TODO: 없는 사실을 추측하지 않는 처리
        "output": "status, answer, source_ids, missing_information 네 필드 JSON",       # TODO: 출력할 네 필드 이름
    }
    # END D21-01-B01
    # 완성된 딕셔너리를 main()에 돌려줍니다.
    return requirements

def build_prompt(question, rows, requirements):
    # question은 질문 문자열, rows는 선택한 지식 목록,
    # requirements는 build_requirements()가 만든 딕셔너리입니다.
    # TODO D21-01-B02: Template에 질문·직렬화한 지식·출력 계약을 대입하세요.
    # 입력·대상: question, rows, requirements; CONTRACT_TEXT는 제공 코드
    # 완료 기준: 질문·지식 원문·네 필드 약속이 prompt에 있으며 $question이 남지 않습니다.
    # 핵심: 지식 본문은 데이터입니다. Template 대입은 코드 실행 또는 보안 권한 부여가 아닙니다.
    # BEGIN D21-01-B02
    # 파이썬 목록과 딕셔너리를 프롬프트에 넣을 JSON 문자열로 바꿉니다.
    # ensure_ascii=False는 한글을 유니코드 이스케이프 대신 그대로 표시합니다.
    context_json = json.dumps(rows, ensure_ascii=False)  # TODO: rows
    # JSON 중괄호와 충돌하지 않도록 $변수 형식인 Template을 사용합니다.
    template = Template("[역할] $role\n[작업] $task\n[입력] $input\n[질문] $question\n[지식 데이터] $context_json\n[제약] $constraints\n[실패] $failure\n[출력] $contract")
    # **requirements는 딕셔너리를 role=..., task=... 등의 이름 있는 인자로 펼칩니다.
    # 나머지 인자로 질문, 지식 JSON, 공통 출력 계약을 넣습니다.
    # substitute()는 필요한 변수의 값이 없으면 KeyError를 발생시킵니다.
    # 템플릿의 \n은 완성된 문자열에서 줄바꿈으로 표시됩니다.
    # requirements의 output 설명 대신 [출력]에는 상세 규칙인 CONTRACT_TEXT가 들어갑니다.
    prompt = template.substitute(
        **requirements,
        question=question,           # TODO: question
        context_json=context_json,       # TODO: 위 context_json
        contract=CONTRACT_TEXT,
    )
    # END D21-01-B02
    # 조립한 프롬프트 문자열을 호출한 main()에 돌려줍니다.
    return prompt

def main():
    # 1단계. 프롬프트에 사용할 여섯 가지 요구사항을 준비합니다.
    requirements=build_requirements()
    # 2단계. JSON 파일을 파이썬 객체로 읽고, [:1]로 첫 번째 항목만 목록으로 선택합니다.
    rows=read_json(DATA / "knowledge_v1.json")[:1]
    # 3단계. 고정 질문, 선택한 지식, 요구사항을 전달해 프롬프트를 만듭니다.
    prompt=build_prompt("파일을 어디에 보관하나요?",rows,requirements)
    # TODO D21-01-B03: 요구사항과 Prompt 원문을 저장하고 기준 질문을 확인하세요.
    # 입력·대상: 아래 함수 인자와 제공 데이터만 사용합니다.
    # 완료 기준: outputs/requirements.json 및 prompt.txt 생성. 아직 실제 모델 품질 결과는 없습니다.
    # BEGIN D21-01-B03
    # 4단계. 현재 검사는 요구사항의 개수만 확인합니다.
    # assert 조건이 거짓이면 AssertionError가 발생하며 아래 저장 단계로 진행하지 않습니다.
    # 각 값이 비어 있는지 확인하는 검사와 질문·지식 포함 검사는 아직 TODO로 남아 있습니다.
    assert len(requirements) == 6
    # TODO: prompt에 질문과 '개인 폴더'가 있는지 확인하세요.
    # 5단계. 경로의 / 연산자는 폴더와 파일 이름을 연결합니다.
    # 공통 저장 함수가 outputs 폴더를 만들고 UTF-8로 기록합니다.
    # requirements.json에는 요구사항 딕셔너리, prompt.txt에는 조립한 문자열이 저장됩니다.
    write_json(HERE / "outputs" / "requirements.json", requirements)  # TODO: requirements
    write_text(HERE / "outputs" / "prompt.txt", prompt)        # TODO: prompt
    # END D21-01-B03
    print("[01] Prompt 요구사항과 원문 저장 완료. 모델 결과가 아닙니다.")

if __name__ == "__main__":
    # 이 파일을 직접 실행할 때의 진입점입니다. import할 때는 실행되지 않습니다.
    # 공통 run()이 main()을 호출하고 미완성 TODO나 파일·데이터 관련 오류를 안내합니다.
    # AssertionError는 run()의 처리 대상에 포함되지 않아 그대로 전파됩니다.
    run(main)
