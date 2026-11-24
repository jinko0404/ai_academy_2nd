"""Day26 실습. 실행 위치와 관계없이 현재 역할 폴더의 지원 모듈을 사용합니다."""

# 실습 목적: LangChain 구성요소 네 가지의 입력과 출력을 연결하고, 속성 읽기와 invoke 실행을 구분합니다.
# 전체 흐름: 입력·출력·접근 방식 표 만들기 → 뒤바뀐 출력 예시 고치기 → 수정 전후를 JSON으로 저장합니다.
# 이 실습은 각 구성요소의 역할을 문자열로 정리하는 로컬 개념 실습입니다.

from pathlib import Path
import sys

# 현재 파일 경로에서 실습 폴더와 멘티 루트를 찾아 공통 지원 모듈을 불러옵니다.
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import save_json, show_success
from coursekit.data import example_hit, load_hits
from coursekit.reporting import execution_record

# 강의 예시의 입력을 정리한 표입니다. 구성요소 이름을 키로 사용해 출력과 접근 방식도 연결합니다.
COMPONENT_INPUTS = {
    "Document": "page_content와 metadata를 담아 생성",
    "Retriever": "query·category·candidate_k가 있는 dict",
    "Prompt": "question·context가 있는 dict",
    "Chat Model": "ChatPromptValue 또는 메시지 목록",
}

def component_outputs() -> dict[str, str]:
    """강의에서 다룬 네 구성요소의 출력 자료형을 대응시킵니다."""
    # BEGIN D26-03-I01
    # TODO D26-03-I01: 네 구성요소의 반환값을 구분해 dict로 반환합니다.
    # 힌트: Prompt는 ChatPromptValue, Chat Model은 AIMessage입니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # Retriever는 검색 결과를 Document 목록으로 돌려줍니다. 선택한 문서의 본문은 Context를 만드는 데 쓰입니다.
    # Prompt는 질문과 Context를 메시지로 구성해 ChatPromptValue를 반환하고,
    # Chat Model은 준비된 메시지를 입력받아 생성한 응답을 AIMessage로 반환합니다.
    return {
        "Document": "본문·출처를 보관하는 데이터 객체",
        "Retriever": "list[Document]",
        "Prompt": "ChatPromptValue",
        "Chat Model": "AIMessage",
    }
    # END D26-03-I01

def access_kinds() -> dict[str, str]:
    """데이터를 읽는 경우와 invoke로 작업을 실행하는 경우를 구분합니다."""
    # BEGIN D26-03-I02
    # TODO D26-03-I02: Document는 data, 실행 구성요소는 invoke로 표시합니다.
    # 힌트: 값 읽기와 실행을 구분합니다. 여기서는 실제 API 호출을 하지 않습니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # Document는 document.page_content와 document.metadata처럼 속성을 읽는 데이터 객체입니다.
    # invoke는 각 구성요소에 입력을 전달해 검색, 메시지 구성, 답변 생성 같은 작업을 실행하는 메서드입니다.
    return {
        "Document": "data",
        "Retriever": "invoke",
        "Prompt": "invoke",
        "Chat Model": "invoke",
    }
        # END D
    # END D26-03-I02

def repair_swapped_returns(swapped: dict[str, str]) -> dict[str, str]:
    """입력 표를 복사하고 Prompt와 Chat Model의 잘못된 출력만 고칩니다."""
    # BEGIN D26-03-I03
    # TODO D26-03-I03: 잘못된 표를 복사해 Prompt·Chat Model의 출력 두 곳을 고칩니다.
    # 힌트: component_outputs에서 올바른 값을 읽고 원본 dict는 유지합니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # 수정 전후를 비교할 수 있도록 원본 표를 복사하고, 복사본의 두 출력만 올바른 값으로 바꿉니다.
    fixed = dict(swapped)
    outputs = component_outputs()
    fixed["Prompt"] = outputs["Prompt"]
    fixed["Chat Model"] = outputs["Chat Model"]
    return fixed
    # END D26-03-I03

def main():
    # 구성요소 이름을 공통 키로 써서 각 행에 입력, 출력, 접근 방식을 모읍니다.
    outputs = component_outputs()
    access = access_kinds()
    report = {
        # 실행 정보에는 개념 검토 모드와 DB·임베딩·LLM 실행 여부(False)가 기록됩니다.
        "execution": execution_record("concept_review"),
        "components": [{"component": name, "input": value,
                        "output": outputs[name], "access": access[name]}
                       for name, value in COMPONENT_INPUTS.items()],
        "observation": "입력·출력 대응표; Prompt·모델 invoke는 실행하지 않음",
    }
    # 올바른 표를 펼친 뒤 두 출력을 일부러 바꾼 예시를 만들고, 수정 결과를 함께 기록합니다.
    before = {**outputs, "Prompt": "AIMessage", "Chat Model": "ChatPromptValue"}
    report["before"] = before
    report["after"] = repair_swapped_returns(before)
    # 현재 실습 폴더의 outputs/object_contracts.json에 저장하고 터미널에 저장 경로를 표시합니다.
    path = save_json(HERE, "object_contracts.json", report)
    show_success("03 응용", path)

# tasks.py를 직접 실행하면 main.py와 같은 main()을 호출합니다.
if __name__ == "__main__":
    main()
