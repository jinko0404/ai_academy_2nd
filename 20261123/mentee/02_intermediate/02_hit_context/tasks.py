"""Day26 실습. 실행 위치와 관계없이 현재 역할 폴더의 지원 모듈을 사용합니다."""

# 실습 목적: RAG의 Context Builder가 선택한 검색 결과(Hit)를 모델에 전달할 근거로 바꾸고,
# Context에 실제로 넣은 근거의 인용 ID를 함께 기록하는 방법을 익힙니다.
# 전체 흐름: 예시 Hit 읽기 → 입력 순서대로 선택 → Hit별 블록 구성 → Context와 ID 목록을 JSON에 저장.

from pathlib import Path
import sys

# 현재 파일의 경로를 기준으로 실습 폴더와 멘티 루트를 찾아 공통 지원 모듈을 불러옵니다.
HERE = Path(__file__).resolve().parent
ROLE_ROOT = HERE.parents[1]
sys.path.insert(0, str(ROLE_ROOT))
from coursekit.io import save_json, show_success
from coursekit.data import example_hit, load_hits
from coursekit.reporting import execution_record

def format_hit(hit: dict) -> str:
    """Hit 한 건의 ID·위치·본문을 한 문자열 블록으로 반환합니다."""
    # BEGIN D26-02-I01
    # TODO D26-02-I01: [ID] 원문위치, 줄바꿈, 본문 순서로 한 블록을 만듭니다.
    # 힌트: citation_id·source_locator·body를 읽어 f-string으로 연결합니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # 첫 줄에 인용 ID와 원문 위치를 표시하고, 다음 줄에 본문을 그대로 붙입니다.
    # ID와 위치를 본문과 함께 묶어 두면 Context에서 어떤 출처를 사용했는지 확인할 수 있습니다.
    citation_id = hit["citation_id"]
    locator = hit["source_locator"]
    body = hit["body"]
    return f"[{citation_id}] {locator}\n{body}"
    # END D26-02-I01

def select_hits(hits: list[dict], max_items: int = 2) -> list[dict]:
    """입력 순서를 유지해 앞의 최대 max_items건만 고릅니다."""
    # BEGIN D26-02-I02
    # TODO D26-02-I02: 입력 순서대로 앞의 최대 max_items건을 선택합니다.
    # 힌트: 목록 슬라이스를 사용합니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # 슬라이스는 입력 순서대로 앞부분을 선택합니다. 기본값은 두 건이며, 한 건만 있으면 한 건이 반환됩니다.
    return hits[:max_items]
    # END D26-02-I02

def make_context(hits: list[dict], max_items: int = 2) -> tuple[str, list[int]]:
    """선택한 목록에서 Context와 포함 ID를 동시에 만듭니다."""
    # BEGIN D26-02-I03
    # TODO D26-02-I03: 선택 결과를 문자열로 연결하고 같은 Hit의 citation_id 목록을 반환합니다.
    # 힌트: 본문과 ID 수집을 같은 for 반복문 안에 둡니다.
    # 함수의 docstring·README·expected/EXPECTED.md에서 입력과 완료 조건을 확인하세요.
    # 구현 후 아래 NotImplementedError를 제거하세요. 정답은 이 아래에 숨겨져 있지 않습니다.

    # 먼저 사용할 Hit를 확정한 뒤, 각 Hit의 텍스트 블록과 인용 ID를 같은 순서로 모읍니다.
    # 같은 반복에서 두 값을 추가하면 Context에 들어간 근거와 included_ids를 한 건씩 대조할 수 있습니다.
    selected = select_hits(hits, max_items)
    blocks = []
    included_ids = []
    for hit in selected:
        blocks.append(format_hit(hit))
        included_ids.append(hit["citation_id"])
    # 블록 사이에는 빈 줄 하나를 둡니다. 선택한 Hit가 없으면 빈 문자열과 빈 ID 목록을 반환합니다.
    return "\n\n".join(blocks), included_ids
    # END D26-02-I03

def main():
    # 공통 예시 데이터의 첫 번째 Hit인 출석 문서로 Context 생성 과정을 확인합니다.
    hit = example_hit()
    context, included_ids = make_context([hit])
    report = {
        # 실행 정보에는 로컬 Python 모드와 DB·임베딩·LLM 실행 여부(False)가 기록됩니다.
        "execution": execution_record("local_python"),
        "context": context,
        "included_ids": included_ids,
        "input_source": "lecture_fixture",
        # len()은 줄바꿈을 포함해 Context 문자열의 문자 수를 셉니다.
        "character_count": len(context),
    }
    # 현재 실습 폴더의 outputs/hit_context.json에 저장하고 터미널에 저장 경로를 표시합니다.
    path = save_json(HERE, "hit_context.json", report)
    show_success("02 응용", path)

# tasks.py를 직접 실행하면 main.py와 같은 main()을 호출합니다.
if __name__ == "__main__":
    main()
