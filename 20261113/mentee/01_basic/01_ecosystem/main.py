# 이 파일의 전체 흐름:
# 1. 구성요소의 역할과 연결 정보를 파이썬 데이터로 작성한다.
# 2. 연결 정보에서 이름을 모으고, 그림에 쓸 번호를 붙인다.
# 3. 그림을 그리는 Mermaid 문장들을 만든다.
# 4. Markdown 파일과 JSON 파일로 저장한다.

# Path: 파일이나 폴더의 위치(경로)를 다루는 도구.
from pathlib import Path
# sys: 파이썬 실행 환경을 다루는 도구. 여기서는 모듈을 찾을 폴더를 추가한다.
import sys

# __file__은 현재 main.py의 경로이다.
# resolve()는 절대 경로로 바꾸고, parent는 그 파일이 들어 있는 폴더를 뜻한다.
HERE = Path(__file__).resolve().parent
# HERE의 상위 폴더인 01_basic 폴더. 이 안에 common 폴더가 있다.
LEVEL_ROOT = HERE.parent
# sys.path는 import할 모듈을 찾는 폴더 목록이다.
# insert(0, ...)은 목록 맨 앞에 추가한다. str()은 경로를 문자열로 바꾼다.
sys.path.insert(0, str(LEVEL_ROOT))  # 어느 작업 폴더에서 실행해도 common 패키지를 찾습니다.
# 공통 파일에 정의된 저장 함수 두 개를 가져온다.
# write_json은 데이터를 JSON으로, write_md는 문자열을 Markdown 파일로 저장한다.
from common.support import write_json, write_md

# 강의 PPT 15장: 개념도에 실제 구현한 것과 향후 구현할 것을 구분합니다.
# LLM을 실제 호출하지 않습니다. diagram.md는 그림을 그리는 Mermaid 텍스트입니다.
# def는 함수를 정의하는 문법이다. 아래 들여쓴 코드들을 main이라는 이름으로 묶는다.
# 함수는 정의만으로 실행되지 않고, 맨 아래의 main()에서 호출할 때 실행된다.
def main():

    # TODO D20-01-B01: FastAPI·PostgreSQL·LLM·응답의 역할 네 가지를 짧은 문장으로 작성하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-B01
    # 딕셔너리 {키: 값}: 이름으로 값을 찾아볼 수 있는 자료형이다.
    # 예: roles["FastAPI"]를 읽으면 FastAPI에 해당하는 설명 문자열이 나온다.
    roles = {  # 네 구성요소의 역할을 한 문장씩 채워 주세요.
        "FastAPI": "HTTP 요청을 검증하고 함수·응답을 연결한다",       # 힌트: HTTP 요청과 응답을 연결
        "PostgreSQL": "원본 지식·분류를 관계형 테이블에 저장한다",  # 힌트: 원본 지식 데이터를 보관
        "LLM": "문맥과 입력을 바탕으로 다음 Token을 생성한다",           # 힌트: 다음 Token 계산
        "Response": "호출한 Client에 JSON 결과를 전달한다",      # 힌트: Client에게 결과 반환
    }
    # END D20-01-B01

    # TODO D20-01-B02: 요청→API→데이터·LLM→응답 연결을 올바른 순서의 화살표로 표현하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-B02
    # edges는 연결 정보 여러 개를 담은 리스트 [ ... ]이다.
    # 리스트 안의 각 튜플 ( ... )에는 값 세 개가 한 묶음으로 들어 있다.
    # 예: ("Client", "FastAPI", "HTTP 요청")은
    #     Client에서 FastAPI로 향하는 화살표에 "HTTP 요청"이라고 적겠다는 뜻이다.
    # 이 데이터 자체가 HTTP 요청을 보내거나 DB/LLM을 실행하는 것은 아니다.
    edges = [
        ("Client", "FastAPI", "HTTP 요청"),
        ("FastAPI", "PostgreSQL", "원본 지식 조회·수정"),
        ("PostgreSQL", "FastAPI", "원본 데이터"),
        ("FastAPI", "LLM", "입력·문맥(후속 구현)"),
        ("LLM", "Response", "생성 결과(후속 구현)"),
        ("Response", "Client" , "JSON")
    ]  # 힌트: (출발점, 도착점, 화살표 설명) 순서로 추가
    # 예: edges.append(("Client", "FastAPI", "HTTP 요청"))
    # TODO: API→DB, API→LLM(후속), LLM→응답(후속) 등의 연결 추가

    # END D20-01-B02
    # 저장과 생성의 담당을 구분한 설명이다. 아래에서 JSON에 함께 담는다.
    responsibility = {"storage":"DB는 원본 저장", "generation":"LLM은 언어 생성"}
    # 이 그림이 설명하는 범위를 기록한다.
    trust_boundary = {"scope":"후속 구현 개념도"}

    # TODO D20-01-B03: 완성한 연결 관계를 Mermaid 그림과 JSON 요약으로 저장하세요.
    # 입력·대상: FastAPI/DB/LLM의 역할과 실제 구현 범위를 혼동하지 않습니다.
    # 완료 확인: outputs/architecture.md 와 architecture.json을 확인합니다.
    # 주의: 그림의 RAG·Agent는 후속 수업의 개념이며 오늘 동작하지 않습니다.
    # BEGIN D20-01-B03
    # 노드(node)는 그림의 상자 하나, 엣지(edge)는 상자 사이의 연결 하나를 뜻한다.
    # 아래 한 줄은 "이름을 하나씩 꺼내기 → 중복 제거 → 정렬"을 수행한다.
    #
    # for a,b,_ in edges: 연결 정보 한 묶음을 꺼내 세 변수에 나눠 넣는다.
    # 첫 묶음에서는 a="Client", b="FastAPI", _="HTTP 요청"이 된다.
    # _도 변수이지만, 여기서는 사용하지 않을 값에 붙인 이름이다.
    # for n in (a,b): a와 b를 차례대로 꺼낸다. n에는 이름 문자열 하나씩 들어간다.
    # {n ...}: 꺼낸 이름을 집합(set)에 모은다. 같은 이름은 한 번만 남는다.
    # sorted(...): 집합의 이름들을 정렬해서 새로운 리스트로 돌려준다.
    # 즉, n에 리스트를 넣는 것이 아니라 n의 값들을 모아 리스트를 만드는 것이다.
    #
    # 풀어 쓰면 다음과 같다. 아래 주석 코드는 설명용이라 실행되지 않는다.
    # names = set()
    # for a, b, _ in edges:
    #     for n in (a, b):
    #         names.add(n)
    # node_names = sorted(names)
    #
    # 현재 데이터의 결과:
    # ["Client", "FastAPI", "LLM", "PostgreSQL", "Response"]
    node_names = sorted({n for a,b,_ in edges for n in (a,b)})  # 힌트: edges에서 시작·도착 노드 이름을 모으세요.

    # 각 이름에 그림에서 사용할 ID를 붙이는 딕셔너리 컴프리헨션이다.
    # enumerate(node_names)는 (0, "Client"), (1, "FastAPI"), ...를 차례로 준다.
    # i에는 번호, name에는 이름이 들어간다.
    # f"N{i}"는 i의 값을 문자열 안에 넣는다. i가 0이면 "N0"이 된다.
    # 결과: {"Client": "N0", "FastAPI": "N1", "LLM": "N2", ...}
    # 풀어 쓰면:
    # node_id = {}
    # for i, name in enumerate(node_names):
    #     node_id[name] = f"N{i}"
    node_id = {name:f"N{i}" for i,name in enumerate(node_names)}              # 힌트: {노드이름: "N0"} 형태의 ID 표

    # lines는 파일에 쓸 문장들을 한 줄씩 담는 리스트이다.
    # flowchart LR은 Mermaid에서 "왼쪽(Left)에서 오른쪽(Right)으로 그려라"는 뜻이다.
    lines = ["flowchart LR"] # Mermaid 그래프 시작 선언

    # [표현식 for ...]은 반복해서 만든 결과들을 리스트에 모으는 문법이다.
    # name이 "Client"일 때 node_id[name]은 "N0"이므로
    # f문자열의 결과는 '  N0["Client"]'가 된다. 이는 상자를 만드는 Mermaid 문장이다.
    # lines += [...]는 만든 문장들을 기존 lines 리스트 뒤에 하나씩 덧붙인다.
    # 풀어 쓰면:
    # for name in node_names:
    #     lines.append(f'  {node_id[name]}["{name}"]')
    lines += [f'  {node_id[name]}["{name}"]' for name in node_names]               # TODO: 노드 생성 줄을 추가하세요.

    # 이번에는 각 연결의 출발점 a, 도착점 b, 설명 label을 모두 사용한다.
    # 첫 연결로 만드는 문장: '  N0 -->|"HTTP 요청"| N1'
    # Mermaid가 이 문장을 읽으면 Client → FastAPI 화살표와 설명을 그린다.
    # 풀어 쓰면:
    # for a, b, label in edges:
    #     lines.append(f'  {node_id[a]} -->|"{label}"| {node_id[b]}')
    lines += [f'  {node_id[a]} -->|"{label}"| {node_id[b]}' for a,b,label in edges]               # TODO: 연결 화살표 줄을 추가하세요.

    # Path 객체의 /는 여기서 나눗셈이 아니라 폴더와 파일 경로를 이어 붙이는 연산이다.
    # 저장 위치는 이 main.py와 같은 폴더 아래의 outputs/architecture.md이다.
    # \n은 줄바꿈 문자이고, "\n".join(lines)는 문장 사이에 줄바꿈을 넣어
    # 리스트 안의 여러 문자열을 긴 문자열 하나로 합친다.
    # 문자열 사이의 +는 문자열을 이어 붙인다.
    # 최종 내용은 제목 + Mermaid 코드 블록 시작 + 그림 문장들 + 코드 블록 끝이다.
    # ```mermaid로 감싸 두면 Mermaid를 지원하는 뷰어에서 그림으로 볼 수 있다.
    # write_md가 outputs 폴더를 필요하면 만들고, 파일을 UTF-8로 저장한다.
    write_md(HERE/"outputs"/"architecture.md", "# AI 지식 플랫폼 개념도\n\n```mermaid\n"+"\n".join(lines)+"\n```\n")  # TODO: 제목과 Mermaid 코드 문자열

    # 같은 정보를 프로그램이 읽기 편한 JSON 파일로도 저장한다.
    # 두 번째 인자인 { ... }는 저장할 데이터를 모은 딕셔너리이다.
    # "roles": roles에서 왼쪽은 JSON에 쓸 항목 이름, 오른쪽은 위에서 만든 변수이다.
    # [list(x) for x in edges]는 각 연결 튜플 x를 리스트로 바꾸어 모은다.
    # 예: ("Client", "FastAPI", "HTTP 요청") → ["Client", "FastAPI", "HTTP 요청"]
    # write_json은 이 딕셔너리를 JSON 문자열로 변환해 UTF-8 파일로 저장한다.
    write_json(HERE/"outputs"/"architecture.json", {
            "roles":roles,"edges":[list(x) for x in edges],
            "responsibility":responsibility,"trust_boundary":trust_boundary,
        })  # TODO: roles·edges 등 요약
    # END D20-01-B03
    # print는 파일 저장이 아니라 터미널 화면에 완료 메시지를 출력한다.
    print("[01] architecture.md / architecture.json 생성 완료")

# 이 파일을 python main.py처럼 직접 실행하면 __name__은 "__main__"이 된다.
# 다른 파일에서 import했을 때는 아래 main()을 자동 실행하지 않는다.
if __name__=="__main__":
    # 위에서 정의한 main 함수 안의 코드들을 실행한다.
    main()
