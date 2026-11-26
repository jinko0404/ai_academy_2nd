"""05 · PostgreSQL Read-only lookup_status · 기본필수.

실습 목적
    보고서에 저장된 인덱싱 상태(index_status)와 현재 내용에 맞는 벡터가 있는지(vector_fresh)를 확인합니다.
    예를 들어 ready 상태라도 보고서의 현재 버전과 벡터를 만든 버전이 다르면 vector_fresh는 False입니다.

전체 흐름
    입력값을 보고서 ID로 변환한 뒤 main.py에서 기존 report_agent_lab을 SELECT로 조회합니다.
    status_view()는 DB 응답에서 상태 확인에 필요한 값만 꺼내 출력용 딕셔너리로 만듭니다.
"""

def parse_report_id(value) -> int:
    """조회할 보고서의 ID(reports.report_id)를 양의 정수로 변환합니다."""
    # TODO D29-05-B-01: 입력이 보고서 ID 형식에 맞는지 검증합니다.
    from coursekit.contracts import report_id_from
    # 예를 들어 '63'은 정수 63으로 변환하고, 0이나 숫자가 아닌 문자열은 ValueError로 처리합니다.
    return report_id_from(value)

def status_view(raw: dict) -> dict:
    """DB 응답을 받아 화면에 표시할 보고서 상태를 정리합니다."""
    # TODO D29-05-B-02: pending/ready/failed와 실제 벡터 정합성을 구분해 표시.
    # raw['status']는 조회 성공 여부입니다. 보고서를 찾지 못했으면 상태 코드와 요청한 ID를 반환합니다.
    if raw['status']!='ok':return {'status':'not_found','report_id':raw['report_id']}
    # index_status는 인덱싱 진행 상태(pending 대기, ready 완료, failed 실패)입니다.
    # revision은 현재 보고서 버전입니다. vector_fresh는 ready 상태에서 벡터가 존재하고,
    # 벡터를 만든 버전·임베딩 모델·개요 해시가 현재 기준과 모두 맞을 때 True입니다.
    return {'report_id':raw['report_id'],'index_status':raw['index_status'],
            'revision':raw['revision'],'vector_fresh':raw['vector_fresh']}


