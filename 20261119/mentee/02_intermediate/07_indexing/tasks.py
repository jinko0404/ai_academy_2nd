# 실습 목표 — 문서 청크를 E5 임베딩으로 변환해 pgvector에 저장하고, 질문과 가까운 청크를 원문 출처와 함께 찾습니다.
# 전체 흐름 — 문서 조회·파싱 → 정제·분할 → (--live-db) 벡터 계산 → 저장(Stage) → 공개(Publish) → 검색 → outputs/indexing_result.json 저장.
# 기본 실행은 청크를 준비하고 저장·검색 상태를 not_run으로 기록합니다. --live-db를 붙인 실행에서 실제 모델과 DB를 사용합니다.
"""실습 07 Intermediate · --live-db 필수: 실제 DB 결과를 확인할 때."""
from pathlib import Path
import sys
# 현재 실습 폴더를 기준으로 공통 coursekit을 가져오고 결과 파일을 저장합니다.
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import save_json,show_success
from coursekit.documents import by_doc,parse_one
from coursekit.processing import make_chunks


# 파서가 반환한 단위마다 정제·분할을 적용합니다. 문서와 청크를 함께 반환해 이후 저장에서 출처를 연결합니다.
def prepare(doc_id:str)->tuple[dict,list[dict]]:
    # BEGIN D24-07-I01
    # TODO D24-07-I01 : 여러 ParsedUnit을 정제·분할하고 입력 청크의 출처·버전 계약을 통일합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    # 문서 ID로 manifest에서 원문 정보를 찾고, 파싱을 통과한 결과의 본문 단위를 사용합니다.
    doc=by_doc(doc_id);parsed=parse_one(doc)
    if parsed['status']!='parsed':raise ValueError(parsed['status'])
    chunks=[]
    # ParsedUnit은 파서가 나눈 본문 단위입니다. 여기서는 각 단위를 따로 처리해 출처가 섞이지 않게 합니다.
    for unit in parsed['units']:
        # 정제 본문을 최대 110자씩 자르고 이웃 청크가 15자씩 겹치게 합니다. extend()는 만든 청크를 하나씩 목록에 붙입니다.
        chunks.extend(make_chunks(doc,unit,size=110,overlap=15))
    if not chunks:raise ValueError('empty chunk set')
    # 저장할 때 문서 정보와 청크가 함께 필요하므로 둘을 튜플로 반환합니다.
    return doc,chunks
    # END D24-07-I01
# 청크 하나에 벡터 한 개가 대응합니다. 각 벡터의 384개 값이 유한한 수인지도 확인합니다.
def embed(model,chunks:list[dict]):
    # BEGIN D24-07-I02
    # TODO D24-07-I02 : 실제 E5 Encoder로 384차원 passage 벡터를 만들고 개수·길이·유한값을 확인합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    from coursekit.live import encode_passages
    # 문서용 입력은 'passage: '가 붙은 embedding_text입니다. 청크 수만큼 벡터가 나오고 각 벡터에 384개 숫자가 들어가는지 공통 도구가 확인합니다.
    return encode_passages(model,chunks)
    # END D24-07-I02
# Stage는 청크와 벡터를 먼저 저장해 두는 단계입니다. Transaction은 DB 변경을 한 묶음으로 처리합니다.
def stage(conn,doc:dict,chunks:list[dict],vectors)->None:
    # BEGIN D24-07-I03
    # TODO D24-07-I03 : 짧은 DB Transaction 안에서 Chunk·벡터를 Staging 합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    from coursekit.db import stage_version
    with conn.transaction():
        # 각 청크와 벡터를 한 쌍씩 DB에 기록합니다. 이 저장 과정은 블록이 끝날 때 반영되고 오류가 나면 되돌아갑니다.
        stage_version(conn,doc,doc['file'],chunks,vectors)
    # END D24-07-I03
# Publish는 검색이 참고하는 현재 버전을 바꾸는 단계입니다. Top-k는 거리가 가까운 결과를 최대 k건 읽습니다.
def publish_search(conn,doc:dict,chunks:list[dict],model)->list[tuple]:
    # BEGIN D24-07-I04
    # TODO D24-07-I04 : 검증된 버전만 Publish하고 동일 계약의 query: 벡터로 Top-k를 조회합니다.
    # 요구: 입력 변수·반환값의 자료형을 유지하고, 완료 기준은 README/expected를 참조하세요.
    from coursekit.db import publish_version,search_current
    from coursekit.live import encode_query
    with conn.transaction():
        # 저장된 개수가 맞는지 확인한 뒤, 이 문서의 원문 버전과 처리 버전 쌍을 현재 검색 대상으로 지정합니다.
        publish_version(conn,doc['doc_id'],doc['source_version'],chunks[0]['pipeline_version'])
    # 질문에 'query: '를 붙여 문서와 같은 E5 모델로 질문 벡터를 만듭니다.
    qvec=encode_query(model,'과제 파일은 어디에 보관하나요?')
    # 현재 공개된 학사 청크에서 질문 벡터와 코사인 거리가 가까운 순서로 최대 3건을 가져옵니다. 거리가 작을수록 두 벡터가 가깝습니다.
    return search_current(conn,qvec,category='학사',k=3)
    # END D24-07-I04
def main():
    doc,chunks=prepare('DOC-ASSIGN')
    # 먼저 기본 결과를 채웁니다. DB 실행 옵션이 있을 때만 아래에서 저장 건수와 검색 결과를 채웁니다.
    result={'source_mode':'prepared_only_not_db','doc_id':doc['doc_id'],
            'chunk_count':len(chunks),'stored':'not_run','retrieved':'not_run'}
    # 기본 실행은 청크 정보와 not_run 상태를 남깁니다. 이 옵션을 넘기면 실제 모델과 DB를 사용합니다.
    if '--live-db' in sys.argv:
        from coursekit.live import load_encoder,connect_db
        from coursekit.db import initialize,use_schema
        # 벡터를 먼저 계산해 DB가 변경 작업을 기다리는 시간을 줄입니다.
        model=load_encoder(); vectors=embed(model,chunks)
        # DB 설정은 수강생 패키지의 루트인 mentee/.env에서 읽습니다.
        with connect_db() as conn:
            # 실습용 스키마 d24_inter07을 선택하고 필요한 테이블을 준비합니다. 이어서 저장, 공개, 검색 순서로 실행합니다.
            use_schema(conn,'d24_inter07')
            initialize(conn);stage(conn,doc,chunks,vectors)
            hits=publish_search(conn,doc,chunks,model)
            # DB에서 받은 튜플의 위치별 값을 이름 있는 JSON 필드에 옮깁니다.
            # citation_id는 DB가 부여한 인용 번호이고, source_locator는 검색된 본문을 원문에서 다시 찾기 위한 위치입니다.
            # stored는 이번에 준비해 저장한 청크 수입니다. retrieved는 검색으로 돌아온 행을 필드별로 정리한 목록입니다.
            result.update(source_mode='e5_pgvector_live',stored=len(chunks),
                          retrieved=[{'citation_id':r[0],'chunk_id':r[1],
                                      'body':r[2],'source_locator':r[3],'distance':r[6]} for r in hits])
    # 두 실행 방식 모두 같은 보고서 경로를 사용합니다. source_mode로 해당 실행의 처리 범위를 확인할 수 있습니다.
    out=save_json(HERE,'indexing_result.json',result)
    show_success('07 Intermediate',out)
