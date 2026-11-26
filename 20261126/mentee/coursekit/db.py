"""기존 report_agent_lab 데이터에 대한 SELECT 전용 Repository.

인증/소유권 테이블은 실제 Schema에 없습니다. author는 로그인 ID가 아닙니다.
"""
from functools import lru_cache
from contextlib import contextmanager
from coursekit.contracts import report_id_from
from coursekit.settings import settings

@lru_cache(maxsize=1)
def engine():
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    url=make_url(settings().database_url)
    if url.database!='report_agent_lab':
        raise RuntimeError(f'잘못된 DB 대상: {url.database!r}. 반드시 기존 report_agent_lab을 사용하세요.')
    if url.get_backend_name()!='postgresql':
        raise RuntimeError('PostgreSQL DB 연결 문자열만 지원합니다.')
    return create_engine(settings().database_url, pool_pre_ping=True, pool_size=3, max_overflow=2)

@contextmanager
def readonly_connection():
    """연결 내 트랜잭션을 READ ONLY로 지정; DDL/DML 없음."""
    from sqlalchemy import text
    with engine().connect() as conn:
        with conn.begin():
            conn.execute(text('SET TRANSACTION READ ONLY'))
            yield conn


def preflight()->dict:
    from sqlalchemy import text
    with readonly_connection() as db:
        state=db.execute(text('''SELECT current_database() db,
          EXISTS(SELECT 1 FROM pg_extension WHERE extname='vector') vector_active,
          (SELECT count(*) FROM reports) reports,
          (SELECT count(*) FROM report_embeddings) vectors,
          (SELECT count(*) FROM reports WHERE index_status='ready') ready,
          (SELECT count(*) FROM report_embeddings WHERE model_name=:model) matching_model_vectors,
          (SELECT count(*) FROM report_embeddings e JOIN reports r USING(report_id)
           WHERE r.index_status='ready' AND e.model_name=:model
             AND e.indexed_revision=r.revision
             AND e.overview_hash=encode(sha256(convert_to(r.overview,'UTF8')),'hex')) fresh_vectors,
          EXISTS(SELECT 1 FROM information_schema.views WHERE table_schema='public' AND table_name='report_read_view') view_available'''),{'model':settings().embedding_model}).mappings().one()
    if state['db']!='report_agent_lab': raise RuntimeError('현재 접속한 DB가 report_agent_lab이 아닙니다.')
    return dict(state)


def fetch_report_status(report_id: int)->dict:
    from sqlalchemy import text
    rid=report_id_from(report_id)
    with readonly_connection() as db:
        row=db.execute(text('''SELECT r.report_id, c.code AS category_code,
            r.index_status, r.revision, e.indexed_revision, e.model_name,
            e.indexed_at,
            CASE WHEN r.index_status='ready' AND e.report_id IS NOT NULL
             AND e.indexed_revision=r.revision
             AND e.model_name=:model
             AND e.overview_hash=encode(sha256(convert_to(r.overview,'UTF8')),'hex')
             THEN true ELSE false END AS vector_fresh
          FROM reports r
          JOIN report_categories c ON c.category_id=r.category_id
          LEFT JOIN report_embeddings e ON e.report_id=r.report_id
          WHERE r.report_id=:rid'''),{'rid':rid,'model':settings().embedding_model}).mappings().first()
    if row is None:return {'status':'not_found','report_id':rid,'source':'reports'}
    return {'status':'ok','report_id':rid,'category_code':row['category_code'],
            'index_status':row['index_status'],'revision':int(row['revision']),
            'vector_fresh':bool(row['vector_fresh']),
            'indexed_revision':row['indexed_revision'],'embedding_model':row['model_name']}


def fetch_report_read_view(report_id:int)->dict|None:
    from sqlalchemy import text
    rid=report_id_from(report_id)
    with readonly_connection() as db:
        row=db.execute(text('''SELECT report_id,category_code,title,overview,content,author,revision,index_status
                               FROM report_read_view WHERE report_id=:rid'''),{'rid':rid}).mappings().first()
    return dict(row) if row is not None else None


def select_semantic_candidates(query_vector:list[float], category:str, k:int)->list[dict]:
    """11/20의 overview E5 Vector(384)만 사용합니다. 개요를 임의 재색인하지 않습니다."""
    from sqlalchemy import text
    if len(query_vector)!=384:
        raise ValueError('Embedding 차원은 schema.sql에 명시된 vector(384)여야 합니다.')
    if not 1<=k<=20:raise ValueError('candidate_k는 1~20이어야 합니다.')
    # SQLAlchemy + psycopg에 바인딩할 벡터 텍스트입니다. 문자열 연결 SQL이 아닙니다.
    vector = '['+','.join(f'{float(v):.9g}' for v in query_vector)+']'
    # category='ALL'도 서버에서 허용한 Category만 조회합니다. 사용자별 AuthZ는 별도 과제입니다.
    where_category = "AND v.category_code=:category" if category!='ALL' else ''
    sql=f'''SELECT v.report_id, v.title, v.overview, v.content, v.category_code, v.revision,
            (e.embedding <=> CAST(:query_vector AS vector(384))) AS cosine_distance
        FROM report_read_view v
        JOIN report_embeddings e ON e.report_id=v.report_id
        WHERE v.index_status='ready'
          AND e.indexed_revision=v.revision
          AND e.model_name=:embedding_model
          AND e.overview_hash=encode(sha256(convert_to(v.overview,'UTF8')),'hex')
          AND v.category_code IN :allowed_categories
          {where_category}
        ORDER BY e.embedding <=> CAST(:query_vector AS vector(384)), v.report_id
        LIMIT :candidate_k'''
    from sqlalchemy import bindparam
    params={'query_vector':vector,'embedding_model':settings().embedding_model,
            'candidate_k':k,'category':category,
            'allowed_categories':settings().allowed_categories}
    statement=text(sql).bindparams(bindparam('allowed_categories',expanding=True))
    with readonly_connection() as db:
        rows=db.execute(statement,params).mappings().all()
    return [dict(row) for row in rows]
