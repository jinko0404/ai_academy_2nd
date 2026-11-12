from functools import lru_cache
from contextlib import contextmanager
from collections.abc import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from app.settings import get_settings

@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    if settings.db_password == "CHANGE_ME":
        raise RuntimeError(".env의 DB_PASSWORD를 실제 학습용 비밀번호로 바꾸세요.")
    return create_engine(settings.database_url, pool_pre_ping=True)

@lru_cache
def get_session_factory():
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)

def get_db() -> Generator[Session, None, None]:
    # TODO B02: 요청마다 Session을 열어 yield하고 요청이 끝나면 닫히게 하세요.
    # Session은 DB에 조회·저장 작업을 요청할 때 사용하는 객체입니다.
    # 1. Session을 만드는 도구(factory)를 가져옵니다. 아직 Session을 만든 것은 아닙니다.
    factory = get_session_factory()

    # 2. factory()로 이번 요청에서 사용할 Session을 하나 만듭니다.
    #    with를 사용하면 이 블록을 벗어날 때 Session이 자동으로 닫힙니다.
    #    Session 생성만으로 바로 DB에 접속하는 것은 아니며, 필요할 때 연결을 사용합니다.
    with factory() as session:
        # 3. FastAPI가 이 Session을 Depends(get_db)를 사용하는 API 함수에 전달합니다.
        #    예: def list_knowledge(db: Session = Depends(get_db))의 db가 됩니다.
        #    yield는 값을 전달하고 이 함수의 실행을 잠시 멈춥니다.
        #    그동안 API 함수가 전달받은 Session으로 DB 작업을 수행합니다.
        yield session

    # 4. FastAPI가 요청 처리를 마무리하면서 이 함수를 이어 실행하여 정리합니다.
    #    API 처리 중 오류가 나도 with의 정리 과정에서 Session.close()가 호출됩니다.
    #    정상 종료 시에는 위 with 블록을 벗어난 뒤 이 위치까지 내려옵니다.
    #    닫기는 자원 정리이며, 변경을 확정하는 commit은 transaction()에서 따로 합니다.


@contextmanager
def transaction(db: Session):
    """한 쓰기 요청의 flush와 commit을 같은 경계로 묶습니다."""
    try:
        yield
        db.commit()
    except Exception:
        db.rollback()
        raise

def check_database() -> tuple[str, str]:
    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT current_database(), current_user")).one()
        return str(row[0]), str(row[1])
