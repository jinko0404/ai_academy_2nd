from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, joinedload
from app.db import get_db, transaction
from app.models import Category, Knowledge
from app.schemas import KnowledgeRead, KnowledgeWrite

app = FastAPI(title="Knowledge API · PostgreSQL")

def to_read(item: Knowledge) -> KnowledgeRead:
    return KnowledgeRead(id=item.id, title=item.title, content=item.content, category=item.category.name)

def load_one(db: Session, knowledge_id: int) -> Knowledge | None:
    return db.scalar(select(Knowledge).options(joinedload(Knowledge.category)).where(Knowledge.id == knowledge_id))

def get_or_create_category(db: Session, name: str) -> Category:
    category = db.scalar(select(Category).where(Category.name == name))
    if category is None:
        category = Category(name=name)
        db.add(category)
        db.flush()  # 같은 Transaction 안에서 category.id를 확보합니다.
    return category

def raise_conflict(exc: IntegrityError) -> None:
    # HTTP 변환은 transaction 바깥에서 수행합니다. transaction은 먼저 rollback하고 예외를 다시 전달합니다.
    raise HTTPException(status_code=409, detail="Database constraint conflict") from exc

@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "reachable"}
    except OperationalError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

# GET /api/knowledge 요청을 아래 함수에 연결합니다.
# response_model은 응답을 KnowledgeRead 형식의 목록으로 검증·직렬화하는 설정입니다.
@app.get("/api/knowledge", response_model=list[KnowledgeRead])
def list_knowledge(category: str | None = Query(default=None), db: Session = Depends(get_db)):
    # TODO B03: Knowledge와 Category를 함께 읽어 id 순으로 목록을 반환하세요.
    # category가 오면 분류를 필터합니다.
    # [함수의 매개변수]
    # category: URL의 ?category=learning 같은 쿼리 매개변수를 받습니다.
    # str | None은 문자열 또는 None을 허용한다는 뜻입니다.
    # Query(default=None)이므로 category를 생략하면 None이 들어옵니다.
    # db: FastAPI가 get_db()의 yield로 전달받은 이번 요청용 Session입니다.

    # [1. 조회문 구성] stmt는 statement(구문)의 줄임말로, 여기서는 조회문 객체입니다.
    # 아직 DB에 요청을 보내지는 않습니다. 어떤 데이터를 가져올지만 정합니다.
    stmt = (
        # Knowledge 모델에 대응하는 knowledge 테이블의 데이터를 조회합니다.
        select(Knowledge)
        # Knowledge.category 관계를 따라 연결된 Category 객체도 JOIN으로 함께 읽습니다.
        # 나중에 item.category.name에 접근할 때 분류를 추가로 조회하지 않도록 합니다.
        .options(joinedload(Knowledge.category))
        # 지식 ID를 기준으로 오름차순 정렬합니다. SQL의 ORDER BY knowledge.id와 같습니다.
        .order_by(Knowledge.id)
    )

    # [2. 선택적 필터] category를 전달한 요청에만 WHERE 조건을 추가합니다.
    # /api/knowledge → 전체 목록
    # /api/knowledge?category=learning → 분류명이 learning인 목록
    # 빈 문자열도 None과는 다르므로 ?category= 요청에는 빈 문자열 조건이 적용됩니다.
    if category is not None:
        # join: 모델에 선언된 관계를 이용해 knowledge.category_id = categories.id로 연결합니다.
        # where: 연결된 categories 테이블의 name이 입력값과 같은 행만 선택합니다.
        # 위 joinedload는 '분류 객체를 함께 가져오기', 이 join은 '분류로 필터하기'가 목적입니다.
        # 두 JOIN은 역할이 다르며, joinedload의 JOIN을 이 필터가 그대로 재사용하지는 않습니다.
        # 조건을 추가한 새 조회문을 stmt에 다시 넣습니다. 여기서도 아직 실행하지 않습니다.
        stmt = stmt.join(Knowledge.category).where(Category.name == category)

    # [3. 조회 실행] 여기서 Session을 통해 SQL을 DB에 보내고 결과를 가져옵니다.
    # scalars(stmt): 각 결과 행에서 선택한 Knowledge 객체를 꺼내는 결과 객체를 받습니다.
    #                여기서는 scalars가 정수·문자열이 아니라 Knowledge 객체를 반환합니다.
    # unique(): 같은 ORM 객체가 결과에 반복되면 중복을 제거합니다.
    #           현재처럼 지식 하나에 분류 하나가 연결된 조회에서는 필수는 아닙니다.
    # all(): 남은 결과를 모두 읽어 Python 리스트로 만듭니다. 결과가 없으면 []입니다.
    items = db.scalars(stmt).unique().all()

    # [4. 응답 변환] 리스트의 각 Knowledge 객체를 to_read()에 하나씩 전달합니다.
    # to_read()는 id/title/content와 item.category.name으로 KnowledgeRead를 만듭니다.
    # DB의 category_id 숫자 대신 분류 이름 문자열을 API 사용자에게 전달하는 것입니다.
    # 아래 리스트 컴프리헨션은 다음 반복문과 같은 뜻입니다.
    # results = []
    # for item in items:
    #     results.append(to_read(item))
    # return results
    # 반환한 목록은 FastAPI가 response_model에 맞춰 처리하여 JSON 배열로 응답합니다.
    return [to_read(item) for item in items]

@app.get("/api/knowledge/{knowledge_id}", response_model=KnowledgeRead)
def read_knowledge(knowledge_id: int, db: Session = Depends(get_db)):
    item = load_one(db, knowledge_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    return to_read(item)

@app.post("/api/knowledge", response_model=KnowledgeRead, status_code=status.HTTP_201_CREATED)
def create_knowledge(payload: KnowledgeWrite, db: Session = Depends(get_db)):
    try:
        with transaction(db):
            category = get_or_create_category(db, payload.category)
            item = Knowledge(title=payload.title, content=payload.content, category_id=category.id, category=category)
            db.add(item)
            db.flush()  # DB가 발급한 id를 commit 전에 확인합니다.
            result = to_read(item)
        return result
    except IntegrityError as exc:
        raise_conflict(exc)

@app.put("/api/knowledge/{knowledge_id}", response_model=KnowledgeRead)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite, db: Session = Depends(get_db)):
    item = load_one(db, knowledge_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    try:
        with transaction(db):
            category = get_or_create_category(db, payload.category)
            item.title, item.content, item.category_id = payload.title, payload.content, category.id
            item.category = category
            db.flush()
            result = to_read(item)
        return result
    except IntegrityError as exc:
        raise_conflict(exc)

@app.delete("/api/knowledge/{knowledge_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge(knowledge_id: int, db: Session = Depends(get_db)) -> Response:
    item = load_one(db, knowledge_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    try:
        with transaction(db):
            db.delete(item)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except IntegrityError as exc:
        raise_conflict(exc)
