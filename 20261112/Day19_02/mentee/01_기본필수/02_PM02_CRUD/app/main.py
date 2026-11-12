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
    # TODO B04: 분류명을 조회하고 없으면 Category를 add→flush하여 같은 Transaction에서 id를 확보하세요.
    # 이름이 같은 분류가 있는지 조회
    category = db.scalar(
        select(Category).where(Category.name == name)
    )

    if category is None:
        category = Category(name=name)  # 새 분류 객체 생성
        db.add(category)               # 저장할 대상으로 Session에 등록
        db.flush()                    # INSERT 실행 → 생성된 ID가 category.id에 들어옴

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

@app.get("/api/knowledge", response_model=list[KnowledgeRead])
def list_knowledge(category: str | None = Query(default=None), db: Session = Depends(get_db)):
    stmt = select(Knowledge).options(joinedload(Knowledge.category)).order_by(Knowledge.id)
    if category is not None:
        stmt = stmt.join(Knowledge.category).where(Category.name == category)
    return [to_read(x) for x in db.scalars(stmt).unique().all()]

@app.get("/api/knowledge/{knowledge_id}", response_model=KnowledgeRead)
def read_knowledge(knowledge_id: int, db: Session = Depends(get_db)):
    # TODO B05: PK로 조회하고 없으면 404, 있으면 KnowledgeRead 형식으로 반환하세요.
    # 요청받은 ID로 지식 조회
    item = load_one(db, knowledge_id)

    # 해당 ID의 지식이 없으면 404 응답
    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Knowledge not found",
        ) 

    return to_read(item)

@app.post("/api/knowledge", response_model=KnowledgeRead, status_code=status.HTTP_201_CREATED)
def create_knowledge(payload: KnowledgeWrite, db: Session = Depends(get_db)):
    # TODO B06: 분류를 연결하고 Knowledge를 저장한 뒤 commit하고 생성된 id를 포함해 반환하세요.
    # payload: 사용자가 보낸 JSON을 KnowledgeWrite 형식으로 검증한 객체입니다.
    #          payload.title/content/category로 입력값에 접근합니다.
    # db: get_db()가 이번 요청에 제공한 Session입니다. DB 조회·저장에 사용합니다.
    # 위 데코레이터는 POST 요청을 이 함수에 연결하고, 성공 시 201 Created를 응답하게 합니다.
    # response_model은 응답 데이터의 형식을 KnowledgeRead로 지정합니다.

    # 1. 분류 생성과 지식 저장을 하나의 트랜잭션(함께 확정하거나 취소할 작업 묶음)으로 처리합니다.
    #    transaction()은 블록이 정상 종료되면 commit(), 오류가 나면 rollback() 후 예외를 전달합니다.
    with transaction(db):
        # 2. 입력받은 분류명으로 Category를 조회하고, 없으면 새로 만듭니다.
        #    예: payload.category가 "python"이면 이름이 python인 분류 객체를 가져옵니다.
        #    새 분류일 때는 이 함수 내부의 flush()가 INSERT를 실행하여 category.id를 확보합니다.
        #    category.id는 '분류 번호'이며, 새 지식의 item.id와는 다른 번호입니다.
        category = get_or_create_category(db=db, name=payload.category)

        # 3. DB에 저장할 Knowledge 객체를 만듭니다. 생성만으로 INSERT되지는 않습니다.
        item = Knowledge(
            title=payload.title,
            content=payload.content,
            # 실제 DB의 FK 열에 연결할 분류의 ID를 넣습니다.
            category_id=category.id,
            # ORM 관계 속성에는 Category 객체를 넣습니다.
            # 문자열인 payload.category를 넣는 자리가 아닙니다.
            # 이렇게 연결하면 item.category.name으로 분류 이름을 읽을 수 있습니다.
            category=category
        )

        # 4. 새 지식을 저장할 대상으로 Session에 등록합니다.
        db.add(item)

        # 5. 등록한 변경을 DB에 SQL로 반영합니다. 여기서 지식 INSERT가 실행됩니다.
        #    DB가 자동 생성한 기본키를 받아 item.id에 채워줍니다.
        #    flush는 저장 확정이 아니므로 이후 오류가 나면 이 INSERT도 rollback할 수 있습니다.
        db.flush()

        # 6. DB 객체를 응답용 KnowledgeRead로 변환합니다.
        #    생성된 item.id와 title/content, item.category.name이 응답에 들어갑니다.
        #    아직 반환하지 않고 결과를 준비합니다. 뒤의 commit이 성공해야 반환합니다.
        result = to_read(item=item)

    # 7. with 블록을 정상적으로 빠져나오면 transaction()의 commit이 완료된 상태입니다.
    #    commit을 포함해 중간에 오류가 발생하면 아래 return까지 도달하지 않습니다.
    #    FastAPI가 result를 JSON으로 변환하여 생성된 지식 정보와 201 상태 코드를 응답합니다.
    return result

@app.put("/api/knowledge/{knowledge_id}", response_model=KnowledgeRead)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite, db: Session = Depends(get_db)):
    # PUT /api/knowledge/2라면 knowledge_id에 2가 들어옵니다.
    # payload는 사용자가 보낸 수정 내용(title/content/category)을 검증한 객체입니다.
    # db는 이번 요청에서 조회·수정 작업에 사용하는 Session입니다.

    # 1. 기본키(id)로 기존 지식을 조회합니다. 연결된 분류도 함께 가져옵니다.
    #    이 item은 이미 Session이 관리하는 객체이므로 새로 db.add()할 필요가 없습니다.
    item = load_one(db, knowledge_id)
    # 해당 ID가 없으면 수정하지 않고 404 응답으로 처리를 끝냅니다.
    if item is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")

    # 2. 입력한 분류명을 조회하고, 없으면 분류를 생성하여 ID를 확보합니다.
    #    여기서 새 분류가 생성되어도 flush만 하므로 아직 commit된 것은 아닙니다.
    #    같은 Session에서 시작된 트랜잭션을 아래 transaction(db)가 commit합니다.
    #    다만 이 줄은 아래 with의 오류 처리 범위 밖입니다. 분류 생성 중 오류도
    #    transaction()의 rollback으로 처리하려면 이 줄을 with 안의 첫 줄로 옮기면 됩니다.
    category = get_or_create_category(db, payload.category)
    # TODO B07: 기존 id는 유지하고 title·content·category를 교체한 뒤 commit하세요.

    # 3. 아래 수정이 성공하면 commit하고, 오류가 나면 rollback합니다.
    #    transaction()은 이 프로젝트에서 만든 함수이며 내부에서 commit/rollback을 호출합니다.
    with transaction(db):
        # 기존 객체의 속성을 바꿉니다. Knowledge(...)로 새 객체를 만드는 작업과 다릅니다.
        # Session이 속성 변경을 추적하여 저장 시 필요한 UPDATE를 만듭니다.
        item.title=payload.title
        item.content=payload.content
        # category는 연결된 Category 객체, category_id는 DB에 저장되는 분류 ID입니다.
        # 둘 다 새 분류를 가리키게 하여 객체 관계와 FK 값을 맞춥니다.
        item.category=category
        item.category_id=category.id

        # 4. 변경 사항을 UPDATE SQL로 DB에 반영합니다. 아직 최종 확정은 아닙니다.
        #    item.id는 변경하지 않았으므로 기존 지식의 ID가 그대로 유지됩니다.
        db.flush()

    # 5. with가 정상 종료되면 commit이 완료된 상태입니다.
    #    수정된 객체를 id/title/content/category 문자열이 담긴 응답 형식으로 바꿉니다.
    result = to_read(item)

    # response_model에 따라 JSON 응답을 만듭니다. 이 PUT의 기본 성공 상태 코드는 200입니다.
    return result

    

@app.delete("/api/knowledge/{knowledge_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge(knowledge_id: int, db: Session = Depends(get_db)) -> Response:
    # DELETE /api/knowledge/2라면 ID 2의 지식 하나를 삭제합니다.
    # 1. 삭제할 지식을 먼저 조회합니다. URL로 받은 ID를 조회 조건으로 사용합니다.
    item = load_one(db, knowledge_id)
    # 이미 없거나 존재하지 않는 ID라면 삭제 대신 404를 응답합니다.
    if item is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    # TODO B08: 대상 행을 삭제하고 commit한 뒤 Body 없는 204를 반환하세요.
    # 2. 삭제를 확정할 트랜잭션 범위입니다. 오류가 나면 rollback합니다.
    with transaction(db):
        # 이 객체를 삭제 대상으로 Session에 등록합니다.
        # 기본 설정에서 이 호출만으로 즉시 DELETE SQL이 실행되는 것은 아닙니다.
        # with 종료 시 transaction()이 commit()을 호출하고, commit의 자동 flush가
        # DELETE SQL을 실행한 뒤 삭제를 확정하므로 여기에는 별도 flush가 없어도 됩니다.
        # 현재 모델 설정에서는 연결된 Category를 함께 삭제하지 않습니다.
        db.delete(item)

    # 3. commit이 성공한 뒤 204 No Content로 응답합니다.
    #    204는 '성공했으며 응답 본문이 없음'을 뜻하므로 JSON이나 메시지를 담지 않습니다.
    return Response(status_code=204)
