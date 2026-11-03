import os
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from app.core.database import Database
from app.core.http import Document, ingest, instrument
from app.core.model import ModelError
from app.core.retrieval import Knowledge, configured_encoder
from app.core.security import Principal, authenticate
from app.schemas import Comparison, Query
from app.service import Procurement


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = Database(os.getenv("DATA_PATH", "data/vendor.sqlite3"))
    app.state.knowledge = Knowledge(app.state.db, configured_encoder())
    app.state.service = Procurement(app.state.knowledge)
    yield


app = FastAPI(title="Vendor Evidence RAG", lifespan=lifespan)
instrument(app)
Identity = Annotated[Principal, Depends(authenticate)]
Key = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=100)]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "vendor-evidence-rag"}


@app.post("/documents")
def documents(body: Document, request: Request, principal: Identity) -> dict:
    return ingest(request.app, principal, body)


@app.post("/comparisons", status_code=201)
def compare(body: Comparison, request: Request, principal: Identity, key: Key) -> dict:
    try:
        return request.app.state.service.compare(principal, body, key)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    except ModelError:
        raise HTTPException(503, "Local model unavailable or invalid output") from None


@app.get("/comparisons/{report_id}")
def report(report_id: UUID, request: Request, principal: Identity) -> dict:
    import json

    with request.app.state.db.connect() as db:
        row = db.execute(
            "SELECT result,creator FROM reports WHERE tenant=? AND id=?",
            (principal.tenant, str(report_id)),
        ).fetchone()
        if not row or (row["creator"] != principal.subject and principal.role != "reviewer"):
            raise HTTPException(404, "Comparison not found")
        return json.loads(row["result"])


@app.post("/query")
def query(body: Query, request: Request, principal: Identity) -> dict:
    try:
        return request.app.state.service.query(principal, body)
    except ModelError:
        raise HTTPException(503, "Local model unavailable or invalid output") from None
