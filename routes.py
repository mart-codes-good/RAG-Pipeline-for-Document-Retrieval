import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app import rag_pipeline as rag
from app.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    EvalRequest,
    EvalResponse,
    DocumentInfo,
)

router = APIRouter()

# Tracks what's been ingested this session.
_ingested_documents: list[DocumentInfo] = []


@router.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": rag.get_reranker() is not None,
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF -> parse -> chunk -> build/update the index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    # Save the upload to a temp file so PyMuPDF can open it by path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        documents = rag.load_pdf_as_documents(tmp_path)
        if not documents:
            raise HTTPException(status_code=400, detail="No extractable text found in this PDF.")

        index, num_chunks = rag.chunk_and_build_index(documents)
        rag.set_active_index(index)

        doc_info = DocumentInfo(
            filename=file.filename,
            pages=len(documents),
            chunks=num_chunks,
        )
        _ingested_documents.append(doc_info)

        return IngestResponse(
            filename=file.filename,
            pages_loaded=len(documents),
            chunks_created=num_chunks,
            status="indexed",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question against whatever's currently indexed."""
    query_engine = rag.get_active_query_engine()
    retriever = rag.get_active_retriever()

    if query_engine is None or retriever is None:
        raise HTTPException(status_code=400, detail="No document indexed yet — call /ingest first.")

    top_n = request.top_k or rag.config.TOP_N_RERANK
    reranked = rag.rerank_nodes(retriever, request.question, top_n=top_n)

    response = await query_engine.aquery(request.question)

    sources = [
        SourceChunk(
            page=node.node.metadata.get("page"),
            score=float(node.score) if node.score is not None else 0.0,
            text_preview=node.node.text[:200].replace("\n", " "),
        )
        for node in reranked
    ]

    return QueryResponse(
        question=request.question,
        answer=str(response),
        sources=sources,
    )

# TODO #3 — GET /documents

# TODO #4 — POST /eval
