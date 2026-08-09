# Wires everything together
from fastapi import FastAPI

from app.routes import router
from app.rag_pipeline import init_models

app = FastAPI(
    title="RAG Pipeline API",
    description="Upload a PDF, ask questions, then get some answers.",
    version="1",
)


@app.on_event("startup")
def on_startup():
    """Load LLM, embedding model, and reranker. Everything starts once when server
    starts."""
    init_models()


app.include_router(router)


@app.get("/")
def root():
    return {"message": "RAG Pipeline API is running. See /docs for endpoints."}
