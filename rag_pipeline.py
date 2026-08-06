import os
import fitz  # PyMuPDF

from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss

from app import config

# ---------------------------------------------------------------------------
# Module state — set by init_models() / chunk_and_build_index() at runtime
# ---------------------------------------------------------------------------
_index = None
_retriever: VectorIndexRetriever | None = None
_reranker: SentenceTransformerRerank | None = None
_query_engine: RetrieverQueryEngine | None = None


def init_models() -> None:
    """Called once on app startup. Sets the global LLM + embedding model,
    same as Settings.llm / Settings.embed_model in your notebook's Cell 6."""
    llm = GoogleGenAI(model=config.LLM_MODEL_NAME, api_key=config.GOOGLE_API_KEY)
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_NAME)

    Settings.llm = llm
    Settings.embed_model = embed_model

    global _reranker
    _reranker = SentenceTransformerRerank(
        model=config.RERANKER_MODEL_NAME,
        top_n=config.TOP_N_RERANK,
    )
    print("Models initialized (GoogleGenAI + HuggingFace embeddings + reranker).")


def load_pdf_as_documents(pdf_path: str) -> list[Document]:
    """Direct port of your notebook's Cell 5. Parses a PDF into one
    llama-index Document per page."""
    pdf = fitz.open(pdf_path)
    docs = []
    for i, page in enumerate(pdf):
        text = page.get_text("text").strip()
        if text:
            docs.append(Document(text=text, metadata={"page": i + 1}))
    return docs


def build_index(nodes, persist_dir: str = config.PERSIST_DIR, force_rebuild: bool = False):
    """Direct port of your notebook's Cell 7. Builds a FAISS-backed
    VectorStoreIndex, or loads it from disk if it already exists."""
    if os.path.exists(persist_dir) and not force_rebuild:
        print(f"Loading existing index from {persist_dir}...")
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=persist_dir
        )
        return load_index_from_storage(storage_context)

    print(f"Building new index ({len(nodes)} nodes) and persisting to {persist_dir}...")
    faiss_index = faiss.IndexFlatL2(config.EMBED_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    return index


# ---------------------------------------------------------------------------
# TODO #1 — chunk_and_build_index
#
# This is the piece that turns raw Documents (one per PDF page) into chunked
# nodes and hands them to build_index(). In your notebook this was just two
# lines inline in Cell 8:
#
#     splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     nodes = splitter.get_nodes_from_documents(documents)
#
# Your job: implement this function so it:
#   1. Creates a SentenceSplitter using config.CHUNK_SIZE / config.CHUNK_OVERLAP
#   2. Calls splitter.get_nodes_from_documents(documents) to get `nodes`
#   3. Calls build_index(nodes, force_rebuild=True) and returns the result
#   4. Also returns len(nodes) so the /ingest route can report chunk count
#
# Suggested signature (feel free to adjust):
#   def chunk_and_build_index(documents: list[Document]) -> tuple[VectorStoreIndex, int]:
# ---------------------------------------------------------------------------
def chunk_and_build_index(documents: list[Document]):
    raise NotImplementedError("TODO #1: implement chunk_and_build_index (see comment above)")


def get_retriever(index, top_k: int = config.TOP_K_RETRIEVE) -> VectorIndexRetriever:
    """Direct port of your notebook's retriever setup in Cell 8."""
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)


def get_query_engine(retriever: VectorIndexRetriever) -> RetrieverQueryEngine:
    """Direct port of your notebook's Cell 8 query engine assembly."""
    synthesizer = get_response_synthesizer(response_mode="compact")
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[_reranker],
    )


# ---------------------------------------------------------------------------
# TODO #2 — rerank_nodes
#
# This is the "debug_retrieval" logic from your notebook's Cell 9, minus the
# print statements — it needs to become a real function the /query route can
# call to get back scored, reranked nodes (used to populate `sources` in the
# response).
#
# Your job: implement this function so it:
#   1. Sets retriever.similarity_top_k = config.TOP_K_RETRIEVE
#   2. Calls retriever.retrieve(question) to get baseline nodes
#   3. Calls _reranker.postprocess_nodes(baseline, query_str=question)
#   4. Slices to the first `top_n` results and returns them
#
# Suggested signature (feel free to adjust):
#   def rerank_nodes(retriever, question: str, top_n: int = config.TOP_N_RERANK):
# ---------------------------------------------------------------------------
def rerank_nodes(retriever: VectorIndexRetriever, question: str, top_n: int = config.TOP_N_RERANK):
    raise NotImplementedError("TODO #2: implement rerank_nodes (see comment above)")


def evaluate_retrieval(eval_set: list[dict], retrieve_fn, k: int) -> dict:
    """Direct port of your notebook's Cell 10 evaluation function.
    retrieve_fn(question, k) -> list of nodes with .node.metadata['page'].
    Computes hit rate@k and MRR."""
    hits = 0
    reciprocal_ranks = []

    for item in eval_set:
        results = retrieve_fn(item["question"], k)
        retrieved_pages = [r.node.metadata.get("page") for r in results]

        expected = set(item["expected_pages"])
        hit_rank = next(
            (i for i, p in enumerate(retrieved_pages, 1) if p in expected), None
        )

        if hit_rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(eval_set)
    return {
        "hit_rate": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "n_questions": n,
    }


# ---------------------------------------------------------------------------
# Accessors for the route layer — keep route code from touching globals
# directly.
# ---------------------------------------------------------------------------
def set_active_index(index) -> None:
    global _index, _retriever, _query_engine
    _index = index
    _retriever = get_retriever(index)
    _query_engine = get_query_engine(_retriever)


def get_active_query_engine() -> RetrieverQueryEngine | None:
    return _query_engine


def get_active_retriever() -> VectorIndexRetriever | None:
    return _retriever


def get_reranker() -> SentenceTransformerRerank | None:
    return _reranker