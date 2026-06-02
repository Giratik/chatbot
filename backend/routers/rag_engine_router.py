"""FastAPI router exposing core RAG engine functionalities as HTTP endpoints.

Endpoints provided:
* GET  /rag/collections                      – list available ChromaDB collections
* GET  /rag/collections/{name}/dates         – list available doc_date values for a collection
* GET  /rag/models                           – list available Ollama generative models
* POST /rag/search                           – perform hybrid search and return contexts, sources, detailed chunks
* POST /rag/rewrite                          – rewrite a user query using the LLM
* POST /rag/stream_answer                    – stream LLM answer given a system prompt and user query
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from engines.rag_engine import (
    make_chroma_client,
    make_ollama_client,
    list_collections,
    list_generative_models,
    get_collection,
    retrieve_context_hybrid,
    rewrite_query,
    stream_answer,
    list_doc_dates,
)

router = APIRouter(prefix="/rag", tags=["RAG Engine"])


@router.get("/collections")
def get_collections_endpoint():
    client = make_chroma_client()
    try:
        return {"collections": list_collections(client)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/dates")
def get_collection_dates_endpoint(collection_name: str):
    client = make_chroma_client()
    try:
        collection = get_collection(client, collection_name)
        return list_doc_dates(collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
def get_models_endpoint():
    client = make_ollama_client()
    try:
        return {"models": list_generative_models(client)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    collection_name: str
    query: str
    model: str
    n_results: int = 5
    seuil: float = 0.5
    alpha: float = 0.5
    use_hyde: bool = False
    use_expansion: bool = False
    use_reranker: bool = True
    doc_date_filter: str = ""


@router.post("/search")
def search_endpoint(req: SearchRequest):
    chroma_client = make_chroma_client()
    collection = get_collection(chroma_client, req.collection_name)
    try:
        contexts, sources, detailed_chunks = retrieve_context_hybrid(
            collection,
            req.query,
            make_ollama_client(),
            req.model,
            req.n_results,
            req.seuil,
            req.alpha,
            req.use_hyde,
            req.use_expansion,
            req.use_reranker,
            doc_date_filter=req.doc_date_filter,
        )
        return {
            "contexts": contexts,
            "sources": sources,
            "detailed_chunks": detailed_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RewriteRequest(BaseModel):
    query: str
    model: str
    chat_history: List[Dict[str, str]] = []


@router.post("/rewrite")
def rewrite_endpoint(req: RewriteRequest):
    ollama_client = make_ollama_client()
    try:
        rewritten = rewrite_query(ollama_client, req.model, req.query, req.chat_history)
        return {"rewritten_query": rewritten}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StreamAnswerRequest(BaseModel):
    system_prompt: str
    query: str
    model: str
    chat_history: Optional[List[Dict[str, str]]] = None


def _extract_token(chunk) -> str:
    """Normalise un chunk de stream quelle que soit sa forme :
    - str directe (SimpleOllamaClient)
    - dict {message: {content: ...}} (ollama natif dict)
    - objet ollama avec attribut message.content
    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return chunk.get("message", {}).get("content", "")
    try:
        return chunk.message.content
    except AttributeError:
        return str(chunk)


@router.post("/stream_answer")
def stream_answer_endpoint(req: StreamAnswerRequest):
    ollama_client = make_ollama_client()

    def generator():
        try:
            for chunk in stream_answer(
                ollama_client,
                req.model,
                req.system_prompt,
                req.query,
                req.chat_history,
            ):
                yield _extract_token(chunk)
        except Exception as e:
            yield f"\nERROR:{str(e)}"

    return StreamingResponse(generator(), media_type="text/plain")