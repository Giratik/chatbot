#backend/routers/rag_engine_router.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from engines.rag_engine import (
    make_qdrant_client,
    make_ollama_client,
    list_collections,
    list_generative_models,
    retrieve_context_hybrid,
    rewrite_query,
    stream_answer,
    list_doc_dates,
)

router = APIRouter(prefix="/rag", tags=["RAG Engine"])


@router.get("/collections")
def get_collections_endpoint():
    client = make_qdrant_client()
    try:
        return {"collections": list_collections(client)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/dates")
def get_collection_dates_endpoint(collection_name: str):
    client = make_qdrant_client()
    try:
        # ⬅️ Changement ici : on passe directement le client et le nom de la collection
        return list_doc_dates(client, collection_name)
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
    doc_date_filter: str = ""


@router.post("/search")
def search_endpoint(req: SearchRequest):
    qdrant_client = make_qdrant_client()
    try:
        # ⬅️ Changement ici : plus d'objet "collection", on utilise le client et la string
        contexts, sources, detailed_chunks = retrieve_context_hybrid(
            qdrant_client,
            req.collection_name,
            req.query,
            make_ollama_client(),
            req.model,
            req.n_results,
            req.seuil,
            req.alpha,
            req.use_hyde,
            req.use_expansion,
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


import random

@router.get("/collections/{collection_name}/random")
def get_random_chunk_endpoint(collection_name: str):
    client = make_qdrant_client()
    try:
        # On récupère un lot de 100 points maximum pour piocher dedans
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        if not records:
            return {"chunk": None}
        
        # On choisit un record au hasard et on renvoie son payload
        choice = random.choice(records)
        return {"chunk": choice.payload}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))