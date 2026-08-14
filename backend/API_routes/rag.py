#backend/API_routes/rag.py

import os
from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
CHATBOT_ROLE = os.environ.get("CHATBOT_ROLE", "general")

from engines.rag_engine import (
    make_qdrant_client,
    make_ollama_client,
    list_collections,
    retrieve_context_hybrid,
    rewrite_query,
    list_doc_dates,
    list_registry,
    registry_for_tool_calling,
    ensure_registry,
)

router = APIRouter(prefix="/rag", tags=["RAG Links"])


@router.get("/collections",
    summary="List available collections",
    description="Returns a list of all available collections in the vector database",
    response_description="List of collection names",
    responses={
        200: {
            "description": "Successfully returned list of collections",
            "content": {
                "application/json": {
                    "example": {"collections": ["collection1", "collection2"]}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error listing collections"}
                }
            }
        }
    })
def get_collections_endpoint():
    client = make_qdrant_client()
    try:
        return {"collections": list_collections(client)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/collections/{collection_name}/dates",
    summary="Get document dates for a collection",
    description="Returns the available document dates for a specific collection",
    response_description="List of document dates",
    responses={
        200: {
            "description": "Successfully returned document dates",
            "content": {
                "application/json": {
                    "example": {"dates": ["2023-01-01", "2023-01-02"]}
                }
            }
        },
        404: {
            "description": "Collection not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Collection not found"}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error retrieving dates"}
                }
            }
        }
    })
def get_collection_dates_endpoint(
    collection_name: str = Path(..., description="Name of the collection to get dates for")
):
    client = make_qdrant_client()
    try:
        # ⬅️ Changement ici : on passe directement le client et le nom de la collection
        return list_doc_dates(client, collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry_evolve",
    summary="Special route to expose to the app which collection from qdrant it can access",
    description="Return entries from _registry collection which are all the collections this app has access to based on its role.",
    response_description="List of filtered registry entries",
    responses={
        200: {
            "description": "Successfully returned entries from _registry",
            "content": {
                "application/json": {
                    "example": {
                        "registry": [
                            {"name": "collection_1", "description": "In this collection you'll find ..."},
                            {"name": "collection_2", "description": "In this collection you'll find ..."}
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error retrieving collections"}
                }
            }
        }
    })
def get_registry_endpoint_evolve():
    client = make_qdrant_client()
    try:
        all_entries = list_registry(client)
        registry_entries = registry_for_tool_calling(client, role=CHATBOT_ROLE)
        return {"registry": registry_entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    collection_name: str = Query(..., description="Name of the collection to search in")
    query: str = Query(..., description="Search query text")
    model: str = Query(..., description="Generative model to use for search")
    n_results: int = Query(5, description="Number of results to return", ge=1, le=50)
    seuil: float = Query(0.5, description="Similarity threshold for results", ge=0.0, le=1.0)
    alpha: float = Query(0.5, description="Hybrid search weight parameter", ge=0.0, le=1.0)
    use_hyde: bool = Query(False, description="Whether to use Hypothetical Document Embeddings")
    use_expansion: bool = Query(False, description="Whether to use query expansion")
    doc_date_filter: str = Query("", description="Optional date filter for documents")

    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "my_collection",
                "query": "What is the capital of France?",
                "model": "gemma4:e4b",
                "n_results": 5,
                "seuil": 0.5,
                "alpha": 0.5,
                "use_hyde": False,
                "use_expansion": False,
                "doc_date_filter": "2023-01-01"
            }
        }



@router.post("/search",
    summary="Search in a collection",
    description="Perform a hybrid search in the specified collection using the given query and parameters",
    response_description="Search results with contexts, sources, and detailed chunks",
    responses={
        200: {
            "description": "Successfully performed search",
            "content": {
                "application/json": {
                    "example": {
                        "contexts": ["context1", "context2"],
                        "sources": ["source1", "source2"],
                        "detailed_chunks": [{"text": "chunk1", "score": 0.95}]
                    }
                }
            }
        },
        404: {
            "description": "Collection not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Collection not found"}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error during search"}
                }
            }
        }
    })
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
    query: str = Query(..., description="Original query to rewrite")
    model: str = Query(..., description="Generative model to use for rewriting")
    chat_history: List[Dict[str, str]] = Query(
        [],
        description="Optional chat history for context-aware rewriting"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "model": "llama2",
                "chat_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        }



class StreamAnswerRequest(BaseModel):
    system_prompt: str = Query(..., description="System prompt for the LLM")
    query: str = Query(..., description="Query to answer")
    model: str = Query(..., description="Generative model to use for answering")
    chat_history: Optional[List[Dict[str, str]]] = Query(
        None,
        description="Optional chat history for conversational context"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "system_prompt": "You are a helpful assistant.",
                "query": "What is the capital of France?",
                "model": "llama2",
                "chat_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        }


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

import random

@router.get("/collections/{collection_name}/random",
    summary="Get random chunk from collection (just used in debugging)",
    description="Returns a random document chunk from the specified collection",
    response_description="Random chunk data or null if collection is empty",
    responses={
        200: {
            "description": "Successfully returned random chunk",
            "content": {
                "application/json": {
                    "example": {
                        "chunk": {
                            "text": "Sample document text",
                            "metadata": {"source": "document.pdf", "page": 1}
                        }
                    }
                }
            }
        },
        404: {
            "description": "Collection not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Collection not found"}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error retrieving random chunk"}
                }
            }
        }
    })
def get_random_chunk_endpoint(
    collection_name: str = Path(..., description="Name of the collection to get random chunk from")
):
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
