#backend/routers/chat.py

"""
Routeur API : Conversations (Chat)
Description : Définit les points d'entrée de l'API pour l'envoi des messages utilisateur, 
                la gestion de l'historique et la génération de réponses classiques ou RAG.
"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from core.config import SYSTEM_PROMPT
from services.ollama_client import inferring_ollama

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from engines.rag_engine import make_qdrant_client, retrieve_context_hybrid, build_system_prompt


router = APIRouter(tags=["Chat"])




class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    think: bool = False
    collection_name: Optional[str] = None
    n_results: Optional[int] = None
    seuil: Optional[float] = None
    alpha: Optional[float] = None
    use_hyde: Optional[bool] = None
    use_expansion: Optional[bool] = None
    doc_date_filter: Optional[str] = None

class RetrieveRequest(BaseModel):
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

class RewriteRequest(BaseModel):
    query: str
    model: str
    chat_history: List[Message] = []

class StreamChatRequest(BaseModel):
    collection_name: str
    query: str
    model: str
    system_prompt_context: str
    chat_history: List[Message] = []


@router.post("/chat")
async def generer_chat(requete: ChatRequest):
    try:
        stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}

        # Check if RAG parameters are provided and collection is valid
        if (requete.collection_name and requete.collection_name != "aucune_collection" and
            requete.collection_name != "aucune_collection_trouvee"):
            # Use RAG-enhanced chat
            qdrant_client = make_qdrant_client()

            # Get the user's actual query (last user message)
            user_query = ""
            for msg in reversed(requete.messages):
                if msg["role"] == "user":
                    user_query = msg["content"]
                    break

            if user_query:
                # Retrieve context from Qdrant
                from engines.rag_engine import make_ollama_client
                ollama_client = make_ollama_client()

                contexts, sources, detailed_chunks = retrieve_context_hybrid(
                    qdrant_client=qdrant_client,
                    collection_name=requete.collection_name,
                    query=user_query,
                    ollama_client=ollama_client,
                    model=requete.modele,
                    n_results=requete.n_results or 5,
                    seuil=requete.seuil or 0.5,
                    alpha=requete.alpha or 0.5,
                    use_hyde=requete.use_hyde if requete.use_hyde is not None else False,
                    use_expansion=requete.use_expansion if requete.use_expansion is not None else False,
                    doc_date_filter=requete.doc_date_filter or "",
                )

                # Build context string for system prompt
                context_str = "\n\n".join(contexts) if contexts else "Aucun contexte pertinent trouvé."

                # Build RAG-enhanced system prompt
                rag_system_prompt = build_system_prompt(context_str)

                messages_pour_ollama = [{"role": "system", "content": rag_system_prompt}] + requete.messages
            else:
                # Fallback to regular chat if no user query found
                messages_pour_ollama = [{"role": "system", "content": SYSTEM_PROMPT}] + requete.messages
        else:
            # Regular chat without RAG
            messages_pour_ollama = [{"role": "system", "content": SYSTEM_PROMPT}] + requete.messages

        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama, model=requete.modele,
                temperature=requete.temperature, stream=True, stats_dict=stats_dict,
                context_size=requete.context_size, think=requete.think,
            ):
                yield chunk
            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"

        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        print(f"Erreur /chat : {str(e)}")
        # Fallback to regular chat if RAG fails
        stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}
        messages_pour_ollama = [{"role": "system", "content": SYSTEM_PROMPT}] + requete.messages

        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama, model=requete.modele,
                temperature=requete.temperature, stream=True, stats_dict=stats_dict,
                context_size=requete.context_size, think=requete.think,
            ):
                yield chunk
            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"

        return StreamingResponse(stream_generator(), media_type="text/plain")
