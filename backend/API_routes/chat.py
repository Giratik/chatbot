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



QDRANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rechercher_dans_qdrant",
            "description": "Recherche des informations spécifiques dans la base de données de l'entreprise. "
                           "À utiliser uniquement si la question demande des données factuelles, "
                           "des contacts, des procédures ou des informations sur le personnel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Nom de la collection dans laquelle chercher. "
                                       "Exemples : 'organigramme' (membres, postes, emails, téléphones) "
                                       "ou 'fiches_generales' (siège, horaires, liens intranet, contacts).",
                    },
                    "query": {
                        "type": "string",
                        "description": "La requête ou les mots-clés optimisés pour la recherche.",
                    },
                },
                "required": ["collection_name", "query"],
            },
        },
    }
]

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
        
        from engines.rag_engine import make_ollama_client
        ollama_client = make_ollama_client()

        # --- ÉTAPE 1 : Le LLM évalue s'il a besoin d'un outil ---
        # Appel rapide non-streamé pour vérifier l'intention
        response_eval = ollama_client.chat(
            model=requete.modele,
            messages=requete.messages,
            tools=QDRANT_TOOLS,
        )

        messages_pour_ollama = []

        # Verification si le modèle souhaite appeler l'outil de recherche
        if response_eval.message.tool_calls:
            tool_call = response_eval.message.tool_calls[0]
            args = tool_call.function.arguments
            
            collection_cible = args.get("collection_name", requete.collection_name)
            query_outil = args.get("query")

            # Execute la recherche Qdrant ciblée
            qdrant_client = make_qdrant_client()
            contexts, sources, detailed_chunks = retrieve_context_hybrid(
                qdrant_client=qdrant_client,
                collection_name=collection_cible,
                query=query_outil,
                ollama_client=ollama_client,
                model=requete.modele,
                n_results=requete.n_results or 5,
                seuil=requete.seuil or 0.5,
                alpha=requete.alpha or 0.5,
            )

            context_str = "\n\n".join(contexts) if contexts else "Aucun contexte pertinent trouvé."
            rag_system_prompt = build_system_prompt(context_str)
            
            messages_pour_ollama = [{"role": "system", "content": rag_system_prompt}] + requete.messages
        else:
            # Le modèle a estimé ne pas avoir besoin de chercher dans Qdrant
            messages_pour_ollama = [{"role": "system", "content": SYSTEM_PROMPT}] + requete.messages

        # --- ÉTAPE 2 : Génération de la réponse finale en streaming ---
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
