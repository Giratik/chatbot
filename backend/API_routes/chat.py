# backend/routers/chat.py

"""
Routeur API : Conversations (Chat)
Description : Définit les points d'entrée de l'API pour l'envoi des messages utilisateur, 
                la gestion de l'historique et la génération de réponses classiques ou RAG.
"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from core.config import SYSTEM_PROMPT, CHATBOT_ROLE
from services.ollama_client import inferring_ollama, client as ollama_sdk_client

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from engines.rag_engine import make_qdrant_client, retrieve_context_hybrid, build_system_prompt
from API_routes.rag import registry_for_tool_calling

import os
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 22000))

router = APIRouter(tags=["Chat"])


# =============================================================================
# MODÈLES
# =============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    think: bool = False

class ChatWithToolsRequest(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    think: bool = False
    n_results: Optional[int] = 5
    seuil: Optional[float] = 0.5
    alpha: Optional[float] = 0.5

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


# =============================================================================
# HELPERS
# =============================================================================

def build_qdrant_tools(qdrant_client) -> list:
    """
    Construit le schéma d'outils pour le tool calling en récupérant
    dynamiquement les collections disponibles depuis _registry.
    Fallback sur une description générique si le registre est inaccessible.
    """
    try:
        registry_entries = registry_for_tool_calling(qdrant_client, role=CHATBOT_ROLE)

        if registry_entries:
            texte_description = "Nom de la collection dans laquelle chercher. Voici les choix obligatoires :\n"
            for entry in registry_entries:
                if entry.get("nom") and entry.get("description"):
                    texte_description += f"- '{entry['nom']}' : à utiliser pour des recherches concernant {entry['description']}.\n"
        else:
            texte_description = (
                "Nom de la collection dans laquelle chercher. "
                "(Attention : aucune collection disponible dans le registre)."
            )
    except Exception as e:
        print(f"Erreur build_qdrant_tools : {e}")
        texte_description = "Erreur. Qdrant est injoignable."

    return [
        {
            "type": "function",
            "function": {
                "name": "rechercher_dans_qdrant",
                "description": (
                    "Recherche des informations spécifiques dans la base de données de l'entreprise. "
                    "À utiliser uniquement si la question demande des données factuelles, "
                    "des contacts, des procédures ou des informations sur le personnel."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": texte_description,
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


# =============================================================================
# ROUTES
# =============================================================================

@router.post("/chat")
async def generer_chat(requete: ChatRequest):
    """Chat direct sans RAG ni tool calling."""
    try:
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
    except Exception as e:
        print(f"Erreur /chat : {str(e)}")


@router.post("/chat_with_tools")
async def generer_chat_with_tools(requete: ChatWithToolsRequest):
    """
    Chat avec tool calling dynamique :
    1. Le LLM évalue s'il a besoin de chercher dans Qdrant (via SDK Ollama).
    2. Si oui : recherche hybride Qdrant → system prompt RAG → réponse streamée.
    3. Si non : réponse directe streamée avec system prompt standard.
    """
    stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}

    try:
        qdrant_client = make_qdrant_client()
        qdrant_tools = build_qdrant_tools(qdrant_client)

        # --- ÉTAPE 1 : Évaluation tool calling (SDK direct, non-streamé) ---
        response_eval = ollama_sdk_client.chat(
            model=requete.modele,
            messages=requete.messages,
            tools=qdrant_tools,
            options={
                "num_ctx": requete.context_size,
                "temperature": 0.0,  # déterministe pour le choix d'outil
            },
        )

        if response_eval.message.tool_calls:
            tool_call = response_eval.message.tool_calls[0]
            args = tool_call.function.arguments
            collection_cible = args.get("collection_name")
            query_outil = args.get("query")

            # --- ÉTAPE 2 : Recherche Qdrant ciblée ---
            from engines.rag_engine import make_ollama_client
            contexts, sources, detailed_chunks = retrieve_context_hybrid(
                qdrant_client=qdrant_client,
                collection_name=collection_cible,
                query=query_outil,
                ollama_client=make_ollama_client(),
                model=requete.modele,
                n_results=requete.n_results,
                seuil=requete.seuil,
                alpha=requete.alpha,
                use_hyde=False,
                use_expansion=False,
            )

            context_str = "\n\n".join(contexts) if contexts else "Aucun contexte pertinent trouvé."
            messages_pour_ollama = (
                [{"role": "system", "content": build_system_prompt(context_str)}]
                + requete.messages
            )
        else:
            # --- Pas d'outil : réponse directe ---
            messages_pour_ollama = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + requete.messages
            )

        # --- ÉTAPE 3 : Génération streamée ---
        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                stats_dict=stats_dict,
                context_size=requete.context_size,
                think=requete.think,
            ):
                yield chunk
            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        print(f"Erreur /chat_with_tools : {str(e)}")

        # Fallback : réponse directe sans outil
        messages_pour_ollama = [{"role": "system", "content": SYSTEM_PROMPT}] + requete.messages

        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                stats_dict=stats_dict,
                context_size=requete.context_size,
                think=requete.think,
            ):
                yield chunk
            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"

        return StreamingResponse(stream_generator(), media_type="text/plain")