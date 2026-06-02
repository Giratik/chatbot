#backend/routers/chat.py

"""
Routeur API : Conversations (Chat)
Description : Définit les points d'entrée de l'API pour l'envoi des messages utilisateur, 
                la gestion de l'historique et la génération de réponses classiques ou RAG.
"""

import json
import chromadb
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from core.config import SYSTEM_PROMPT
from services.ollama_client import inferring_ollama

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


router = APIRouter(tags=["Chat & RAG"])




class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    think: bool = False

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

#@router.get("/lexique")
#def lexique():
#    client = chromadb.PersistentClient(path="./chromadb")
#    try:
#        collection = client.get_collection(name="base_connaissances_globale_acronymes")
#        tous_les_docs = collection.get()
#        if not tous_les_docs['documents']:
#            return {"message": "La base est actuellement vide."}
#        return {
#            "total_acronymes": len(tous_les_docs['documents']),
#            "donnees": tous_les_docs['documents']
#        }
#    except ValueError:
#        return {"erreur": "La collection n'existe pas. Avez-vous lancé le script d'ingestion ?"}

@router.post("/chat")
async def generer_chat(requete: ChatRequest):
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

# The RAG‑specific endpoint has been moved to a dedicated router in `chat_rag.py`.