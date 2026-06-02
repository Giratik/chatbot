#"""Router API : Chat with RAG
#Description : Endpoint that enriches the user query with acronym resolution using the RAG engine
#"""
#
#import json
#from fastapi import APIRouter
#from fastapi.responses import StreamingResponse
#from core.config import SYSTEM_PROMPT
#from services.ollama_client import inferring_ollama
#from engines.rag_engine import recherche_depuis_texte, get_collection
#
#router = APIRouter(tags=["Chat & RAG"])
#
## Re‑use the same request models as in the general chat router
#from pydantic import BaseModel
#from typing import List, Dict, Any
#
#
#class Message(BaseModel):
#    role: str
#    content: str
#
#
#class ChatRequest(BaseModel):
#    messages: List[Dict[str, Any]]
#    modele: str
#    temperature: float
#    context_size: int
#    think: bool = False
#
#
#@router.post("/chat_with_rag")
#async def generer_chat_rag(requete: ChatRequest):
#    """Generate a response using Ollama with RAG‑enhanced system prompt.
#
#    The last user message is analysed to extract acronyms via the RAG engine.
#    Those acronyms are appended to the system prompt before calling the LLM.
#    """
#    texte_user = requete.messages[-1]["content"]
#    # Resolve acronyms from the user's question
#    acronymes_resolus = recherche_depuis_texte(texte_user, get_collection())
#    contexte = "\n".join(f"{acr} = {sig}" for acr, sig in acronymes_resolus.items())
#
#    try:
#        stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}
#        system_prompt = (
#            f"{SYSTEM_PROMPT}\n"
#            "Voici le lexique des acronymes détectés dans la question :\n"
#            f"{contexte}\n..."
#        )
#        messages_pour_ollama = [{"role": "system", "content": system_prompt}] + requete.messages
#
#        def stream_generator():
#            for chunk in inferring_ollama(
#                messages=messages_pour_ollama,
#                model=requete.modele,
#                temperature=requete.temperature,
#                stream=True,
#                stats_dict=stats_dict,
#                context_size=requete.context_size,
#                think=requete.think,
#            ):
#                yield chunk
#            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"
#
#        return StreamingResponse(stream_generator(), media_type="text/plain")
#    except Exception as e:
#        print(f"Erreur /chat_with_rag : {str(e)}")
#