import json
import chromadb
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from config import SYSTEM_PROMPT
from schemas.chat import ChatRequest
from ollama_client import inferring_ollama
from rag_engine import recherche_depuis_texte, get_collection

router = APIRouter(tags=["Chat & RAG"])

@router.get("/lexique")
def lexique():
    client = chromadb.PersistentClient(path="./chromadb")
    try:
        collection = client.get_collection(name="base_connaissances_globale_acronymes")
        tous_les_docs = collection.get()
        if not tous_les_docs['documents']:
            return {"message": "La base est actuellement vide."}
        return {
            "total_acronymes": len(tous_les_docs['documents']),
            "donnees": tous_les_docs['documents']
        }
    except ValueError:
        return {"erreur": "La collection n'existe pas. Avez-vous lancé le script d'ingestion ?"}

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

@router.post("/chat_with_rag")
async def generer_chat_rag(requete: ChatRequest):
    texte_user = requete.messages[-1]["content"]
    acronymes_resolus = recherche_depuis_texte(texte_user, get_collection())
    contexte = "\n".join(f"{acr} = {sig}" for acr, sig in acronymes_resolus.items())

    try:
        stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}
        system_prompt = f"""{SYSTEM_PROMPT}\nVoici le lexique des acronymes de Eau de Paris détectés dans la question :\n{contexte}\n..."""
        messages_pour_ollama = [{"role": "system", "content": system_prompt}] + requete.messages

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
        print(f"Erreur /chat_with_rag : {str(e)}")