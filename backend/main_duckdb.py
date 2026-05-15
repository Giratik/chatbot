# backend/main.py

import json
import traceback

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import io
import os
import time
import threading
import chromadb

# Imports locaux existants (inchangés)
from ollama_client import inferring_ollama
from file_type_action import analyser_contenu_fichier
from rag_engine import remplir_database_chroma, recherche_lexique, recherche_depuis_texte, get_collection

# Nouveau module DuckDB (remplace new_xlsx_parser)
import duckdb_session as ddb

CONTEXT_SIZE = os.environ.get("CONTEXT_SIZE", 12288)
URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


SYSTEM_PROMPT = """
Tu es "EDP-IA", l'assistant IA officiel de l'entreprise Eau de Paris. 

--- TON IDENTITÉ ET TON RÔLE ---
* Tu es un expert technique, professionnel, mais toujours amical et concis.
* Ton but est d'aider les salariés de EDP à analyser leurs documents et à répondre à leurs questions.
* Tu ne dois jamais inventer d'informations (pas d'hallucinations). Si tu ne sais pas, dis-le simplement.

--- TES CONNAISSANCES DE BASE ---
* L'entreprise se spécialise dans la distribution de l'eau dans la ville de Paris.

--- RÈGLES DE FORMATAGE ---
* Réponds toujours en français.
* Utilise le format Markdown pour structurer tes réponses (listes à puces, texte en gras pour mettre en évidence les éléments clés).
* Ne sois pas trop bavard : va droit au but.
* Répond avec un minimum de déférence.
"""

SYSTEM_PROMPT_DATA_ANALYST = """
Tu es un expert SQL et data analyst pour DuckDB.
Tu as accès à une base DuckDB in-memory avec les tables suivantes :

{schema}

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec du SQL DuckDB valide, entouré de ```sql et ```.
2. Utilise EXACTEMENT les noms de tables et de colonnes fournis ci-dessus.
3. Pour une question texte/résumé : génère un SELECT adapté.
4. Pour un graphe : génère un SELECT avec les colonnes nécessaires et ajoute un commentaire
   -- CHART_TYPE: bar|line|pie|scatter
   -- CHART_X: nom_colonne_x
   -- CHART_Y: nom_colonne_y
   -- CHART_TITLE: titre lisible
5. Ne génère jamais de code Python. Uniquement du SQL.
6. Si la question est impossible à répondre avec les données disponibles, dis-le clairement en français (sans SQL).
"""


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="API EDP Chatbot")

verrou_vlm_image = asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# Modèles Pydantic
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int

class ChatRequest_csv(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    session_id: str = "default"

class SqlRequest(BaseModel):
    sql: str
    session_id: str = "default"

class SessionRequest(BaseModel):
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Démarrage
# ---------------------------------------------------------------------------

def routine_demarrage():
    remplir_database_chroma()

@app.on_event("startup")
def startup_event():
    threading.Thread(target=routine_demarrage).start()


# ---------------------------------------------------------------------------
# Routes existantes — inchangées
# ---------------------------------------------------------------------------

@app.get("/lexique")
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


@app.post("/upload_fichier")
async def traiter_fichier(file: UploadFile = File(...), modele: str = Form(...)):
    start_time = time.time()
    try:
        file_bytes = await file.read()
        faux_fichier_streamlit = io.BytesIO(file_bytes)
        faux_fichier_streamlit.name = file.filename
        faux_fichier_streamlit.type = file.content_type
        async with verrou_vlm_image:
            contenu = analyser_contenu_fichier(faux_fichier_streamlit, modele)
        return {"nom_fichier": file.filename, "contenu": contenu}
    except Exception as e:
        print(f"Erreur backend lors de l'analyse : {str(e)}")
        return {"erreur": str(e)}


@app.post("/chat")
async def generer_chat(requete: ChatRequest):
    try:
        stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}
        messages_pour_ollama = [{"role": "system", "content": SYSTEM_PROMPT}] + requete.messages

        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                stats_dict=stats_dict,
                context_size=requete.context_size,
            ):
                yield chunk
            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"

        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        print(f"Erreur /chat : {str(e)}")


@app.post("/chat_with_rag")
async def generer_chat_rag(requete: ChatRequest):
    texte_user = requete.messages[-1]["content"]
    acronymes_resolus = recherche_depuis_texte(texte_user, get_collection())
    contexte = "\n".join(f"{acr} = {sig}" for acr, sig in acronymes_resolus.items())

    try:
        stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}
        system_prompt = f"""{SYSTEM_PROMPT}
Voici le lexique des acronymes de Eau de Paris détectés dans la question :
{contexte}
ATTENTION : n'invente aucune information, si l'acronyme apparait dans le lexique tu ne peux utiliser
dans ta réponse que la définition associée et pas provenant de tes connaissances.
Si la requête de l'utilisateur comporte un acronyme qui n'apparait pas dans ton lexique,
tu dois l'en informer et à ce moment utilise tes connaissances pour répondre.
Si la requête n'a rien à voir avec les acronymes détectés, ignore-les et converse normalement.
Requête de l'utilisateur :"""

        messages_pour_ollama = [{"role": "system", "content": system_prompt}] + requete.messages

        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                stats_dict=stats_dict,
                context_size=requete.context_size,
            ):
                yield chunk
            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"

        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        print(f"Erreur /chat_with_rag : {str(e)}")


# ---------------------------------------------------------------------------
# Routes Excel — réécrites avec DuckDB
# ---------------------------------------------------------------------------

@app.post("/parse_excel")
async def parse_excel(
    file: UploadFile = File(...),
    sheet_name: str = Query("Sheet1"),
    session_id: str = Query("default"),
):
    """
    Upload d'un fichier Excel :
      1. Détecte les îlots de données
      2. Crée une session DuckDB in-memory
      3. Enregistre chaque îlot comme table SQL
      4. Retourne le schéma pour affichage côté frontend

    Remplace : /parse_every_tab_excel + /knowledge_graphe
    """
    try:
        file_bytes = await file.read()

        # Création / remplacement de la session DuckDB
        session = ddb.registry.create(session_id)
        table_names = session.load_excel(file_bytes, sheet_name)

        return {
            "status": "success",
            "session_id": session.session_id,
            "nom_fichier": file.filename,
            "sheet_name": sheet_name,
            "tables": session.get_tables_info(),
        }

    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        print(f"Erreur /parse_excel : {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


@app.post("/chat_data_analyst")
async def chat_data_analyst(requete: ChatRequest_csv):
    """
    Assistant data analyst : le LLM génère du SQL DuckDB à partir du schéma.
    Le SQL est retourné en streaming, exécuté par /execute_sql côté client
    ou directement ici selon le besoin.

    Remplace : /chat_data_analyst (Polars codegen) + /chat_csv_rag (ChromaDB)
    """
    try:
        session = ddb.registry.get_or_raise(requete.session_id)
        schema = session.build_schema_prompt()

        system_prompt = SYSTEM_PROMPT_DATA_ANALYST.format(schema=schema)
        messages_pour_ollama = [{"role": "system", "content": system_prompt}] + requete.messages

        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                context_size=requete.context_size,
                max_tokens=800,
            ):
                yield chunk

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except ValueError as e:
        def erreur_stream():
            yield f"❌ {str(e)}"
        return StreamingResponse(erreur_stream(), media_type="text/plain")
    except Exception as e:
        print(f"Erreur /chat_data_analyst : {traceback.format_exc()}")
        def erreur_stream():
            yield f"❌ Erreur inattendue : {str(e)}"
        return StreamingResponse(erreur_stream(), media_type="text/plain")


@app.post("/execute_sql")
async def execute_sql(requete: SqlRequest):
    """
    Exécute une requête SQL sur la session DuckDB de l'utilisateur.
    Retourne les données JSON pour que le frontend construise le graphe (Plotly).

    Workflow typique :
      1. /chat_data_analyst → LLM génère du SQL avec métadonnées graphe
      2. Frontend parse le SQL + les commentaires CHART_*
      3. Frontend POST /execute_sql avec le SQL
      4. Frontend reçoit les données + construit le graphe Plotly
    """
    try:
        session = ddb.registry.get_or_raise(requete.session_id)
        records = session.query_to_records(requete.sql)

        return {
            "status": "success",
            "session_id": requete.session_id,
            "n_rows": len(records),
            "data": records,
        }

    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except duckdb.Error as e:
        return {"status": "sql_error", "message": f"Erreur SQL : {str(e)}"}
    except Exception as e:
        print(f"Erreur /execute_sql : {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Libère explicitement la session DuckDB (fin de conversation)."""
    ddb.registry.delete(session_id)
    return {"status": "success", "message": f"Session '{session_id}' supprimée."}


@app.get("/sessions/count")
async def sessions_count():
    """Nombre de sessions DuckDB actives (monitoring)."""
    return {"active_sessions": ddb.registry.active_count()}