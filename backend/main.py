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
import re

# Imports locaux existants (inchangés)
from ollama_client import inferring_ollama
from file_type_action import analyser_contenu_fichier
from rag_engine import remplir_database_chroma, recherche_lexique, recherche_depuis_texte, get_collection

# Nouveau module DuckDB (remplace new_xlsx_parser)
import duckdb_session as ddb
import duckdb as duckdb

CONTEXT_SIZE = os.environ.get("CONTEXT_SIZE", 12288)
URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


SYSTEM_PROMPT = """
Tu es "EDP-IA", l'assistant IA officiel de l'entreprise Eau de Paris. 

--- TON IDENTITÉ ET TON RÔLE ---
* Tu es un expert technique, professionnel, mais toujours amical et concis.
* Tu es utilisé exclusivement par les salariés de Eau de Paris.
* Ton but est de répondre au mieux à leurs questions générales ainsi que leur demandes concernant des fichiers qu'ils joindront. 
* Tu ne dois jamais inventer d'informations (pas d'hallucinations). Si tu ne sais pas, dis-le simplement.

--- RÈGLES DE FORMATAGE ---
* Réponds toujours en français.
* Utilise le format Markdown pour structurer tes réponses (listes à puces, texte en gras pour mettre en évidence les éléments clés).
* Ne sois pas trop bavard : va droit au but.
* Répond avec un minimum de déférence.
* Ne donne pas de conseils non demandés par l'utilisateur. écrit ta réponse et rien d'autre.
* L'utilisateur travaille chez Eau de Paris et à conscience que tu es un outil de Eau de Paris, rapelle le lui quand-même s'il demande ton identité.
"""

SYSTEM_PROMPT_DATA_ANALYST = """
Tu es un expert SQL et data analyst pour DuckDB.
Tu as accès à une base DuckDB in-memory avec les tables suivantes :

{schema}

Selon la question de l'utilisateur, adopte le bon comportement :

CAS 1 — Question descriptive, analytique ou conversationnelle
(ex: "que représente ce tableau", "décris les données", "quelles colonnes as-tu")
→ Réponds en français, en langage naturel, en Markdown.
→ Ne génère PAS de SQL.

CAS 2 — Demande de calcul, agrégation, filtre ou graphique
(ex: "fais un graphique", "donne-moi le total par région", "filtre les lignes > 100")
→ Génère UNIQUEMENT un bloc SQL DuckDB valide entre ```sql et ```.
→ Pour un graphique, ajoute ces commentaires AVANT le SELECT :
   -- CHART_TYPE: bar|line|pie|scatter
   -- CHART_X: nom_colonne_x
   -- CHART_Y: nom_colonne_y
   -- CHART_TITLE: titre lisible
→ Utilise EXACTEMENT les noms de tables et colonnes fournis ci-dessus.
→ Ne génère jamais de code Python.
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
    mode: str = "discussion"  # "discussion" ou "graphique"

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
    try:
        session = ddb.registry.get_or_raise(requete.session_id)
        schema = session.build_schema_prompt()
        is_graphique = requete.mode == "graphique"

        # ── Étape 1 : routing ──────────────────────────────────────────
        system_routing = f"""Tu identifies dans quelle(s) table(s) chercher une information.
Tables disponibles :
{schema}
Réponds UNIQUEMENT avec le ou les noms de tables SQL pertinentes, séparés par des virgules."""

        routing_response = ""
        for chunk in inferring_ollama(
            messages=[
                {"role": "system", "content": system_routing},
                {"role": "user", "content": requete.messages[-1]["content"]}
            ],
            model=requete.modele, temperature=0.0, stream=True,
            context_size=requete.context_size, max_tokens=50,
        ):
            routing_response += chunk
        tables_cibles = routing_response.strip()

        # ── Étape 2 : génération SQL ───────────────────────────────────
        chart_rules = """
- Si l'utilisateur demande un graphique, ajoute ces commentaires AVANT le SELECT :
  -- CHART_TYPE: bar|line|pie|scatter
  -- CHART_X: colonne_x
  -- CHART_Y: colonne_y
  -- CHART_TITLE: titre lisible""" if is_graphique else ""

        system_sql = f"""Tu es un expert SQL DuckDB.
Tables disponibles :
{schema}
La question porte probablement sur : {tables_cibles}

RÈGLES :
- Génère TOUJOURS du SQL, même si la question est vague.
- Pour une question descriptive, génère SELECT * FROM table LIMIT 5.
- Pour les recherches textuelles, utilise ILIKE '%valeur%'.
- Plusieurs tables → plusieurs SELECT séparés dans le même bloc.
{chart_rules}
- Génère UNIQUEMENT un bloc ```sql ... ```, rien d'autre."""

        sql_response = ""
        for chunk in inferring_ollama(
            messages=[{"role": "system", "content": system_sql}] + requete.messages,
            model=requete.modele, temperature=0.1, stream=True,
            context_size=requete.context_size, max_tokens=400,
        ):
            sql_response += chunk

        # ── Étape 3 : exécution SQL ────────────────────────────────────
        sql_match = re.search(r"```sql\n(.*?)\n```", sql_response, re.DOTALL)
        chart_comments = ""
        sql_pur_final = ""
        data_context = ""

        if not sql_match:
            def stream_direct():
                yield sql_response
            return StreamingResponse(stream_direct(), media_type="text/plain")

        # Sépare commentaires CHART_* et SQL pur
        toutes_lignes = sql_match.group(1).splitlines()
        chart_comments = "\n".join(l for l in toutes_lignes if l.strip().startswith("--"))
        lignes_sql = [l for l in toutes_lignes if not l.strip().startswith("--")]

        requetes = [r.strip().rstrip(";") for r in re.split(r'(?=SELECT)', "\n".join(lignes_sql), flags=re.IGNORECASE) if r.strip()]

        resultats = []
        for req in requetes:
            if not req:
                continue
            try:
                df_r = session.query(req)
                is_partial = "LIMIT" in req.upper()
                total_rows = len(df_r)

                if is_partial:
                    table_match = re.search(r"FROM\s+(\w+)", req, re.IGNORECASE)
                    if table_match:
                        try:
                            count_r = session.query(f"SELECT COUNT(*) as n FROM {table_match.group(1)}")
                            total_rows = count_r.row(0)[0]
                        except Exception:
                            pass

                apercu = df_r.to_pandas().to_markdown(index=False)
                table_match = re.search(r"FROM\s+(\w+)", req, re.IGNORECASE)
                nom = table_match.group(1) if table_match else "?"
                label = f"({total_rows} lignes au total, aperçu {len(df_r)})" if is_partial else f"({total_rows} lignes)"
                resultats.append(f"**{nom}** {label} :\n{apercu}")
                sql_pur_final = req  # garde le dernier pour le graphe

            except Exception as e:
                resultats.append(f"Erreur sur `{req}` : {e}")

        data_context = "\n\n---\n\n".join(resultats) if resultats else "Aucun résultat."

        # ── Étape 4 : synthèse ─────────────────────────────────────────
        synthese_extra = """
Si les données sont partielles, dis-le et invite l'utilisateur à poser une question plus ciblée.
Ne mentionne jamais SQL dans ta réponse."""

        graphique_extra = """
Si un graphique est demandé, décris brièvement ce que le graphique va montrer.""" if is_graphique else ""

        system_synthese = f"""Tu es EDP-IA, assistant de Eau de Paris.
Tu viens d'interroger les données Excel de l'utilisateur. Voici les résultats :

{data_context}

RÈGLES :
1. Réponds en français, clairement, en te basant UNIQUEMENT sur ces données.
2. Ne mentionne jamais SQL dans ta réponse.
3. Si les données sont partielles, dis-le et invite l'utilisateur à poser une question plus ciblée.
4. Si l'utilisateur demande un graphique OU si les données s'y prêtent naturellement
   (comparaison, évolution, répartition), réponds normalement EN TEXTE — 
   le graphique sera généré automatiquement par le système si les métadonnées CHART_* 
   sont présentes dans le bloc SQL préfixé."""

        def stream_synthese():
            # Préfixe les métadonnées graphe pour le frontend
            if is_graphique and chart_comments and sql_pur_final:
                yield f"```sql\n{chart_comments}\n{sql_pur_final}\n```\n"
            for chunk in inferring_ollama(
                messages=[{"role": "system", "content": system_synthese}] + requete.messages,
                model=requete.modele, temperature=requete.temperature, stream=True,
                context_size=requete.context_size, max_tokens=800,
            ):
                yield chunk

        return StreamingResponse(stream_synthese(), media_type="text/plain")

    except ValueError as e:
            msg = str(e)
            return StreamingResponse(iter([f"❌ {msg}"]), media_type="text/plain")
    except Exception as e:
        msg = str(e)
        print(f"Erreur /chat_data_analyst : {traceback.format_exc()}")
        return StreamingResponse(iter([f"❌ Erreur : {msg}"]), media_type="text/plain")


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