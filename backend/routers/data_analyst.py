# backend/routers/data_analyst.py

"""
Routeur API : Assistant Data Analyst
Description : Gère les requêtes spécifiques à l'analyse de fichiers structurés (génération de graphiques, 
                requêtes SQL/DuckDB sur les fichiers importés).
"""

import re
import traceback
import duckdb
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse

import core.duckdb_session as ddb
from services.ollama_client import inferring_ollama
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from utils.excel_utils import extraire_sql_et_metadata, construire_graphe, executer_sql_backend, analyser_reponse_excel

router = APIRouter(tags=["Data Analyst"])


class ChatRequest_csv(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    session_id: str = "default"
    mode: str = "discussion"
    think: bool = False

class SqlRequest(BaseModel):
    sql: str
    session_id: str = "default"

class SessionRequest(BaseModel):
    session_id: str = "default"

class SqlExtractionRequest(BaseModel):
    llm_response: str

class ChartBuildingRequest(BaseModel):
    data: List[Dict[str, Any]]
    chart_meta: Dict[str, str]

class SqlExecutionRequest(BaseModel):
    sql: str
    session_id: str = "default"

class ExcelAnalysisRequest(BaseModel):
    llm_response: str

@router.post("/parse_excel")
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


@router.post("/chat_data_analyst")
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
            think=requete.think,
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
            think=requete.think,
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
                think=requete.think,
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


@router.post("/execute_sql")
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


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Libère explicitement la session DuckDB (fin de conversation)."""
    ddb.registry.delete(session_id)
    return {"status": "success", "message": f"Session '{session_id}' supprimée."}


@router.get("/sessions/count")
async def sessions_count():
    """Nombre de sessions DuckDB actives (monitoring)."""
    return {"active_sessions": ddb.registry.active_count()}

# --- NOUVELLES ROUTES POUR LES UTILITAIRES EXCEL ---

@router.post("/extract_sql_metadata")
async def extract_sql_metadata(request: SqlExtractionRequest):
    """
    Extrait le SQL et les métadonnées de graphique d'une réponse LLM.

    Args:
        llm_response: Réponse texte du modèle LLM

    Returns:
        dict: Contient sql (str) et chart_meta (dict)
    """
    try:
        sql, chart_meta = extraire_sql_et_metadata(request.llm_response)
        return {
            "status": "success",
            "sql": sql,
            "chart_meta": chart_meta
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/build_chart")
async def build_chart(request: ChartBuildingRequest):
    """
    Construit une spécification de graphique à partir de données et de métadonnées.

    Args:
        data: Données au format JSON (liste de dictionnaires)
        chart_meta: Métadonnées de graphique

    Returns:
        dict: Spécification de graphique au format JSON
    """
    try:
        # Convertir les données JSON en DataFrame
        df = pd.DataFrame(request.data)

        # Construire le graphique
        chart_spec = construire_graphe(df, request.chart_meta)

        if chart_spec:
            return {
                "status": "success",
                "chart_spec": chart_spec
            }
        else:
            return {
                "status": "error",
                "message": "Impossible de construire le graphique avec les données et métadonnées fournies"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/execute_sql_excel")
async def execute_sql_excel(request: SqlExecutionRequest):
    """
    Exécute une requête SQL et retourne les résultats pour construction de graphique.

    Args:
        sql: Requête SQL à exécuter
        session_id: Identifiant de session DuckDB

    Returns:
        dict: Contient les données et métadonnées pour construction de graphique
    """
    try:
        # Exécuter la requête SQL
        df = executer_sql_backend(request.sql, request.session_id)

        return {
            "status": "success",
            "n_rows": len(df),
            "data": df.to_dict(orient='records'),
            "columns": list(df.columns)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/analyze_excel_response")
async def analyze_excel_response(request: ExcelAnalysisRequest):
    """
    Analyse une réponse LLM complète pour extraire tous les composants Excel.

    Args:
        llm_response: Réponse complète du modèle LLM

    Returns:
        dict: Analyse structurée avec SQL, métadonnées et spécification de graphique
    """
    try:
        result = analyser_reponse_excel(request.llm_response)
        return {
            "status": "success",
            **result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
