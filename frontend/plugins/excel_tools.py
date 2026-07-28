# frontend/plugins/excel_tools.py - Fonctions Excel/SQL/Graphiques
# (extraites de general_purpose_chat_ui.py pour alléger le fichier principal)

import io
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://backend:8000")


# --- FONCTIONS EXCEL INTÉGRÉES (version locale comme excel_analyst_ui.py) ---
def extraire_sql_et_metadata(llm_response: str) -> tuple[str | None, dict]:
    """
    Extrait le SQL et les métadonnées de graphique d'une réponse LLM.
    Utilisé pour générer des graphiques à partir des réponses du modèle.

    Args:
        llm_response: Réponse brute du modèle LLM

    Returns:
        tuple: (sql_query, chart_metadata) où sql_query peut être None
    """
    sql_match = re.search(r"```sql\n(.*?)\n```", llm_response, re.DOTALL)
    if not sql_match:
        return None, {}

    bloc = sql_match.group(1).strip()
    chart_meta = {}
    for key in ["CHART_TYPE", "CHART_X", "CHART_Y", "CHART_TITLE", "CHART_COLOR"]:
        m = re.search(rf"--\s*{key}:\s*(.+)", bloc)
        if m:
            chart_meta[key] = m.group(1).strip()

    lignes_sql = [l for l in bloc.splitlines() if not l.strip().startswith("--")]
    sql_pur = "\n".join(lignes_sql).strip()
    return sql_pur, chart_meta


def construire_graphe(df: pd.DataFrame, meta: dict) -> go.Figure | None:
    """
    Construit un graphique localement à partir d'un DataFrame et de métadonnées.

    Args:
        df: DataFrame contenant les données à visualiser
        meta: Dictionnaire de métadonnées (CHART_TYPE, CHART_X, CHART_Y, etc.)

    Returns:
        go.Figure: Objet graphique Plotly ou None en cas d'erreur

    Types de graphiques supportés: bar, line, pie, scatter
    """
    chart_type = meta.get("CHART_TYPE", "bar").lower()
    x = meta.get("CHART_X")
    y = meta.get("CHART_Y")
    title = meta.get("CHART_TITLE", "")
    color = meta.get("CHART_COLOR")

    if x and x not in df.columns:
        x = df.columns[0] if len(df.columns) > 0 else None
    if y and y not in df.columns:
        y = df.columns[1] if len(df.columns) > 1 else None

    try:
        kwargs = dict(data_frame=df, x=x, y=y, title=title)
        if color and color in df.columns:
            kwargs["color"] = color
        if chart_type == "bar":
            return px.bar(**kwargs)
        elif chart_type == "line":
            return px.line(**kwargs)
        elif chart_type == "pie":
            return px.pie(df, names=x, values=y, title=title)
        elif chart_type == "scatter":
            return px.scatter(**kwargs)
        else:
            return px.bar(**kwargs)
    except Exception as e:
        st.warning(f"⚠️ Graphe impossible à construire : {e}")
        return None


def executer_sql_backend(sql: str) -> pd.DataFrame | None:
    """
    Exécute SQL via le backend et retourne les résultats.

    Args:
        sql: Requête SQL à exécuter

    Returns:
        pd.DataFrame: Résultats de la requête ou None en cas d'erreur
    """
    try:
        resp = requests.post(
            f"{API_URL}/execute_sql",
            json={"sql": sql, "session_id": st.session_state.session_id},
            timeout=30,
        )
        data = resp.json()
        if data.get("status") == "success":
            return pd.DataFrame(data["data"])
        else:
            st.error(f"❌ Erreur SQL : {data.get('message')}")
            return None
    except Exception as e:
        st.error(f"❌ Connexion backend : {e}")
        return None


# --- FONCTION DE PARSING EXCEL ---
def parse_and_load_excel():
    """Envoie le fichier au backend et charge les tables. Appelé après sélection de l'onglet."""
    file_bytes = io.BytesIO(st.session_state.pending_excel_file)

    try:
        resp = requests.post(
            f"{API_URL}/parse_excel",
            files={"file": (st.session_state.pending_excel_name, file_bytes)},
            params={
                "sheet_name": st.session_state.selected_sheet,
                "session_id": st.session_state.session_id,
            },
            timeout=60,
        )
        data = resp.json()

        if resp.status_code == 200 and data.get("status") == "success":
            st.session_state.tables_info = data["tables"]
            st.session_state.knowledge_ready = True
            st.session_state.excel_sheet = st.session_state.selected_sheet

            for table in data["tables"]:
                try:
                    r = requests.post(
                        f"{API_URL}/execute_sql",
                        json={
                            "sql": f'SELECT * FROM "{table["name"]}"',
                            "session_id": st.session_state.session_id,
                        },
                        timeout=30,
                    )
                    d = r.json()
                    if d.get("status") == "success":
                        st.session_state.tables_data[table["name"]] = pd.DataFrame(d["data"])
                except Exception:
                    pass
            # Utiliser la query de l'utilisateur si elle existe, sinon l'instruction
            # par défaut — dans les deux cas, ce sera traité comme dans le flux normal
            # (1 seul appel LLM, pas de message de confirmation séparé qui dupliquerait
            # ce que le LLM va lui-même répondre).
            if st.session_state.get("pending_user_query"):
                st.session_state.query_to_execute = st.session_state.pending_user_query
            else:
                st.session_state.query_to_execute = "Prends connaissance du fichier joint et attends mes instructions."
        else:
            st.error(f"❌ Erreur chargement Excel: {data.get('message', 'Erreur inconnue')}")

    except Exception as e:
        st.error(f"❌ Erreur traitement Excel: {e}")

    finally:
        st.session_state.pending_excel_file = None
        st.session_state.pending_sheet_names = []
        st.session_state.stage = 0
        st.session_state.pending_user_query = None
        # NB: query_to_execute est intentionnellement conservé ici —
        # il sera consommé par la boucle principale après le rerun.