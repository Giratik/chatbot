"""
Module : Utilitaires Excel pour le Backend
Description : Fonctions utilitaires pour le traitement des données Excel et la génération de graphiques.
"""

import re
import pandas as pd
from typing import Optional, Dict, Any
import json
import plotly.express as px
import plotly.graph_objects as go
from fastapi import HTTPException

def extraire_sql_et_metadata(llm_response: str) -> tuple[Optional[str], Dict[str, str]]:
    """
    Extrait le code SQL et les métadonnées de graphique de la réponse LLM.

    Args:
        llm_response: Réponse texte du modèle LLM contenant potentiellement du SQL et des métadonnées

    Returns:
        tuple: (sql_pur, chart_meta) où sql_pur est la requête SQL extraite et chart_meta est un dictionnaire
               des métadonnées de graphique (CHART_TYPE, CHART_X, CHART_Y, CHART_TITLE, CHART_COLOR)
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

def construire_graphe(df: pd.DataFrame, meta: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Construit un graphique à partir des données et des métadonnées et retourne une spécification JSON.

    Args:
        df: DataFrame pandas contenant les données à visualiser
        meta: Dictionnaire de métadonnées de graphique

    Returns:
        dict: Spécification JSON du graphique au format Plotly, ou None en cas d'erreur
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

    if x is None or y is None:
        return None

    try:
        # Convertir le DataFrame en format JSON sérialisable
        data_json = df.to_dict(orient='records')

        # Construire la spécification du graphique
        chart_spec = {
            "type": chart_type,
            "data": data_json,
            "layout": {
                "xaxis": {"title": x},
                "yaxis": {"title": y},
                "title": title
            }
        }

        # Ajouter les informations de couleur si spécifiée
        if color and color in df.columns:
            chart_spec["color"] = color

        return chart_spec

    except Exception as e:
        # En backend, on lève une exception plutôt que d'utiliser st.warning
        raise HTTPException(status_code=400, detail=f"Graphe impossible à construire : {str(e)}")

def executer_sql_backend(sql: str, session_id: str) -> pd.DataFrame:
    """
    Exécute une requête SQL via le backend DuckDB et retourne les résultats.

    Args:
        sql: Requête SQL à exécuter
        session_id: Identifiant de session pour le contexte DuckDB

    Returns:
        pd.DataFrame: DataFrame contenant les résultats de la requête

    Raises:
        HTTPException: En cas d'erreur d'exécution SQL
    """
    try:
        # Import local pour éviter les dépendances circulaires
        from core.duckdb_session import execute_sql_query

        result = execute_sql_query(sql, session_id)
        return pd.DataFrame(result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur SQL : {str(e)}")

def analyser_reponse_excel(llm_response: str) -> Dict[str, Any]:
    """
    Analyse une réponse LLM pour extraire les composants Excel (SQL, métadonnées, graphique).

    Args:
        llm_response: Réponse complète du modèle LLM

    Returns:
        dict: Analyse structurée contenant sql, chart_meta, et chart_spec si disponible
    """
    sql, chart_meta = extraire_sql_et_metadata(llm_response)

    result = {
        "sql": sql,
        "chart_meta": chart_meta,
        "chart_spec": None
    }

    return result
