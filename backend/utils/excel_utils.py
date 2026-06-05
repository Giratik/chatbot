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

def construire_graphe_from_chart_tag(llm_response: str) -> Optional[Dict[str, Any]]:
    """
    Construit un graphique à partir d'un format <chart> tag généré par le LLM.

    Args:
        llm_response: Réponse texte du modèle LLM contenant le format <chart>

    Returns:
        dict: Spécification JSON du graphique au format Plotly, ou None si le format est invalide
    """
    try:
        # Extraire le bloc chart
        chart_match = re.search(r'<chart>(.*?)</chart>', llm_response, re.DOTALL)
        if not chart_match:
            return None

        chart_content = chart_match.group(1).strip()

        # Parser les métadonnées du chart
        chart_meta = {}
        chart_data = []

        # Extraire les métadonnées
        meta_patterns = {
            'CHART_TYPE': r'CHART_TYPE:\s*(\w+)',
            'CHART_TITLE': r'CHART_TITLE:\s*(.+?)(?=\s*CHART_|$)',
            'CHART_X_AXIS_LABEL': r'CHART_X_AXIS_LABEL:\s*(.+?)(?=\s*CHART_|$)',
            'CHART_Y_AXIS_LABEL': r'CHART_Y_AXIS_LABEL:\s*(.+?)(?=\s*CHART_|$)',
        }

        for key, pattern in meta_patterns.items():
            match = re.search(pattern, chart_content)
            if match:
                chart_meta[key] = match.group(1).strip()

        # Extraire les données
        data_match = re.search(r'CHART_DATA:\s*(.+)', chart_content, re.DOTALL)
        if data_match:
            data_str = data_match.group(1).strip()
            # Parser les lignes de données
            data_lines = data_str.split('-')
            for line in data_lines:
                line = line.strip()
                if line:
                    # Extraire label et value
                    parts = line.split(':')
                    if len(parts) == 2:
                        label = parts[0].strip()
                        value = parts[1].strip()
                        try:
                            # Convertir la valeur en nombre si possible
                            numeric_value = float(value)
                            chart_data.append({"label": label, "value": numeric_value})
                        except ValueError:
                            # Garder comme string si ce n'est pas un nombre
                            chart_data.append({"label": label, "value": value})

        if not chart_data:
            return None

        # Construire la spécification du graphique
        chart_type = chart_meta.get('CHART_TYPE', 'bar').lower()
        x_title = chart_meta.get('CHART_X_AXIS_LABEL', 'Catégorie')
        y_title = chart_meta.get('CHART_Y_AXIS_LABEL', 'Valeur')
        title = chart_meta.get('CHART_TITLE', 'Graphique')

        # Convertir les données au format attendu
        df_data = pd.DataFrame({
            x_title: [item["label"] for item in chart_data],
            y_title: [item["value"] for item in chart_data]
        })

        # Construire la spécification
        chart_spec = {
            "type": chart_type,
            "data": df_data.to_dict(orient='records'),
            "layout": {
                "xaxis": {"title": x_title},
                "yaxis": {"title": y_title},
                "title": title
            }
        }

        return chart_spec

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Graphe impossible à construire à partir du format chart: {str(e)}")

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
        from core.duckdb_session import registry

        # Obtenir la session DuckDB
        session = registry.get_or_raise(session_id)
        result = session.query_to_records(sql)
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
