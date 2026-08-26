# frontend/utility/session_state_central.py

from dataclasses import dataclass, field
from typing import Any
import streamlit as st
import uuid
import requests
import os

API_URL = os.environ.get("API_URL", "http://backend:8000")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "gemma4:e4b")

# ─── Constantes (noms des clés) ───────────────────────────────────────────────

class SK:
    """Session Keys — toutes les clés en un seul endroit."""
    # Identité & conversation
    SESSION_ID         = "session_id"
    MESSAGES           = "messages"
    THINK_MODE         = "think_mode"

    # RAG
    RAG_CONFIG         = "rag_config"

    # Fichiers génériques
    PROCESSED_FILES    = "processed_files"
    LAST_FILE_ID       = "last_file_id"

    # Excel
    EXCEL_MODE         = "excel_mode"
    EXCEL_BYTES        = "excel_bytes"
    EXCEL_NAME         = "excel_name"
    EXCEL_SHEET        = "excel_sheet"
    CURRENT_EXCEL_FILE = "current_excel_file"
    TABLES_INFO        = "tables_info"
    TABLES_DATA        = "tables_data"

    # Workflow Excel (pending / staged)
    STAGE              = "stage"
    SELECTED_SHEET     = "selected_sheet"
    PENDING_EXCEL_FILE = "pending_excel_file"
    PENDING_EXCEL_NAME = "pending_excel_name"
    PENDING_SHEET_NAMES= "pending_sheet_names"
    PENDING_USER_QUERY = "pending_user_query"
    QUERY_TO_EXECUTE   = "query_to_execute"
    REGENERATE_REQUEST = "regenerate_request"

    # Divers
    KNOWLEDGE_READY    = "knowledge_ready"


# ─── Valeurs par défaut ───────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    SK.SESSION_ID:          None,
    SK.MESSAGES:            list,       # ✅ factory — reset_and_rerun() appelle list()
    SK.THINK_MODE:          False,
    SK.RAG_CONFIG:          lambda: {   # ✅ factory pour le dict avec valeurs par défaut
        "collection": "aucune_collection",
        "model": DEFAULT_LLM,
        "doc_date_filter": "",
        "n_results": 250,
        "seuil": 0.6,
        "use_hyde": True,
        "use_expansion": True,
        "alpha": 0.5,
    },
    SK.PROCESSED_FILES:     list,
    SK.LAST_FILE_ID:        None,
    SK.EXCEL_MODE:          False,
    SK.EXCEL_BYTES:         None,
    SK.EXCEL_NAME:          None,
    SK.EXCEL_SHEET:         None,
    SK.CURRENT_EXCEL_FILE:  None,
    SK.TABLES_INFO:         None,
    SK.TABLES_DATA:         dict,       # ✅ factory
    SK.STAGE:               0,
    SK.SELECTED_SHEET:      None,
    SK.PENDING_EXCEL_FILE:  None,
    SK.PENDING_EXCEL_NAME:  None,
    SK.PENDING_SHEET_NAMES: list,       # ✅ factory
    SK.PENDING_USER_QUERY:  None,
    SK.QUERY_TO_EXECUTE:    None,
    SK.REGENERATE_REQUEST:  False,
    SK.KNOWLEDGE_READY:     False,
}


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_session_state() -> None:
    """À appeler une seule fois au démarrage (Main.py)."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            # On ajoute la même vérification que dans reset_and_rerun()
            st.session_state[key] = default() if callable(default) else default

    if st.session_state[SK.SESSION_ID] is None:
        st.session_state[SK.SESSION_ID] = str(uuid.uuid4())


# ─── Accesseurs typés ─────────────────────────────────────────────────────────
# Optionnel mais pratique : autocomplétion + lecture claire dans les autres fichiers.

def get(key: str) -> Any:
    return st.session_state.get(key)

def set(key: str, value: Any) -> None:
    st.session_state[key] = value

def reset_excel_state() -> None:
    """Remet à zéro tout le workflow Excel d'un coup."""
    for key in (
        SK.EXCEL_MODE, SK.EXCEL_BYTES, SK.EXCEL_NAME, SK.EXCEL_SHEET,
        SK.CURRENT_EXCEL_FILE, SK.TABLES_INFO, SK.TABLES_DATA,
        SK.STAGE, SK.SELECTED_SHEET,
        SK.PENDING_EXCEL_FILE, SK.PENDING_EXCEL_NAME,
        SK.PENDING_SHEET_NAMES, SK.PENDING_USER_QUERY,
        SK.QUERY_TO_EXECUTE, SK.KNOWLEDGE_READY,
    ):
        st.session_state[key] = _DEFAULTS[key]

def reset_conversation() -> None:
    """Efface la conversation sans toucher à la config."""
    st.session_state[SK.MESSAGES] = []
    st.session_state[SK.REGENERATE_REQUEST] = False
    st.session_state[SK.QUERY_TO_EXECUTE] = None


def reset_and_rerun() -> None:
    """Réinitialise complètement la session côté backend + frontend."""
    session_id = st.session_state.get(SK.SESSION_ID)
    if session_id:
        try:
            requests.delete(f"{API_URL}/excel_tool/session/{session_id}", timeout=3)
        except Exception:
            pass

    # Remet toutes les clés à leur valeur par défaut — aucun oubli possible
    for key, default in _DEFAULTS.items():
        # Les listes/dicts sont mutables : on recrée l'objet pour éviter
        # que deux sessions partagent la même référence
        st.session_state[key] = default() if callable(default) else default

    st.session_state[SK.SESSION_ID] = str(uuid.uuid4())
    st.rerun()