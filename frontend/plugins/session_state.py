# frontend/plugins/session_state.py - Gestion de l'état de session (chat + Excel)

import os
import uuid
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://backend:8000")


# --- GESTION DE SESSION UNIFIÉE ---
def init_session_state():
    """
    Initialise l'état de session pour le chat hybride.
    Gère à la fois les variables de session standard et Excel.
    """
    # Variables de session standard (chat)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "think_mode" not in st.session_state:
        st.session_state.think_mode = False

    # Variables de session Excel (ajoutées pour la compatibilité Excel)
    if "tables_info" not in st.session_state:
        st.session_state.tables_info = None
    if "knowledge_ready" not in st.session_state:
        st.session_state.knowledge_ready = False
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None
    if "tables_data" not in st.session_state:
        st.session_state.tables_data = {}
    if "excel_mode" not in st.session_state:
        st.session_state.excel_mode = False
    if "current_excel_file" not in st.session_state:
        st.session_state.current_excel_file = None
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    if "selected_sheet" not in st.session_state:
        st.session_state.selected_sheet = None

    if 'pending_excel_file' not in st.session_state:
        st.session_state.pending_excel_file = None  # stocke les bytes du fichier en attente
    if 'pending_excel_name' not in st.session_state:
        st.session_state.pending_excel_name = None
    if 'pending_sheet_names' not in st.session_state:
        st.session_state.pending_sheet_names = []

    if 'pending_user_query' not in st.session_state:
        st.session_state.pending_user_query = None
    if 'query_to_execute' not in st.session_state:
        st.session_state.query_to_execute = None
    if 'regenerate_request' not in st.session_state:
        st.session_state.regenerate_request = False

    # Variables de session RAG (ajoutées pour la configuration RAG)
    if "rag_config" not in st.session_state:
        # Initialisation avec des valeurs par défaut
        st.session_state.rag_config = {
            "collection": "aucune_collection",
            "model": "gemma4:e4b",
            "doc_date_filter": "",
            "n_results": 250,
            "seuil": 0.6,
            "use_hyde": True,
            "use_expansion": True,
            "alpha": 0.5,
        }


def reset_and_rerun():
    """
    Réinitialise complètement la session.
    Supprime toutes les données de session et recharge la page.
    """
    if "session_id" in st.session_state:
        try:
            requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=3)
        except Exception:
            pass
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.processed_files = []
    st.session_state.tables_info = None
    st.session_state.knowledge_ready = False
    st.session_state.last_file_id = None
    st.session_state.tables_data = {}
    st.session_state.excel_mode = False
    st.session_state.current_excel_file = None
    st.session_state.excel_bytes = None
    st.session_state.excel_name = None
    st.session_state.excel_sheet = None
    st.session_state.stage = 0
    st.session_state.selected_sheet = None
    st.rerun()