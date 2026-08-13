# frontend/chatbot_page_utility/general_purpose_chat_ui.py

import json
import os
import random
import re
import time
import uuid

import pandas as pd
import requests
import streamlit as st

from mots_cle import COMPANY, ACRONYME, LOGO_PATH

from utility.session_state_central import SK, get, set as ss_set, init_session_state, reset_and_rerun

from plugins.excel_tools import (
    extraire_sql_et_metadata,
    construire_graphe,
    executer_sql_backend,
    parse_and_load_excel,
)

# --- CONFIGURATION GLOBALE ---

API_URL = os.environ.get("API_URL", "http://backend:8000")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "gemma4:e4b")
DEFAULT_VLM = os.environ.get("DEFAULT_VLM", "gemma4:e4b")
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 22000))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.3))
PAYLOAD_DEBUG = os.environ.get("PAYLOAD_DEBUG", "hide")



# =============================================================================
# COUCHE LLM
# =============================================================================

def _appeler_llm_et_afficher(messages_pour_api, force_new: bool = False):
    """
    Appelle le backend (chat classique ou analyste de données selon le contexte),
    stream la réponse dans un st.chat_message("assistant"), puis gère le
    post-traitement Excel (SQL/graphe/dataframe) le cas échéant.

    Args:
        messages_pour_api: Historique des messages au format attendu par l'API
        force_new: Si True, force une réponse différente (régénération).

    Returns:
        dict: Le message assistant à ajouter à l'historique.
    """
    if get(SK.KNOWLEDGE_READY):
        mode = "graphique"
        endpoint = f"{API_URL}/excel_tool/chat_data_analyst"
        temperature = 0.4
    else:
        mode = "discussion"
        endpoint = f"{API_URL}/chat"
        temperature = TEMPERATURE

    if force_new:
        temperature = min(temperature + 0.15, 1.0)

    with st.chat_message("assistant"):
        start_time = time.time()

        payload = {
            "messages": messages_pour_api,
            "modele": DEFAULT_LLM,
            "temperature": temperature,
            "context_size": CONTEXT_SIZE,
            "session_id": get(SK.SESSION_ID),
            "mode": mode,
            "think": get(SK.THINK_MODE),
            "tables_info": get(SK.TABLES_INFO),
            "request_id": str(uuid.uuid4()),
            "seed": random.randint(1, 2_147_483_647),
        }

        if not get(SK.KNOWLEDGE_READY) and get(SK.RAG_CONFIG):
            rag_config = get(SK.RAG_CONFIG)
            payload.update({
                "collection_name": rag_config.get("collection"),
                "n_results":       rag_config.get("n_results"),
                "seuil":           rag_config.get("seuil"),
                "alpha":           rag_config.get("alpha"),
                "use_hyde":        rag_config.get("use_hyde"),
                "use_expansion":   rag_config.get("use_expansion"),
                "doc_date_filter": rag_config.get("doc_date_filter"),
            })

        with st.sidebar:
            if PAYLOAD_DEBUG == "show":
                st.subheader("🔍 Debug — Payload")
                st.json(payload)
                st.caption(f"Mode: {mode} | Contexte: {CONTEXT_SIZE}")

        mes_stats = {}

        def lire_flux_api():
            try:
                with requests.post(endpoint, json=payload, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            texte = chunk.decode("utf-8")
                            if "STATS_JSON:" in texte:
                                parties = texte.split("STATS_JSON:")
                                if parties[0]:
                                    yield parties[0]
                                stats_recues = json.loads(parties[1])
                                mes_stats.update(stats_recues)
                            else:
                                yield texte
            except Exception as e:
                yield f"❌ Erreur de connexion : {str(e)}"

        with st.spinner("💬 Génération de la réponse en cours..."):
            full_response = st.write_stream(lire_flux_api())
        st.caption(f"⏱️ {time.time() - start_time:.2f}s")

        message_assistant = {
            "role": "assistant",
            "display_content": full_response,
            "content": full_response,
        }

        # Post-traitement SQL/graphe uniquement en mode Excel
        if get(SK.KNOWLEDGE_READY):
            _post_traitement_excel(full_response, message_assistant)

    return message_assistant


def _post_traitement_excel(full_response: str, message_assistant: dict):
    """
    Extrait le SQL de la réponse LLM, exécute la requête et affiche
    le graphe + dataframe. Modifie message_assistant en place.
    """
    sql, chart_meta = extraire_sql_et_metadata(full_response)
    if not (sql and chart_meta):
        return

    with st.spinner("📊 Construction du graphe..."):
        df_result = executer_sql_backend(sql)
        if df_result is None or df_result.empty:
            return

        fig = construire_graphe(df_result, chart_meta)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            message_assistant["chart_data"] = {
                "type": chart_meta.get("CHART_TYPE", "bar"),
                "data": df_result.to_dict(orient="records"),
                "layout": {
                    "x":     chart_meta.get("CHART_X"),
                    "y":     chart_meta.get("CHART_Y"),
                    "title": chart_meta.get("CHART_TITLE"),
                },
            }

        st.dataframe(df_result, use_container_width=True)
        message_assistant["dataframe"] = df_result.to_dict(orient="records")

        csv = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger (CSV)",
            data=csv,
            file_name="resultat.csv",
            mime="text/csv",
            key=f"dl_{len(get(SK.MESSAGES))}",
        )


# =============================================================================
# COUCHE UI — COMPOSANTS PARTAGÉS
# =============================================================================

def _render_sidebar():
    """Sidebar : bouton reset + aperçu du fichier Excel actuel."""
    with st.sidebar:
        if st.button("Nouvelle session", use_container_width=True):
            reset_and_rerun()

        st.divider()

        if get(SK.CURRENT_EXCEL_FILE):
            if get(SK.KNOWLEDGE_READY):
                st.success(f"📂 Fichier actuel: {get(SK.CURRENT_EXCEL_FILE)}")
                for name, df in (get(SK.TABLES_DATA) or {}).items():
                    with st.expander(f"📋 Table: {name}"):
                        st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"⚠️ Fichier chargé: {get(SK.CURRENT_EXCEL_FILE)}")
        else:
            st.info("Si vous uploadez un excel, son contenu s'affichera ici.")


def _render_historique():
    """Affiche l'historique des messages avec graphes/dataframes et bouton régénération."""
    messages = get(SK.MESSAGES)
    nb_messages = len(messages)

    for idx, message in enumerate(messages):
        with st.chat_message(message["role"]):
            raw = message.get("display_content") or message.get("content", "")
            display_content = re.sub(r"```sql\n.*?\n```\n?", "", raw, flags=re.DOTALL).strip()
            st.markdown(display_content)

            if "plot" in message:
                st.plotly_chart(message["plot"], use_container_width=True)

            if "chart_data" in message:
                df = pd.DataFrame(message["chart_data"]["data"])
                fig = construire_graphe(df, {
                    "CHART_TYPE":  message["chart_data"]["type"],
                    "CHART_X":     message["chart_data"]["layout"]["x"],
                    "CHART_Y":     message["chart_data"]["layout"]["y"],
                    "CHART_TITLE": message["chart_data"]["layout"]["title"],
                })
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            if "dataframe" in message:
                st.dataframe(pd.DataFrame(message["dataframe"]), use_container_width=True)

            if idx == nb_messages - 1 and message["role"] == "assistant":
                if st.button("🔄 Régénérer la réponse", key=f"regen_{idx}"):
                    get(SK.MESSAGES).pop()
                    ss_set(SK.REGENERATE_REQUEST, True)
                    st.rerun()


def _handle_regeneration():
    """
    Relance le LLM sur l'historique existant si une régénération a été demandée.
    L'ancien message assistant a déjà été retiré au moment du clic.
    """
    if not get(SK.REGENERATE_REQUEST):
        return

    ss_set(SK.REGENERATE_REQUEST, False)
    messages_pour_api = [
        {"role": m["role"], "content": m.get("content") or m.get("display_content", "")}
        for m in get(SK.MESSAGES)
    ]
    message_assistant = _appeler_llm_et_afficher(messages_pour_api, force_new=True)
    get(SK.MESSAGES).append(message_assistant)
    st.rerun()


# =============================================================================
# COUCHE UI — PIPELINE EXCEL
# =============================================================================

def _render_excel_selection_onglet():
    """Stage 1 — affiche le sélecteur d'onglet quand le fichier a plusieurs feuilles."""
    if get(SK.STAGE) != 1 or not get(SK.PENDING_SHEET_NAMES):
        return

    with st.chat_message("assistant"):
        st.markdown("Quel onglet voulez-vous analyser ?")
        onglet_choisi = st.radio(
            "Sélectionnez un onglet :",
            get(SK.PENDING_SHEET_NAMES),
            key="excel_sheet_choice",
            label_visibility="collapsed",
        )
        if st.button("Confirmer", key="confirm_sheet_choice"):
            ss_set(SK.SELECTED_SHEET, onglet_choisi)
            ss_set(SK.STAGE, 2)
            st.rerun()


def _render_excel_parsing():
    """Stage 2 — envoie le fichier au backend et charge les tables."""
    if get(SK.STAGE) != 2 or not get(SK.PENDING_EXCEL_FILE):
        return

    with st.spinner("⏳ Chargement du fichier Excel..."):
        parse_and_load_excel()
    st.rerun()


def _handle_excel_upload(fichier_joint, user_text: str):
    """
    Traite un fichier .xlsx uploadé :
    - 1 feuille  → stage 2 (parsing direct)
    - N feuilles → stage 1 (sélection d'onglet) + rerun immédiat
    Retourne sans rerun uniquement si le fichier est déjà connu (même file_id).
    """
    file_id = fichier_joint.name + str(fichier_joint.size)
    ss_set(SK.EXCEL_MODE, True)
    ss_set(SK.CURRENT_EXCEL_FILE, fichier_joint.name)

    if file_id == get(SK.LAST_FILE_ID):
        return  # même fichier déjà chargé, rien à faire

    # Nouveau fichier : reset de l'état Excel
    ss_set(SK.MESSAGES, [])
    ss_set(SK.KNOWLEDGE_READY, False)
    ss_set(SK.TABLES_INFO, None)
    ss_set(SK.LAST_FILE_ID, file_id)
    ss_set(SK.TABLES_DATA, {})
    ss_set(SK.SELECTED_SHEET, None)

    xls = pd.ExcelFile(fichier_joint)
    ss_set(SK.PENDING_EXCEL_FILE, fichier_joint.getbuffer().tobytes())
    ss_set(SK.PENDING_EXCEL_NAME, fichier_joint.name)
    ss_set(SK.PENDING_USER_QUERY, user_text or None)

    if len(xls.sheet_names) == 1:
        ss_set(SK.SELECTED_SHEET, xls.sheet_names[0])
        ss_set(SK.STAGE, 2)
    else:
        ss_set(SK.PENDING_SHEET_NAMES, xls.sheet_names)
        ss_set(SK.STAGE, 1)

    st.rerun()


def _restaurer_session_duckdb():
    """
    Si les bytes Excel sont en session mais que DuckDB a perdu son état
    (redémarrage backend), on re-parse silencieusement avant la prochaine requête.
    """
    if not get(SK.EXCEL_BYTES) or get(SK.KNOWLEDGE_READY):
        return

    resp = requests.post(
        f"{API_URL}/excel_tool/parse_excel",
        files={"file": (get(SK.EXCEL_NAME), get(SK.EXCEL_BYTES))},
        params={
            "sheet_name": get(SK.EXCEL_SHEET) or "Sheet1",
            "session_id": get(SK.SESSION_ID),
        },
        timeout=60,
    )
    data = resp.json()
    if resp.status_code == 200 and data.get("status") == "success":
        ss_set(SK.TABLES_INFO, data["tables"])
        ss_set(SK.KNOWLEDGE_READY, True)


# =============================================================================
# COUCHE UI — PIPELINE DISCUSSION (fichiers non-Excel)
# =============================================================================

def _handle_fichier_document(fichier_joint) -> tuple[str, str]:
    """
    Envoie un fichier non-Excel au backend et retourne
    (contenu_extrait, message_erreur).
    """
    files = {"file": (fichier_joint.name, fichier_joint.getvalue(), fichier_joint.type)}
    data = {"modele": DEFAULT_VLM}

    reponse = requests.post(f"{API_URL}/upload_fichier", files=files, data=data)
    if reponse.status_code == 200:
        return reponse.json().get("contenu", "Fichier vide."), ""
    return "", f"Erreur d'analyse pour {fichier_joint.name}"


def _handle_user_input():
    """
    Gère la saisie utilisateur :
    - query différée (post-chargement Excel) ou saisie directe
    - dispatch Excel vs documents vs texte pur
    - appel LLM final
    """
    # Query différée (produite par parse_and_load_excel après un rerun)
    deferred = get(SK.QUERY_TO_EXECUTE)
    if deferred:
        ss_set(SK.QUERY_TO_EXECUTE, None)
        user_input = deferred
    else:
        user_input = st.chat_input(
            "Votre message... (ou glissez-déposez des fichiers)",
            accept_file=True,
            file_type=["pdf", "txt", "md", "docx", "pptx", "jpg", "webp", "png", "xlsx"],
        )

    if not user_input:
        return

    # Extraction du texte brut
    if hasattr(user_input, "text"):
        user_text = user_input.text
    elif isinstance(user_input, str):
        user_text = user_input
    else:
        user_text = ""

    file_list = ""
    conversation_contexte = ""

    # --- Traitement des fichiers joints ---
    if hasattr(user_input, "files") and user_input.files:
        for fichier_joint in user_input.files:
            get(SK.PROCESSED_FILES).append(fichier_joint.name)
            file_list += f"📎 **Fichier joint :** {fichier_joint.name}\n"

            if fichier_joint.name.lower().endswith(".xlsx"):
                # Délègue entièrement au pipeline Excel (rerun inclus)
                _handle_excel_upload(fichier_joint, user_text)
            else:
                contenu, erreur = _handle_fichier_document(fichier_joint)
                if erreur:
                    st.error(erreur)
                else:
                    conversation_contexte += (
                        f"📄 **Contenu du fichier ({fichier_joint.name}) :**\n"
                        f"{contenu}\n\n---\n\n"
                    )

    # Restauration DuckDB si nécessaire avant l'appel LLM
    _restaurer_session_duckdb()

    # Construction du message utilisateur
    instruction = user_text or "Prends connaissance du fichier joint et attends mes instructions."
    display_text = f"{file_list}\n{instruction}" if file_list else instruction
    llm_text = f"{conversation_contexte} **Instruction de l'utilisateur :**\n{instruction}"

    get(SK.MESSAGES).append({
        "role": "user",
        "display_content": display_text,
        "content": llm_text,
    })

    messages_pour_api = [
        {"role": m["role"], "content": m.get("content") or m.get("display_content", "")}
        for m in get(SK.MESSAGES)
    ]

    with st.chat_message("user"):
        st.markdown(display_text)

    message_assistant = _appeler_llm_et_afficher(messages_pour_api)
    get(SK.MESSAGES).append(message_assistant)
    st.rerun()


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def render_general_purpose_chat(title=f"Chatbot {ACRONYME} Hybride"):
    """
    Point d'entrée du chatbot hybride (discussion + Excel/SQL).
    Orchestre les sous-composants dans l'ordre de rendu Streamlit.
    """
    init_session_state()

    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH)

    _render_sidebar()
    st.title(title)

    # — Affichage —
    _render_historique()

    # — Pipeline Excel (stages de chargement) —
    _render_excel_selection_onglet()
    _render_excel_parsing()

    # — Régénération —
    _handle_regeneration()

    # — Saisie & envoi —
    _handle_user_input()