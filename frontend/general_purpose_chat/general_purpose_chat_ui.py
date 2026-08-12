# frontend/general_purpose_chat/general_purpose_chat_ui.py

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

# Nouveau : import depuis le module centralisé
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


# --- APPEL DU LLM ET TRAITEMENT DE LA RÉPONSE (RÉUTILISABLE POUR LA RÉGÉNÉRATION) ---
def _appeler_llm_et_afficher(messages_pour_api, force_new: bool = False):
    """
    Appelle le backend (chat classique ou analyste de données selon le contexte),
    stream la réponse dans un st.chat_message("assistant"), puis gère le
    post-traitement Excel (SQL/graphe/dataframe) le cas échéant.

    Args:
        messages_pour_api: Historique des messages au format attendu par l'API
        force_new: Si True (cas de la régénération), on force une réponse
            différente de la précédente en variant le seed et en augmentant
            légèrement la température, et on ajoute un identifiant unique
            pour éviter tout cache éventuel côté backend/proxy.

    Returns:
        dict: Le message assistant à ajouter à st.session_state.messages
    """
    # Déterminer le mode de traitement
    if get(SK.KNOWLEDGE_READY):
        mode = "graphique"
        endpoint = f"{API_URL}/chat_data_analyst"
        temperature = 0.4
    else:
        mode = "discussion"
        endpoint = f"{API_URL}/chat"
        temperature = TEMPERATURE


    if force_new:
        # Légère hausse de température pour favoriser une réponse différente
        temperature = min(temperature + 0.15, 1.0)

    with st.chat_message("assistant"):
        start_time = time.time()

        payload = {
            "messages": messages_pour_api,
            "modele": DEFAULT_LLM,
            "temperature": temperature,
            "context_size": CONTEXT_SIZE,
            "session_id": get(SK.SESSION_ID),           # ✅
            "mode": mode,
            "think": get(SK.THINK_MODE),                # ✅
            "tables_info": get(SK.TABLES_INFO),         # ✅
            "request_id": str(uuid.uuid4()),
            "seed": random.randint(1, 2_147_483_647),
        }

        # Add RAG parameters if available and not in Excel mode
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
                st.caption(f"Mode: {mode} | Contexte: {context_size}")

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

        # --- TRAITEMENT POST-RÉPONSE (EXCEL) ---
        message_assistant = {
            "role": "assistant",
            "display_content": full_response,
            "content": full_response,
        }

        # Si nous sommes en mode Excel et que des métadonnées SQL sont présentes
        if get(SK.KNOWLEDGE_READY):                     # ✅
            sql, chart_meta = extraire_sql_et_metadata(full_response)
            if sql and chart_meta:
                with st.spinner("📊 Construction du graphe..."):
                    df_result = executer_sql_backend(sql)
                    if df_result is not None and not df_result.empty:
                        fig = construire_graphe(df_result, chart_meta)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            # Stocker les données pour le graphe plutôt que l'objet Figure
                            message_assistant["chart_data"] = {
                                "type": chart_meta.get("CHART_TYPE", "bar"),
                                "data": df_result.to_dict(orient='records'),
                                "layout": {
                                    "x": chart_meta.get("CHART_X"),
                                    "y": chart_meta.get("CHART_Y"),
                                    "title": chart_meta.get("CHART_TITLE")
                                }
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

    return message_assistant


# --- FONCTION PRINCIPALE DE CHAT HYBRIDE ---
def render_general_purpose_chat(title=f"Chatbot {ACRONYME} Hybride"):
    """
    Interface de chat avancée avec support Excel et SQL intégré.
    Fonction principale utilisée par Main.py pour le chatbot généraliste.

    Args:
        title: Titre à afficher pour l'interface de chat
    """
    init_session_state()
    #with st.sidebar:
    #    st.caption(f"Session : {st.session_state[SK.SESSION_ID][:8]}")

    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH)

    # --- SIDEBAR UNIFIÉ ---
    with st.sidebar:
        if st.button("Nouvelle session", use_container_width=True):
            reset_and_rerun()

        st.divider()

        # ✅ Avant : st.session_state.current_excel_file
        if get(SK.CURRENT_EXCEL_FILE):
            if get(SK.KNOWLEDGE_READY):
                st.success(f"📂 Fichier actuel: {get(SK.CURRENT_EXCEL_FILE)}")
                if get(SK.TABLES_DATA):
                    for name, df in get(SK.TABLES_DATA).items():
                        with st.expander(f"📋 Table: {name}"):
                            st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"⚠️ Fichier chargé: {get(SK.CURRENT_EXCEL_FILE)}")
        else:
            st.info("Si vous uploadez un excel, son contenu s'affichera ici.")

    st.title(title)

    # 1. AFFICHAGE DE L'HISTORIQUE AVEC SUPPORT EXCEL
    # Dans la boucle d'affichage de l'historique
    messages = get(SK.MESSAGES)                        # ✅ — une seule lecture
    nb_messages = len(messages)
    for idx, message in enumerate(get(SK.MESSAGES)):
        with st.chat_message(message["role"]):
            raw = message.get("display_content") or message.get("content", "")
            display_content = re.sub(r"```sql\n.*?\n```\n?", "", raw, flags=re.DOTALL).strip()
            st.markdown(display_content)

            if "plot" in message:
                st.plotly_chart(message["plot"], use_container_width=True)

            # ← Ajouter : reconstruire depuis chart_data
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

            # --- BOUTON DE RÉGÉNÉRATION (uniquement sur le dernier message assistant) ---
            # Permet de relancer la génération si la réponse du LLM s'est arrêtée
            # en plein milieu ou n'a pas convenu à l'utilisateur.
            if idx == nb_messages - 1 and message["role"] == "assistant":
                if st.button("🔄 Régénérer la réponse", key=f"regen_{idx}"):
                    get(SK.MESSAGES).pop()             # ✅ (liste mutée en place)
                    ss_set(SK.REGENERATE_REQUEST, True)
                    st.rerun()

    # --- SÉLECTION D'ONGLET EXCEL ---
    if get(SK.STAGE) == 1 and get(SK.PENDING_SHEET_NAMES):
        with st.chat_message("assistant"):
            st.markdown("Quel onglet voulez-vous analyser ?")
            onglet_choisi = st.radio("Sélectionnez un onglet :",
                                     get(SK.PENDING_SHEET_NAMES),
                                     key="excel_sheet_choice",
                                     label_visibility="collapsed")
            if st.button("Confirmer", key="confirm_sheet_choice"):
                ss_set(SK.SELECTED_SHEET, onglet_choisi)
                ss_set(SK.STAGE, 2)
                st.rerun()

    # --- PARSING EXCEL ---
    if get(SK.STAGE) == 2 and get(SK.PENDING_EXCEL_FILE):   # ✅
        with st.spinner("⏳ Chargement du fichier Excel..."):
            parse_and_load_excel()
        st.rerun()

    # --- RÉGÉNÉRATION DE LA DERNIÈRE RÉPONSE ---
    # Déclenchée par le bouton "🔄 Régénérer la réponse" : on relance l'appel LLM
    # avec l'historique existant (qui se termine déjà par le dernier message
    # utilisateur, l'ancienne réponse assistant ayant été retirée au clic).
    if get(SK.REGENERATE_REQUEST):                     # ✅
        ss_set(SK.REGENERATE_REQUEST, False)
        messages_pour_api = [
            {"role": m["role"], "content": m.get("content") or m.get("display_content", "")}
            for m in get(SK.MESSAGES)
        ]
        message_assistant = _appeler_llm_et_afficher(messages_pour_api, force_new=True)
        get(SK.MESSAGES).append(message_assistant)
        st.rerun()

    # 2. SAISIE UTILISATEUR AVEC SUPPORT FICHIERS ÉTENDU
    # Si une query a été mise en attente pendant le chargement Excel multi-onglets,
    # on la récupère et on la traite comme si l'utilisateur venait de la saisir.
    _deferred_query = get(SK.QUERY_TO_EXECUTE)         # ✅
    if _deferred_query:
        ss_set(SK.QUERY_TO_EXECUTE, None)
        user_input = _deferred_query
    else:
        user_input = st.chat_input(
            "Votre message... (ou glissez-déposez des fichiers)",
            accept_file=True,
            file_type=["pdf", "txt", "md", "docx", "pptx", "jpg", "webp", "png", "xlsx"],
        )

    # 3. TRAITEMENT DE L'ENTRÉE UTILISATEUR
    if user_input:
        file_list = ""
        conversation_contexte = ""
        nom_fichiers = []
        contenu_fichiers = []
        excel_processed = False

        # --- A. GESTION DES FICHIERS AVEC DÉTECTION EXCEL ---
        if hasattr(user_input, "files") and user_input.files:
            for fichier_joint in user_input.files:
                file_id = fichier_joint.name + str(fichier_joint.size)
                get(SK.PROCESSED_FILES).append(fichier_joint.name)  # ✅
                file_list += f"📎 **Fichier joint :** {fichier_joint.name}\n"

                # DÉTECTION ET TRAITEMENT SPÉCIFIQUE EXCEL
                if fichier_joint.name.lower().endswith('.xlsx'):
                    ss_set(SK.EXCEL_MODE, True)
                    ss_set(SK.CURRENT_EXCEL_FILE, fichier_joint.name)

                    if file_id != get(SK.LAST_FILE_ID):
                        ss_set(SK.MESSAGES, [])
                        ss_set(SK.KNOWLEDGE_READY, False)
                        ss_set(SK.TABLES_INFO, None)
                        ss_set(SK.LAST_FILE_ID, file_id)
                        ss_set(SK.TABLES_DATA, {})
                        ss_set(SK.SELECTED_SHEET, None)

                        xls = pd.ExcelFile(fichier_joint)
                        if len(xls.sheet_names) == 1:
                            ss_set(SK.SELECTED_SHEET, xls.sheet_names[0])
                            ss_set(SK.PENDING_EXCEL_FILE, fichier_joint.getbuffer().tobytes())
                            ss_set(SK.PENDING_EXCEL_NAME, fichier_joint.name)
                            ss_set(SK.PENDING_USER_QUERY, user_input.text or None)
                            ss_set(SK.STAGE, 2)
                        else:
                            ss_set(SK.PENDING_EXCEL_FILE, fichier_joint.getbuffer().tobytes())
                            ss_set(SK.PENDING_EXCEL_NAME, fichier_joint.name)
                            ss_set(SK.PENDING_SHEET_NAMES, xls.sheet_names)
                            ss_set(SK.PENDING_USER_QUERY, user_input.text or None)
                            ss_set(SK.STAGE, 1)
                            st.rerun()

                    excel_processed = True
                    # NB : le chargement effectif (parsing, tables, appel LLM) est
                    # entièrement géré par le pipeline stage/pending_* ci-dessus
                    # (cas 1 feuille → stage=2 ; cas multi-feuilles → stage=1 puis
                    # rerun déjà effectué). On ne refait pas le parsing ici pour
                    # éviter un double traitement et une double réponse du LLM.
                    st.rerun()

                # TRAITEMENT DES AUTRES TYPES DE FICHIERS (comme dans chat_ui.py original)
                else:
                    files = {"file": (fichier_joint.name, fichier_joint.getvalue(), fichier_joint.type)}
                    data = {"modele": DEFAULT_VLM}

                    reponse = requests.post(f"{API_URL}/upload_fichier", files=files, data=data)

                    if reponse.status_code == 200:
                        contenu_extrait = reponse.json().get("contenu", "Fichier vide.")
                        contenu_fichiers.append(contenu_extrait)
                        nom_fichiers.append(fichier_joint.name)
                        conversation_contexte += f"📄 **Contenu du fichier ({fichier_joint.name}) :**\n{contenu_extrait}\n\n---\n\n"
                    else:
                        st.error(f"Erreur d'analyse pour {fichier_joint.name}")

        # --- B. GESTION DU TEXTE ET SELECTION DU MODE ---
        user_text = ""
        if hasattr(user_input, "text"):
            user_text = user_input.text
        elif isinstance(user_input, str):
            user_text = user_input

        # Restauration automatique si session DuckDB perdue
        if get(SK.EXCEL_BYTES) and not get(SK.KNOWLEDGE_READY):  # ✅
            resp = requests.post(
                f"{API_URL}/parse_excel",
                files={"file": (get(SK.EXCEL_NAME), get(SK.EXCEL_BYTES))},
                params={"sheet_name": get(SK.EXCEL_SHEET) or "Sheet1",
                        "session_id": get(SK.SESSION_ID)},
                timeout=60,
            )
            data = resp.json()
            if resp.status_code == 200 and data.get("status") == "success":
                ss_set(SK.TABLES_INFO, data["tables"])
                ss_set(SK.KNOWLEDGE_READY, True)

        # Utiliser le texte de l'utilisateur directement (comme dans excel_analyst_ui.py)
        instruction = user_text if user_text else "Prends connaissance du fichier joint et attends mes instructions."

        display_text = f"{file_list}\n{instruction}" if file_list else instruction
        llm_text = f"{conversation_contexte} **Instruction de l'utilisateur :**\n{instruction}"

        get(SK.MESSAGES).append({
            "role": "user",
            "display_content": display_text,
            "content": llm_text
        })

        messages_pour_api = [
    {"role": m["role"], "content": m.get("content") or m.get("display_content", "")}
    for m in get(SK.MESSAGES)  # ✅
]

        with st.chat_message("user"):
            st.markdown(display_text)

        # --- C. APPEL DU CHATBOT AVEC LE BON ENDPOINT ---
        message_assistant = _appeler_llm_et_afficher(messages_pour_api)
        get(SK.MESSAGES).append(message_assistant)
        st.rerun()