# chat_ui_v2.py - Chat Hybride avec Support Excel et SQL
"""
Version avancée du chat qui intègre:
- Conversation standard (comme chat_ui.py)
- Traitement des fichiers Excel (comme excel_analyst_ui.py)
- Exécution SQL et génération de graphiques
- Gestion unifiée de session
"""

import json
import streamlit as st
import requests
import uuid
import time
import re
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION GLOBALE ---
LOGO_PATH = "ressource/Eau_de_Paris_bleu.svg.png"
API_URL = os.environ.get("API_URL", "http://backend:8000")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "ministral-3:14b")
DEFAULT_VLM = os.environ.get("DEFAULT_VLM", "ministral-3:14b")
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 12288))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.3))
PAYLOAD_DEBUG = os.environ.get("PAYLOAD_DEBUG", "hide")

# --- GESTION DE SESSION UNIFIÉE ---
def init_session_state():
    """Initialise l'état de session pour le chat hybride."""
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

def reset_and_rerun():
    """Réinitialise complètement la session."""
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
    st.rerun()

# --- FONCTIONS EXCEL INTÉGRÉES (version locale comme excel_analyst_ui.py) ---
def extraire_sql_et_metadata(llm_response: str) -> tuple[str | None, dict]:
    """Extrait le SQL et les métadonnées de graphique - version locale."""
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
    """Construit un graphique localement - version simplifiée."""
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
    """Exécute SQL via le backend - version simplifiée."""
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

# --- FONCTION PRINCIPALE DE CHAT HYBRIDE ---
def render_general_purpose_chat(title="Chatbot EDP Hybride"):
    """
    Interface de chat avancée avec support Excel et SQL intégré.
    """
    init_session_state()

    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH)

    # --- SIDEBAR UNIFIÉ ---
    with st.sidebar:
        st.title("💬📊 Chat Hybride")

        # Contrôles standard
        st.session_state.think_mode = st.toggle(
            "Mode raisonnement",
            value=st.session_state.think_mode,
            help="Active le mode 'raisonnement' des modèles"
        )

        if st.button("Nouvelle session", use_container_width=True):
            reset_and_rerun()

        st.divider()

        # Section Excel (toujours visible mais désactivée si pas de fichier)
        st.subheader("📊 Analyse Excel")

        # Afficher l'état actuel du fichier
        if st.session_state.current_excel_file:
            if st.session_state.knowledge_ready:
                st.success(f"📂 Fichier actuel: {st.session_state.current_excel_file}")
                st.caption("Prêt pour l'analyse SQL et graphiques")

                # Affichage des tables dans le sidebar
                if st.session_state.get("tables_data"):
                    for name, df in st.session_state.tables_data.items():
                        with st.expander(f"📋 Table: {name}"):
                            st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning(f"⚠️ Fichier chargé: {st.session_state.current_excel_file}")
                st.caption("Chargement en cours...")
        else:
            st.info("📌 Chargez un fichier Excel pour activer l'analyse de données")

        st.divider()
        st.caption("© Eau de Paris - Chatbot Avancé")

    st.title(title)

    # 1. AFFICHAGE DE L'HISTORIQUE AVEC SUPPORT EXCEL
# Dans la boucle d'affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            display_content = re.sub(r"```sql\n.*?\n```\n?", "", message["display_content"], flags=re.DOTALL).strip()
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

    # 2. SAISIE UTILISATEUR AVEC SUPPORT FICHIERS ÉTENDU
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
                st.session_state.processed_files.append(fichier_joint.name)
                file_list += f"📎 **Fichier joint :** {fichier_joint.name}\n"

                # DÉTECTION ET TRAITEMENT SPÉCIFIQUE EXCEL
                if fichier_joint.name.lower().endswith('.xlsx'):
                    st.session_state.excel_mode = True
                    st.session_state.current_excel_file = fichier_joint.name
                    excel_processed = True

                    # Initialiser la session Excel
                    if file_id != st.session_state.get("last_file_id"):
                        st.session_state.messages = []
                        st.session_state.knowledge_ready = False
                        st.session_state.tables_info = None
                        st.session_state.last_file_id = file_id
                        st.session_state.tables_data = {}

                    try:
                        # Charger et analyser le fichier Excel
                        with st.spinner("⏳ Chargement du fichier Excel..."):
                            xls = pd.ExcelFile(fichier_joint)
                            onglet_choisi = xls.sheet_names[0]  # Utiliser la première feuille par défaut

                            # Envoyer au backend pour parsing
                            fichier_joint.seek(0)
                            resp = requests.post(
                                f"{API_URL}/parse_excel",
                                files={"file": (fichier_joint.name, fichier_joint.getbuffer())},
                                params={
                                    "sheet_name": onglet_choisi,
                                    "session_id": st.session_state.session_id,
                                },
                                timeout=60,
                            )
                            data = resp.json()

                            if resp.status_code == 200 and data.get("status") == "success":
                                st.session_state.tables_info = data["tables"]
                                st.session_state.knowledge_ready = True
                                st.session_state.excel_bytes = fichier_joint.getbuffer().tobytes()  # ← ajouter
                                st.session_state.excel_name = fichier_joint.name                    # ← ajouter
                                st.session_state.excel_sheet = onglet_choisi   
                                # Charger les données des tables
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

                                st.success(f"✅ Fichier Excel chargé: {len(data['tables'])} table(s) détectée(s)")
                                conversation_contexte += f"📊 **Données Excel chargées :** {fichier_joint.name} - {len(data['tables'])} table(s) disponibles pour analyse SQL\n\n"

                            else:
                                st.error(f"❌ Erreur chargement Excel: {data.get('message', 'Erreur inconnue')}")
                                conversation_contexte += f"⚠️ **Erreur chargement Excel :** {fichier_joint.name}\n\n"

                    except Exception as e:
                        st.sidebar.error(f"❌ Erreur traitement Excel: {e}")
                        conversation_contexte += f"⚠️ **Erreur traitement Excel :** {fichier_joint.name} - {str(e)}\n\n"
                        excel_processed = False

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

        #st.sidebar.write(f"excel_bytes={bool(st.session_state.get('excel_bytes'))} knowledge_ready={st.session_state.knowledge_ready}")


        # Restauration automatique si session DuckDB perdue
        if st.session_state.get("excel_bytes") and not st.session_state.knowledge_ready:
            with st.spinner("🔄 Restauration de la session Excel..."):
                resp = requests.post(
                    f"{API_URL}/parse_excel",
                    files={"file": (st.session_state.excel_name, st.session_state.excel_bytes)},
                    params={
                        "sheet_name": st.session_state.get("excel_sheet", "Sheet1"),
                        "session_id": st.session_state.session_id,
                    },
                    timeout=60,
                )
                data = resp.json()
                if resp.status_code == 200 and data.get("status") == "success":
                    st.session_state.tables_info = data["tables"]
                    st.session_state.knowledge_ready = True

        # Déterminer le mode de traitement
        if st.session_state.knowledge_ready:
            mode = "graphique"
            endpoint = f"{API_URL}/chat_data_analyst"
            context_size = 30000
            temperature = 0.4
        else:
            mode = "discussion"
            endpoint = f"{API_URL}/chat"
            context_size = CONTEXT_SIZE
            temperature = TEMPERATURE

        # Utiliser le texte de l'utilisateur directement (comme dans excel_analyst_ui.py)
        instruction = user_text if user_text else "Prends connaissance du fichier joint et attends mes instructions."

        display_text = f"{file_list}\n{instruction}" if file_list else instruction
        llm_text = f"{conversation_contexte} **Instruction de l'utilisateur :**\n{instruction}"

        st.session_state.messages.append({
            "role": "user",
            "display_content": display_text,
            "content": llm_text
        })

        messages_pour_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        with st.chat_message("user"):
            st.markdown(display_text)

        # --- C. APPEL DU CHATBOT AVEC LE BON ENDPOINT ---
        with st.chat_message("assistant"):
            start_time = time.time()

            payload = {
                "messages": messages_pour_api,
                "modele": DEFAULT_LLM,
                "temperature": temperature,
                "context_size": context_size,
                "session_id": st.session_state.session_id,
                "mode": mode,
                "think": st.session_state.think_mode,
                "tables_info": st.session_state.tables_info,
            }

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

            full_response = st.write_stream(lire_flux_api())
            #st.code(repr(full_response[:300]))
            st.caption(f"⏱️ {time.time() - start_time:.2f}s")

            # --- D. TRAITEMENT POST-RÉPONSE (EXCEL) ---
            message_assistant = {
                "role": "assistant",
                "display_content": full_response,
                "content": full_response,
            }

            # Si nous sommes en mode Excel et que des métadonnées SQL sont présentes
            if st.session_state.knowledge_ready:
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
                                key=f"dl_{len(st.session_state.messages)}",
                            )

        st.session_state.messages.append(message_assistant)
