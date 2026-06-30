# chat_ui_v2.py - Interface de Chat Généraliste avec Support Multi-Fichiers
"""
Version avancée du chat qui intègre:
- Conversation standard pour demandes générales
- Traitement des fichiers Excel, Word, PDF, etc.
- Exécution SQL et génération de graphiques
- Gestion unifiée de session

Rôle dans l'architecture :
- Composant principal utilisé par Main.py pour le chatbot généraliste
- Alternative à Chat.py qui est spécialisé pour les questions RH
- Fournit une interface unifiée pour l'analyse de données et le chat général

Différence avec Chat.py (RAG) :
- Ce fichier gère les demandes générales et l'analyse de fichiers
- Chat.py utilise le pipeline RAG pour les questions spécifiques RH
- Ce composant n'a pas besoin d'accès à une base de connaissances spécifique
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

from mots_cle import (
    COMPANY,
    ACRONYME
)

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

def set_state(i):
    st.session_state.stage = i

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
    import io
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


# --- FONCTION PRINCIPALE DE CHAT HYBRIDE ---
def render_general_purpose_chat(title=f"Chatbot {ACRONYME} Hybride"):
    """
    Interface de chat avancée avec support Excel et SQL intégré.
    Fonction principale utilisée par Main.py pour le chatbot généraliste.

    Args:
        title: Titre à afficher pour l'interface de chat
    """
    init_session_state()

    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH)

    # --- SIDEBAR UNIFIÉ ---
    with st.sidebar:

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
            st.info("Si vous uploadez un excel, son contenu s'affichera ici.")

        st.divider()
        st.caption(f"© {COMPANY} - Chatbot EDP-IA")

    st.title(title)

    # 1. AFFICHAGE DE L'HISTORIQUE AVEC SUPPORT EXCEL
    # Dans la boucle d'affichage de l'historique
    for message in st.session_state.messages:
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

    # --- SÉLECTION D'ONGLET EXCEL ---
    if st.session_state.stage == 1 and st.session_state.pending_sheet_names:
        with st.chat_message("assistant"):
            st.markdown("Quel onglet voulez-vous analyser ?")
            onglet_choisi = st.radio(
                "Sélectionnez un onglet :",
                st.session_state.pending_sheet_names,
                key="excel_sheet_choice",
                label_visibility="collapsed"
            )
            if st.button("Confirmer", key="confirm_sheet_choice"):
                st.session_state.selected_sheet = onglet_choisi
                st.session_state.stage = 2
                st.rerun()

    # --- PARSING EXCEL ---
    if st.session_state.stage == 2 and st.session_state.pending_excel_file:
        st.sidebar.write("🔄 Parsing en cours...")  # à retirer après debug
        with st.spinner("⏳ Chargement du fichier Excel..."):
            parse_and_load_excel()
        st.rerun()

    # 2. SAISIE UTILISATEUR AVEC SUPPORT FICHIERS ÉTENDU
    # Si une query a été mise en attente pendant le chargement Excel multi-onglets,
    # on la récupère et on la traite comme si l'utilisateur venait de la saisir.
    _deferred_query = st.session_state.get("query_to_execute")
    if _deferred_query:
        st.session_state.query_to_execute = None
        user_input = _deferred_query  # str — sera géré par la branche `isinstance(user_input, str)`
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
                st.session_state.processed_files.append(fichier_joint.name)
                file_list += f"📎 **Fichier joint :** {fichier_joint.name}\n"

                # DÉTECTION ET TRAITEMENT SPÉCIFIQUE EXCEL
                if fichier_joint.name.lower().endswith('.xlsx'):
                    st.session_state.excel_mode = True
                    st.session_state.current_excel_file = fichier_joint.name

                    if file_id != st.session_state.get("last_file_id"):
                        st.session_state.messages = []
                        st.session_state.knowledge_ready = False
                        st.session_state.tables_info = None
                        st.session_state.last_file_id = file_id
                        st.session_state.tables_data = {}
                        st.session_state.selected_sheet = None

                        xls = pd.ExcelFile(fichier_joint)

                        if len(xls.sheet_names) == 1:
                            # Une seule feuille : on peut continuer directement
                            st.session_state.selected_sheet = xls.sheet_names[0]
                            st.session_state.pending_excel_file = fichier_joint.getbuffer().tobytes()
                            st.session_state.pending_excel_name = fichier_joint.name
                            st.session_state.pending_user_query = user_input.text or None
                            st.session_state.stage = 2  # Prêt pour le parsing
                        else:
                            # Plusieurs feuilles : on stocke et on attend le choix
                            st.session_state.pending_excel_file = fichier_joint.getbuffer().tobytes()
                            st.session_state.pending_excel_name = fichier_joint.name
                            st.session_state.pending_sheet_names = xls.sheet_names
                            st.session_state.pending_user_query = user_input.text or None
                            st.session_state.stage = 1  # En attente du choix d'onglet
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

        messages_pour_api = [
            {"role": m["role"], "content": m.get("content") or m.get("display_content", "")}
            for m in st.session_state.messages
        ]

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
        st.rerun()