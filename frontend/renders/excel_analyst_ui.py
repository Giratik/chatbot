# excel_analyst_ui.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
import uuid
import time
import re
import os

# --- CONFIGURATION GLOBALE ---
LOGO_PATH = "ressource/Eau_de_Paris_bleu.svg.png"
API_URL = os.environ.get("API_URL", "http://backend:8000")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "ministral-3:14b")
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 30000))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.4))

def init_session_state():
    """Initialise ou restaure les variables de session pour l'assistant Excel."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tables_info" not in st.session_state:
        st.session_state.tables_info = None
    if "knowledge_ready" not in st.session_state:
        st.session_state.knowledge_ready = False
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None
    if "tables_data" not in st.session_state:
        st.session_state.tables_data = {}
    if "think_mode" not in st.session_state:
        st.session_state.think_mode = False

def reset_and_rerun():
    """Réinitialise la session Excel et recharge la page."""
    if "session_id" in st.session_state:
        try:
            requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=3)
        except Exception:
            pass
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.tables_info = None
    st.session_state.knowledge_ready = False
    st.session_state.last_file_id = None
    st.session_state.tables_data = {}
    st.rerun()

def extraire_sql_et_metadata(llm_response: str) -> tuple[str | None, dict]:
    """
    Extrait le code SQL et les métadonnées de graphique via l'API backend.
    """
    try:
        resp = requests.post(
            f"{API_URL}/extract_sql_metadata",
            json={"llm_response": llm_response},
            timeout=30,
        )
        data = resp.json()
        if data.get("status") == "success":
            return data.get("sql"), data.get("chart_meta", {})
        else:
            st.error(f"❌ Erreur extraction SQL : {data.get('message')}")
            return None, {}
    except Exception as e:
        st.error(f"❌ Erreur connexion backend : {e}")
        return None, {}

def construire_graphe(df: pd.DataFrame, meta: dict) -> go.Figure | None:
    """
    Construit un graphique via l'API backend et retourne une figure Plotly.
    """
    try:
        resp = requests.post(
            f"{API_URL}/build_chart",
            json={
                "data": df.to_dict(orient='records'),
                "chart_meta": meta
            },
            timeout=30,
        )
        data = resp.json()
        if data.get("status") == "success":
            chart_spec = data.get("chart_spec", {})

            # Reconstruire la figure Plotly à partir de la spécification
            chart_type = chart_spec.get("type", "bar")
            x = chart_spec.get("layout", {}).get("xaxis", {}).get("title")
            y = chart_spec.get("layout", {}).get("yaxis", {}).get("title")
            title = chart_spec.get("layout", {}).get("title")
            color = chart_spec.get("color")

            # Convertir les données JSON en DataFrame
            df_data = pd.DataFrame(chart_spec.get("data", []))

            try:
                kwargs = dict(data_frame=df_data, x=x, y=y, title=title)
                if color and color in df_data.columns:
                    kwargs["color"] = color

                if chart_type == "bar":
                    return px.bar(**kwargs)
                elif chart_type == "line":
                    return px.line(**kwargs)
                elif chart_type == "pie":
                    return px.pie(df_data, names=x, values=y, title=title)
                elif chart_type == "scatter":
                    return px.scatter(**kwargs)
                else:
                    return px.bar(**kwargs)
            except Exception as e:
                st.warning(f"⚠️ Graphe impossible à construire : {e}")
                return None
        else:
            st.error(f"❌ Erreur construction graphe : {data.get('message')}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur connexion backend : {e}")
        return None

def executer_sql_backend(sql: str) -> pd.DataFrame | None:
    """
    Exécute une requête SQL via le backend et retourne les résultats.
    """
    try:
        resp = requests.post(
            f"{API_URL}/execute_sql_excel",
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

def render_excel_analyst(title="📊 Assistant Data & Graphiques"):
    """
    Fonction principale pour générer l'interface d'analyse Excel.
    Peut être appelée depuis n'importe quelle page Streamlit.
    """
    init_session_state()

    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH)

    # --- SIDEBAR ---
    with st.sidebar:
        st.session_state.think_mode = st.toggle(
            "Mode raisonnement",
            value=st.session_state.think_mode,
            help="Active le mode 'raisonnement' des modèles pour une réflexion approfondie"
        )
        st.divider()

        if st.button("Nouvelle session", use_container_width=True):
            reset_and_rerun()

    st.title(title)

    # --- UPLOAD & CHARGEMENT ---
    uploaded_file = st.sidebar.file_uploader("Chargez un fichier Excel", type=["xlsx"])

    if not uploaded_file:
        st.info("📌 Veuillez charger un fichier Excel dans la barre latérale pour commencer.")
        return

    try:
        file_id = uploaded_file.name + str(uploaded_file.size)

        if file_id != st.session_state.get("last_file_id"):
            st.session_state.messages = []
            st.session_state.knowledge_ready = False
            st.session_state.tables_info = None
            st.session_state.last_file_id = file_id
            st.sidebar.info("📂 Nouveau fichier détecté.")

        xls = pd.ExcelFile(uploaded_file)
        onglet_choisi = st.sidebar.selectbox("📂 Choisissez l'onglet :", xls.sheet_names)

        if st.sidebar.button("Charger en mémoire", use_container_width=True):
            st.session_state.knowledge_ready = False
            st.session_state.tables_info = None
            st.session_state.tables_data = {}

            with st.spinner("⏳ Chargement..."):
                uploaded_file.seek(0)
                resp = requests.post(
                    f"{API_URL}/parse_excel",
                    files={"file": (uploaded_file.name, uploaded_file.getbuffer())},
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

                    # Lecture des données pour affichage
                    uploaded_file.seek(0)
                    xls_data = pd.read_excel(uploaded_file, sheet_name=onglet_choisi, header=None)

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

                    st.success(f"✅ {len(data['tables'])} table(s) chargée(s).")
                else:
                    st.error(f"❌ {data.get('message', 'Erreur inconnue')}")

        if not st.session_state.knowledge_ready:
            st.info("👆 Cliquez sur **Charger en mémoire** pour commencer.")
            return

    except Exception as e:
        st.sidebar.error(f"❌ Erreur : {e}")
        return

    # --- AFFICHAGE DES TABLES DANS SIDEBAR ---
    with st.sidebar:
        if st.session_state.get("tables_data"):
            for name, df in st.session_state.tables_data.items():
                st.subheader(f"`{name}`")
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Chargez un fichier pour visualiser les données.")

    # --- HISTORIQUE DES MESSAGES ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # On masque le bloc sql de l'affichage — interne au backend
            display = re.sub(r"```sql\n.*?\n```\n?", "", message["display_content"], flags=re.DOTALL).strip()
            st.markdown(display)
            if "plot" in message:
                st.plotly_chart(message["plot"], use_container_width=True)
            if "dataframe" in message:
                st.dataframe(message["dataframe"], use_container_width=True)

    # --- SAISIE & APPEL API ---
    user_prompt = st.chat_input("Posez une question sur vos données...")

    if not user_prompt:
        return

    with st.chat_message("user"):
        st.markdown(user_prompt)

    st.session_state.messages.append({
        "role": "user",
        "display_content": user_prompt,
        "content": user_prompt,
    })

    messages_pour_api = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    payload = {
        "messages": messages_pour_api,
        "modele": DEFAULT_LLM,
        "temperature": TEMPERATURE,
        "context_size": CONTEXT_SIZE,
        "session_id": st.session_state.session_id,
        "mode": "graphique",
        "think": st.session_state.think_mode,
    }

    with st.chat_message("assistant"):
        start_time = time.time()

        def lire_flux_api():
            try:
                with requests.post(
                    f"{API_URL}/chat_data_analyst",
                    json=payload, stream=True, timeout=120
                ) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            yield chunk.decode("utf-8")
            except Exception as e:
                yield f"❌ Erreur de connexion : {str(e)}"

        full_response = st.write_stream(lire_flux_api())
        st.caption(f"⏱️ {time.time() - start_time:.2f}s")

        # --- GRAPHE SI MÉTADONNÉES CHART_* PRÉSENTES ---
        sql, chart_meta = extraire_sql_et_metadata(full_response)

        message_assistant = {
            "role": "assistant",
            "display_content": full_response,
            "content": full_response,
        }

        if sql and chart_meta:
            with st.spinner("📊 Construction du graphe..."):
                df_result = executer_sql_backend(sql)
                if df_result is not None and not df_result.empty:
                    fig = construire_graphe(df_result, chart_meta)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        message_assistant["plot"] = fig

                    st.dataframe(df_result, use_container_width=True)
                    message_assistant["dataframe"] = df_result

                    csv = df_result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Télécharger (CSV)",
                        data=csv,
                        file_name="resultat.csv",
                        mime="text/csv",
                        key=f"dl_{len(st.session_state.messages)}",
                    )

        st.session_state.messages.append(message_assistant)
