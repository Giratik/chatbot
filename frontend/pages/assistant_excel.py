#frontend/pages/new_excel_assist.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
import uuid
import time
import re
import os

# --- CONFIGURATION ---
API_URL = os.environ.get("API_URL", "http://backend:8000")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "ministral-3:14b")

selected_context_size = 12288
selected_temperature = 0.4


# ---------------------------------------------------------------------------
# Gestion de session
# ---------------------------------------------------------------------------

def new_session():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.tables_info = None
    st.session_state.knowledge_ready = False
    st.session_state.last_file_id = None
    st.session_state.tables_data = {}

def reset_and_rerun():
    if "session_id" in st.session_state:
        try:
            requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=3)
        except Exception:
            pass
    new_session()
    st.rerun()

if "initialized" not in st.session_state:
    new_session()
    st.session_state.initialized = True


# ---------------------------------------------------------------------------
# Parsing graphe
# ---------------------------------------------------------------------------

def extraire_sql_et_metadata(llm_response: str) -> tuple[str | None, dict]:
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


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Assistant Data & Graphiques", page_icon="📊", layout="wide")
st.title("📊 Assistant Data & Graphiques")

if st.sidebar.button("Nouvelle session", use_container_width=True):
    reset_and_rerun()

# ---------------------------------------------------------------------------
# Upload & chargement
# ---------------------------------------------------------------------------

uploaded_file = st.sidebar.file_uploader("Chargez un fichier Excel", type=["xlsx"])

if not uploaded_file:
    st.info("📌 Veuillez charger un fichier Excel dans la barre latérale pour commencer.")
    st.stop()

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

                # 2. Lecture pandas locale pour l'affichage — zéro aller-retour backend
                uploaded_file.seek(0)
                xls_data = pd.read_excel(uploaded_file, sheet_name=onglet_choisi, header=None)
                
                # Reproduit la même détection d'îlots que le backend : on stocke
                # un DataFrame par table détectée, aligné sur tables_info
                for table in data["tables"]:
                    # Le backend retourne n_rows/n_cols — on extrait la portion correspondante
                    # plus simplement : on relit chaque table via execute_sql une seule fois
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
        st.stop()

except Exception as e:
    st.sidebar.error(f"❌ Erreur : {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Aperçu des tables
# ---------------------------------------------------------------------------

#if st.session_state.tables_info:
#    for i, table in enumerate(st.session_state.tables_info):
#        with st.expander(
#            f"Table {i+1} : `{table['name']}` — {table['n_rows']} lignes × {table['n_cols']} colonnes"
#        ):
#            schema_df = pd.DataFrame([
#                {"Colonne": col, "Type": dtype}
#                for col, dtype in table["dtypes"].items()
#            ])
#            st.dataframe(schema_df, hide_index=True, use_container_width=True)
#
#st.markdown("---")

#tab_chat, tab_data  = st.tabs(["💬 Chat", "📋 Données"])

with st.sidebar:
    if st.session_state.get("tables_data"):
        for name, df in st.session_state.tables_data.items():
            st.subheader(f"`{name}`")
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Chargez un fichier pour visualiser les données.")

# ---------------------------------------------------------------------------
# Historique
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # On masque le bloc ```sql``` de l'affichage — interne au backend
        display = re.sub(r"```sql\n.*?\n```\n?", "", message["display_content"], flags=re.DOTALL).strip()
        st.markdown(display)
        if "plot" in message:
            st.plotly_chart(message["plot"], use_container_width=True)
        if "dataframe" in message:
            st.dataframe(message["dataframe"], use_container_width=True)

# ---------------------------------------------------------------------------
# Saisie & appel API
# ---------------------------------------------------------------------------

user_prompt = st.chat_input("Posez une question sur vos données...")

if not user_prompt:
    st.stop()

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
    "temperature": selected_temperature,
    "context_size": selected_context_size,
    "session_id": st.session_state.session_id,
    "mode": "graphique",  # toujours actif : le backend ajoute CHART_* seulement si pertinent
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

    # -----------------------------------------------------------------------
    # Graphe si métadonnées CHART_* présentes
    # -----------------------------------------------------------------------

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