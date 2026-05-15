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
    st.session_state.tables_info = None   # métadonnées retournées par /parse_excel
    st.session_state.knowledge_ready = False
    st.session_state.last_file_id = None

def reset_and_rerun():
    # Libère la session DuckDB côté backend avant de réinitialiser
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
# Parsing de la réponse SQL du LLM
# ---------------------------------------------------------------------------

def extraire_sql_et_metadata(llm_response: str) -> tuple[str | None, dict]:
    """
    Extrait le bloc SQL et les commentaires CHART_* depuis la réponse du LLM.

    Format attendu :
        ```sql
        -- CHART_TYPE: bar
        -- CHART_X: region
        -- CHART_Y: total
        -- CHART_TITLE: Ventes par région
        SELECT region, SUM(montant) as total FROM tableau_1 GROUP BY region
        ```
    """
    sql_match = re.search(r"```sql\n(.*?)\n```", llm_response, re.DOTALL)
    if not sql_match:
        return None, {}

    bloc = sql_match.group(1).strip()

    # Extraction des métadonnées graphe depuis les commentaires
    chart_meta = {}
    for key in ["CHART_TYPE", "CHART_X", "CHART_Y", "CHART_TITLE", "CHART_COLOR"]:
        m = re.search(rf"--\s*{key}:\s*(.+)", bloc)
        if m:
            chart_meta[key] = m.group(1).strip()

    # Nettoyage : on retire les lignes de commentaires pour garder le SQL pur
    lignes_sql = [l for l in bloc.splitlines() if not l.strip().startswith("--")]
    sql_pur = "\n".join(lignes_sql).strip()

    return sql_pur, chart_meta


def construire_graphe(df: pd.DataFrame, meta: dict) -> go.Figure | None:
    """
    Construit un graphe Plotly à partir du DataFrame résultat et des métadonnées.
    """
    chart_type = meta.get("CHART_TYPE", "bar").lower()
    x = meta.get("CHART_X")
    y = meta.get("CHART_Y")
    title = meta.get("CHART_TITLE", "")
    color = meta.get("CHART_COLOR")

    # Vérification que les colonnes existent
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
            return px.bar(**kwargs)  # fallback
    except Exception as e:
        st.warning(f"⚠️ Impossible de construire le graphe automatiquement : {e}")
        return None


def executer_sql_backend(sql: str) -> pd.DataFrame | None:
    """Envoie le SQL au backend et retourne un DataFrame pandas."""
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

if uploaded_file:
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

        if st.sidebar.button("Charger la page en mémoire", use_container_width=True):
            st.session_state.knowledge_ready = False
            st.session_state.tables_info = None

            with st.spinner("⏳ Chargement et indexation DuckDB..."):
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
                    st.success(f"✅ {len(data['tables'])} table(s) chargée(s) en mémoire.")
                else:
                    st.error(f"❌ {data.get('message', 'Erreur inconnue')}")

        if not st.session_state.knowledge_ready:
            st.info("👆 Cliquez sur **Charger la page en mémoire** pour commencer.")
            st.stop()

    except Exception as e:
        st.sidebar.error(f"❌ Erreur : {e}")
        st.stop()

else:
    st.info("📌 Veuillez charger un fichier Excel dans la barre latérale pour commencer.")
    st.stop()


# ---------------------------------------------------------------------------
# Aperçu des tables chargées
# ---------------------------------------------------------------------------

if st.session_state.tables_info:
    for i, table in enumerate(st.session_state.tables_info):
        with st.expander(f"Table {i+1} : `{table['name']}` — {table['n_rows']} lignes × {table['n_cols']} colonnes"):
            # Affichage du schéma uniquement (les données sont dans DuckDB)
            schema_df = pd.DataFrame([
                {"Colonne": col, "Type": dtype}
                for col, dtype in table["dtypes"].items()
            ])
            st.dataframe(schema_df, hide_index=True, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Mode de chat
# ---------------------------------------------------------------------------

with st.sidebar:
    mode_chat = st.radio(
        "🧠 Mode d'interaction",
        ["📊 Analyse & Graphiques", "💬 Discussion & Recherche"],
    )
    st.markdown("---")
    st.caption(
        "En mode **Analyse**, le LLM génère du SQL exécuté directement sur vos données.\n\n"
        "En mode **Discussion**, posez des questions en langage naturel."
    )

is_code_mode = "Analyse" in mode_chat

# ---------------------------------------------------------------------------
# Historique du chat
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["display_content"])
        if "plot" in message:
            st.plotly_chart(message["plot"], use_container_width=True)
        if "dataframe" in message:
            st.dataframe(message["dataframe"], use_container_width=True)

# ---------------------------------------------------------------------------
# Zone de saisie
# ---------------------------------------------------------------------------

placeholder = (
    "ex: 'Fais un graphique des ventes par région'"
    if is_code_mode
    else "ex: 'Quelles sont les valeurs les plus élevées ?'"
)
user_prompt = st.chat_input(placeholder)

if not user_prompt:
    st.stop()

# Affichage message utilisateur
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

# ---------------------------------------------------------------------------
# Appel API + traitement réponse
# ---------------------------------------------------------------------------

url_cible = (
    f"{API_URL}/chat_data_analyst"
    if is_code_mode
    else f"{API_URL}/chat_data_analyst"  # même route, le system prompt gère les deux cas
)

payload = {
    "messages": messages_pour_api,
    "modele": DEFAULT_LLM,
    "temperature": selected_temperature,
    "context_size": selected_context_size,
    "session_id": st.session_state.session_id,
}

with st.chat_message("assistant"):
    start_time = time.time()

    def lire_flux_api():
        try:
            with requests.post(url_cible, json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        yield chunk.decode("utf-8")
        except Exception as e:
            yield f"❌ Erreur de connexion : {str(e)}"

    full_response = st.write_stream(lire_flux_api())
    elapsed = time.time() - start_time
    st.caption(f"⏱️ {elapsed:.2f}s")

    # -----------------------------------------------------------------------
    # Traitement de la réponse SQL
    # -----------------------------------------------------------------------

    sql, chart_meta = extraire_sql_et_metadata(full_response)

    if sql:
        with st.spinner("⚙️ Exécution SQL..."):
            df_result = executer_sql_backend(sql)

        if df_result is not None and not df_result.empty:
            message_assistant = {
                "role": "assistant",
                "display_content": full_response,
                "content": full_response,
                "dataframe": df_result,
            }

            # Graphe si métadonnées présentes
            if chart_meta:
                fig = construire_graphe(df_result, chart_meta)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    message_assistant["plot"] = fig

            # Tableau résultat
            st.dataframe(df_result, use_container_width=True)

            # Export CSV
            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Télécharger le résultat (CSV)",
                data=csv,
                file_name="resultat.csv",
                mime="text/csv",
                key=f"dl_{len(st.session_state.messages)}",
            )

            st.session_state.messages.append(message_assistant)

        elif df_result is not None and df_result.empty:
            st.info("ℹ️ La requête n'a retourné aucun résultat.")
            st.session_state.messages.append({
                "role": "assistant",
                "display_content": full_response,
                "content": full_response,
            })

    else:
        # Réponse textuelle pure (question sans SQL, ou mode discussion)
        st.session_state.messages.append({
            "role": "assistant",
            "display_content": full_response,
            "content": full_response,
        })