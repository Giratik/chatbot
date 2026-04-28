import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import uuid
import time
import re
import os
import unicodedata


# --- CONFIGURATION ---
LOGO_PATH = "ressource/Eau_de_Paris_bleu.svg.png"
API_URL = os.environ.get("API_URL", "http://backend:8000")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "ministral-3:14b")

selected_context_size = 16384
selected_temperature = 0.4

# --- FONCTIONS DE SESSION ---
def new_session():
    """Création d'une session pour l'utilisateur"""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.dataframe = None
    st.session_state.csv_data = None
    st.session_state.knowledge_ready = False # Pour savoir si le backend a bien le CSV

def reset_and_rerun():
    """Réinitialise le chat"""
    new_session()
    st.rerun()


# 1. Fonction pour retirer les accents d'une chaîne
def enlever_accents(texte):
    if pd.isna(texte):
        return texte
    return unicodedata.normalize('NFKD', str(texte)).encode('ASCII', 'ignore').decode('utf-8')





# --- INTERFACE ---
st.set_page_config(page_title="Assistant Data & Graphiques", page_icon="📊", layout="wide")
st.title("📊 Assistant Data & Graphiques")

if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH) 

# Initialisation de la session
if "session_id" not in st.session_state:
    new_session()

# Bouton pour réinitialiser
if st.sidebar.button("Nouvelle session", use_container_width=True):
    reset_and_rerun()

# === SECTION 1 : UPLOAD DU FICHIER ===
uploaded_file = st.sidebar.file_uploader("Chargez un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # --- Détection du changement de fichier ou d'onglet ---
        file_id = uploaded_file.name + str(uploaded_file.size)
        
        if file_id != st.session_state.get("last_file_id"):
            st.session_state.messages = []
            st.session_state.knowledge_ready = False
            st.session_state.last_file_id = file_id
            st.sidebar.info("📂 Nouveau fichier chargé — conversation réinitialisée.")
        
        # --- Lecture du fichier ---
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        
        else:  # Excel
            xls = pd.ExcelFile(uploaded_file)
            onglet_choisi = st.sidebar.selectbox("📂 Choisissez l'onglet :", xls.sheet_names)
            
            # Envoi au backend
            with st.spinner("⏳ Parsing Excel..."):
                files = {"file": (uploaded_file.name, uploaded_file.getbuffer())}
                params = {"sheet_name": onglet_choisi}
                response = requests.post(f"{API_URL}/parse_excel", files=files, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        # Convertir JSON → DataFrame
                        df = pd.DataFrame(data["tableau"])
                        st.success(f"✅ {len(df)} lignes × {len(df.columns)} colonnes")
                    else:
                        st.error(f"❌ {data.get('message')}")
                        st.stop()
                else:
                    st.error("❌ Erreur backend")
                    st.stop()
            
            if df is None:
                st.stop()
        
        # --- PRÉ-TRAITEMENT ---
        # Nettoyage des NOMS de colonnes
        df.columns = [enlever_accents(col) for col in df.columns]
        
        # Nettoyage du CONTENU des colonnes textuelles
        colonnes_textes = df.select_dtypes(include=['object', 'string']).columns
        for col in colonnes_textes:
            df[col] = df[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        
        # Stockage du DataFrame dans la session
        st.session_state.dataframe = df
        st.sidebar.success(f"✅ Fichier chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        # --- ENVOI AU BACKEND POUR RAG ---
        if not st.session_state.knowledge_ready:
            with st.sidebar.status("🔄 Indexation du fichier ...", expanded=True):
                csv_propre = df.to_csv(index=False).encode('utf-8')
                files = {"file": (uploaded_file.name, csv_propre, "text/csv")}
                data = {"session_id": st.session_state.session_id}
                response = requests.post(f"{API_URL}/ajouter_au_savoir_csv", files=files, data=data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    if response_data.get("statut") == "succès":
                        st.session_state.knowledge_ready = True
                        st.write("✅ Indexation réussie !")
                    else:
                        st.error(f"❌ Erreur du backend : {response_data.get('erreur')}")
                else:
                    st.error("❌ Erreur de connexion au serveur Backend.")
        
        # --- APERÇU DES DONNÉES ---
        with st.expander("📋 Aperçu des données", expanded=True):
            st.write(df.head(10))
            st.caption(f"📊 Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
            
            with st.expander("📝 Colonnes disponibles", expanded=False):
                colonnes_info = "\n".join([f"• **{col}** ({dtype})" for col, dtype in df.dtypes.items()])
                st.markdown(colonnes_info)
            
    except Exception as e:
        st.sidebar.error(f"❌ Erreur : {e}")
        st.stop()

else:
    st.info("📌 Veuillez charger un fichier CSV ou Excel dans la barre latérale pour commencer.")
    st.stop()


# === SECTION 1.5 : CHOIX DU MODE (LES DEUX CERVEAUX) ===
st.markdown("---")
with st.sidebar:
    mode_chat = st.radio(
        "🧠 Comment voulez-vous interagir avec ce fichier ?",
        ["📊 Analyse & Graphiques (Génération de code)", "💬 Discussion & Recherche textuelle (RAG)"],
        horizontal=True
    )
    st.markdown("---")
    st.info(f"""Pour la génération de code, si le bot n'entre pas correctement le nom d'une variable, vous pouvez lui donner le nom exacte entre guillemets.""")

# === SECTION 2 : HISTORIQUE DU CHAT ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["display_content"])
        if "plot" in message:
            st.plotly_chart(message["plot"], use_container_width=True)
        if "dataframe" in message:
            st.dataframe(message["dataframe"])

# === SECTION 3 : ZONE DE SAISIE ===
# On adapte le placeholder selon le mode
placeholder_text = "ex: 'Fais un graphique en barres'" if "Analyse" in mode_chat else "ex: 'Que disent les clients sur les retards ?'"
user_prompt = st.chat_input(placeholder_text)

if user_prompt and st.session_state.dataframe is not None:
    df = st.session_state.dataframe
    
    # Affichage du message utilisateur
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    st.session_state.messages.append({
        "role": "user",
        "display_content": user_prompt,
        "content": user_prompt
    })
    
    # === SECTION 4 : APPEL À L'API ET ROUTAGE ===
    messages_pour_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    colonnes_info = "\n".join([f"- {col}: {dtype}" for col, dtype in df.dtypes.items()])
    
    payload = {
        "messages": messages_pour_api,
        "modele": DEFAULT_LLM,
        "temperature": selected_temperature,
        "context_size": selected_context_size,
        "colonnes_info": colonnes_info,
        "session_id": st.session_state.session_id
    }
    
    # Détermination de la route API selon le mode sélectionné
    is_code_mode = "Analyse" in mode_chat
    url_cible = f"{API_URL}/chat_data_analyst" if is_code_mode else f"{API_URL}/chat_csv_rag"
    
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
        
        # Affichage du flux
        full_response = st.write_stream(lire_flux_api())
        elapsed_time = time.time() - start_time
        st.caption(f"⏱️ **Temps** : {elapsed_time:.2f}s")
        
        # === SECTION 5 : TRAITEMENT DE LA RÉPONSE ===
        
        if is_code_mode:
            # ---------------------------------------------------------
            # MODE 1 : ANALYSE (Extraction et exécution du code Python)
            # ---------------------------------------------------------
            with st.spinner("⚙️ Exécution du code..."):
                try:
                    code_match = re.search(r'```python\n(.*?)\n```', full_response, re.DOTALL)
                    
                    if code_match:
                        code_extrait = code_match.group(1)
                        
                        #with st.expander("👀 Voir le code généré", expanded=False):
                        #    st.code(code_extrait, language="python")
                        
                        execution_env = {"df": df, "px": px, "pd": pd, "__builtins__": __builtins__}
                        exec(code_extrait, execution_env)
                        
                        message_assistant = {
                            "role": "assistant",
                            "display_content": "Résultat généré avec succès ✅",
                            "content": full_response # On garde le code dans le 'content' secret pour l'historique API
                        }
                        
                        if "fig" in execution_env:
                            fig = execution_env["fig"]
                            st.plotly_chart(fig, use_container_width=True)
                            message_assistant["plot"] = fig
                        
                        if "df_resultat" in execution_env:
                            df_resultat = execution_env["df_resultat"]
                            st.dataframe(df_resultat, use_container_width=True)
                            message_assistant["dataframe"] = df_resultat
                            
                            csv = df_resultat.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Télécharger le résultat (CSV)",
                                data=csv,
                                file_name="donnees_traitees.csv",
                                mime="text/csv",
                                key=f"dl_{len(st.session_state.messages)}"
                            )
                            
                            st.session_state.dataframe = df_resultat
                        
                        st.session_state.messages.append(message_assistant)
                        
                    else:
                        st.warning("⚠️ Le code généré n'a pas pu être extrait.")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "display_content": full_response,
                            "content": full_response
                        })
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'exécution : {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "display_content": f"❌ Erreur d'exécution : {str(e)}",
                        "content": full_response
                    })

        else:
            # ---------------------------------------------------------
            # MODE 2 : DISCUSSION RAG (Juste du texte, pas d'exécution)
            # ---------------------------------------------------------
            st.session_state.messages.append({
                "role": "assistant",
                "display_content": full_response,
                "content": full_response
            })