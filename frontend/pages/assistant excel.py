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
API_URL = os.environ.get("API_URL", "http://backend:8000")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "ministral-3:14b")

selected_context_size = 12288
selected_temperature = 0.4

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- FONCTIONS DE SESSION ---
def new_session():
    """Création d'une session pour l'utilisateur"""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.dataframe = None
    st.session_state.csv_data = None
    st.session_state.tableaux_extraits = None
    st.session_state.knowledge_ready = False 

def reset_and_rerun():
    """Réinitialise le chat"""
    new_session()
    st.rerun()


# 1. Fonction pour retirer les accents d'une chaîne
def enlever_accents(texte):
    if pd.isna(texte):
        return texte
    return unicodedata.normalize('NFKD', str(texte)).encode('ASCII', 'ignore').decode('utf-8')

if "initialized" not in st.session_state:
    new_session()
    # On marque la session comme initialisée pour ne pas écraser les données 
    # lors des prochains clics sur des boutons.
    st.session_state.initialized = True



# --- INTERFACE ---
st.set_page_config(page_title="Assistant Data & Graphiques", page_icon="📊", layout="wide")
st.title("📊 Assistant Data & Graphiques")


# Initialisation de la session
if "session_id" not in st.session_state:
    new_session()

# Bouton pour réinitialiser
if st.sidebar.button("Nouvelle session", width='stretch'):
    reset_and_rerun()

# === SECTION 1 : UPLOAD DU FICHIER ===
uploaded_file = st.sidebar.file_uploader("Chargez un fichier Excel", type=["xlsx"])

if uploaded_file:
    
    try:
        # --- Détection du changement de fichier ou d'onglet ---
        file_id = uploaded_file.name + str(uploaded_file.size)
        
        if file_id != st.session_state.get("last_file_id"):
            st.session_state.messages = []
            st.session_state.knowledge_ready = False
            st.session_state.last_file_id = file_id
            st.sidebar.info("📂 Nouveau fichier chargé.")
        


        # --- Lecture du fichier ---
        if uploaded_file.name.endswith('.csv'):
            st.session_state.dataframe = pd.read_csv(uploaded_file)
        
        else:  # Excel
            xls = pd.ExcelFile(uploaded_file)
            onglet_choisi = st.sidebar.selectbox("📂 Choisissez l'onglet :", xls.sheet_names)
            
            # Envoi au backend

            files = {"file": (uploaded_file.name, uploaded_file.getbuffer())}
            params = {"sheet_name": onglet_choisi}
            if st.sidebar.button ("Charger la page en mémoire"):
                st.session_state.knowledge_ready = False
                with st.spinner("⏳ Parsing Excel..."):
                    response = requests.post(f"{API_URL}/parse_every_tab_excel", files=files, params=params)
                    data=response.json()
                    #with st.sidebar:
                    #    data=response.json()
                    #    st.json(data)
#                    
                    if response.status_code == 200:
                        # On suppose que 'data' est = response.json()
                        # Utilisation de simples quotes pour éviter l'erreur de syntaxe dans le f-string
                        liste_tableaux = data.get("tableau", [])

                        st.success(f"{len(liste_tableaux)} tableau(x) détecté(s).")

                        # On stocke uniquement la liste des tableaux dans la session
                        st.session_state.tableaux_extraits = liste_tableaux

                        # On boucle sur la liste de dictionnaires
                        for i, table_dict in enumerate(st.session_state.tableaux_extraits):

                            # Extraction depuis les clés du dictionnaire JSON
                            title = table_dict.get("titre")
                            table_data = table_dict.get("donnees")

                            #with st.expander(f"Tableau n°{i+1} : {title if title else 'Sans titre'}", expanded=True):
                                # Création du DataFrame (Ligne 0 = Headers)
                            st.session_state.dataframe = pd.DataFrame(table_data[1:], columns=table_data[0])
                            #st.dataframe(st.session_state.dataframe, width='stretch')

                                # Option export CSV par tableau
                                #csv = st.session_state.dataframe.to_csv(index=False).encode('utf-8')
                                #st.download_button(
                                #    label="Télécharger en CSV",
                                #    data=csv,
                                #    file_name=f"{onglet_choisi}_table_{i+1}.csv",
                                #    mime="text/csv",
                                #    key=f"btn_{onglet_choisi}_{i}"
                                #)
                    else:
                        st.error("❌ Erreur backend")
                        st.stop()
#            
            if st.session_state.dataframe is None:
                st.stop()
#        
#        ## --- PRÉ-TRAITEMENT ---
#        ## Nettoyage des NOMS de colonnes
#        #st.session_state.dataframe.columns = [enlever_accents(col) for col in st.session_state.dataframe.columns]
#        #
#        ## Nettoyage du CONTENU des colonnes textuelles
#        #colonnes_textes = st.session_state.dataframe.select_dtypes(include=['object', 'string']).columns
#        #for col in colonnes_textes:
#        #    st.session_state.dataframe[col] = st.session_state.dataframe[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
#        
#        # Stockage du DataFrame dans la session
#        
#        #st.sidebar.success(f"✅ Fichier chargé : {st.session_state.dataframe.shape[0]} lignes × {st.session_state.dataframe.shape[1]} colonnes")
#        
#        # --- ENVOI AU BACKEND POUR RAG ---
            if not st.session_state.knowledge_ready:
                with st.sidebar.status("🔄 Indexation des données ...", expanded=True):
                    
                    # 1. On prépare le payload au format JSON
                    payload = {
                        "session_id": st.session_state.session_id,
                        "tableaux": st.session_state.tableaux_extraits
                    }
                    
                    # 2. On utilise 'json=payload' au lieu de 'files=' et 'data='
                    response = requests.post(f"{API_URL}/knowledge_graphe", json=payload)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        if response_data.get("statut") == "succès":
                            st.session_state.knowledge_ready = True
                            st.write("✅ Indexation réussie !")
                        else:
                            st.error(f"❌ Erreur du backend : {response_data.get('erreur')}")
                    else:
                        st.error(f"❌ Erreur de connexion au serveur Backend (Code {response.status_code}).")
#        
#        # --- APERÇU DES DONNÉES ---
#        with st.expander("📋 Aperçu des données", expanded=True):
#            st.write(st.session_state.dataframe.head(10))
#            st.caption(f"📊 Dimensions : {st.session_state.dataframe.shape[0]} lignes × {st.session_state.dataframe.shape[1]} colonnes")
#            
#            with st.expander("📝 Colonnes disponibles", expanded=False):
#                colonnes_info = "\n".join([f"• **{col}** ({dtype})" for col, dtype in st.session_state.dataframe.dtypes.items()])
#                st.markdown(colonnes_info)
#            
    except Exception as e:
        st.sidebar.error(f"❌ Erreur : {e}")
        st.stop()
#
else:
    st.info("📌 Veuillez charger un fichier Excel dans la barre latérale pour commencer.")
    st.stop()
#
#
# === SECTION 1.5 : CHOIX DU MODE (LES DEUX CERVEAUX) ===
for i, table_dict in enumerate(st.session_state.tableaux_extraits):

    # Extraction depuis les clés du dictionnaire JSON
    title = table_dict.get("titre")
    table_data = table_dict.get("donnees")

    with st.expander(f"Tableau n°{i+1} : {title if title else 'Sans titre'}", expanded=True):
        # Création du DataFrame (Ligne 0 = Headers)
        st.session_state.dataframe = pd.DataFrame(table_data[1:], columns=table_data[0])
        st.dataframe(st.session_state.dataframe, width='stretch')

st.markdown("---")
with st.sidebar:
    mode_chat = st.radio(
        "🧠 Comment voulez-vous interagir avec ce fichier ?",
        ["💬 Discussion & Recherche textuelle (RAG)", "📊 Analyse & Graphiques (Génération de code)"],
        horizontal=True
    )
    st.markdown("---")
    st.info(f"""Pour la génération de code, si le bot n'entre pas correctement le nom d'une variable, vous pouvez lui donner #le nom exacte entre guillemets.""")
#mode_chat = "Discussion & Recherche textuelle"
# === SECTION 2 : HISTORIQUE DU CHAT ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["display_content"])
        if "plot" in message:
            st.plotly_chart(message["plot"], width='stretch')
        if "dataframe" in message:
            st.dataframe(message["dataframe"])

# === SECTION 3 : ZONE DE SAISIE ===
placeholder_text = "ex: 'Fais un graphique des ventes'" if "Analyse" in mode_chat else "ex: 'Que disent les clients ?'"
user_prompt = st.chat_input(placeholder_text)

if user_prompt:
    is_code_mode = "Analyse" in mode_chat
    
    # -------------------------------------------------------------
    # NOUVEAUTÉ : Création du dictionnaire de tous les DataFrames
    # -------------------------------------------------------------
    dfs_dict = {}
    colonnes_info_liste = []
    
    if "tableaux_extraits" in st.session_state and st.session_state.tableaux_extraits:
        for i, tab in enumerate(st.session_state.tableaux_extraits):
            # On s'assure d'avoir un titre valide pour la clé du dictionnaire
            titre = tab.get("titre") or f"Tableau_sans_titre_{i+1}"
            data = tab.get("donnees", [])
            
            if len(data) > 1:
                # Création du DataFrame pour ce tableau précis
                df_temp = pd.DataFrame(data[1:], columns=data[0])
                dfs_dict[titre] = df_temp
                
                # Formatage des infos pour le System Prompt du backend
                cols_str = ", ".join([f"{col} ({dtype})" for col, dtype in df_temp.dtypes.items()])
                colonnes_info_liste.append(f'- dfs["{titre}"] -> Colonnes : {cols_str}')
                
    colonnes_info = "\n".join(colonnes_info_liste)

    # Sécurité pour le mode Analyse
    if is_code_mode and not dfs_dict:
        st.error("⚠️ Aucun tableau n'est disponible pour l'analyse en Python.")
        st.stop()
        
    # Affichage du message utilisateur
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    st.session_state.messages.append({"role": "user", "display_content": user_prompt, "content": user_prompt})

    
# === SECTION 4 : APPEL À L'API ET ROUTAGE ===
    messages_pour_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    
    payload = {
        "messages": messages_pour_api,
        "modele": DEFAULT_LLM,
        "temperature": selected_temperature,
        "context_size": selected_context_size,
        "colonnes_info": colonnes_info, # Envoie la liste détaillée de TOUS les tableaux
        "session_id": st.session_state.session_id
    }
    
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
            with st.spinner("⚙️ Exécution du code..."):
                try:
                    code_match = re.search(r'```python\n(.*?)\n```', full_response, re.DOTALL)
                    if code_match:
                        code_extrait = code_match.group(1)
                        
                        # 🚨 LA MODIFICATION CRUCIALE POUR exec() : On passe 'dfs_dict' au lieu de 'st.session_state.dataframe' !
                        execution_env = {"dfs": dfs_dict, "px": px, "pd": pd, "__builtins__": __builtins__}
                        exec(code_extrait, execution_env)
                        
                        message_assistant = {
                            "role": "assistant",
                            "display_content": "Résultat généré avec succès ✅",
                            "content": full_response # On garde le code dans le 'content' secret pour l'historique API
                        }
                        
                        if "fig" in execution_env:
                            fig = execution_env["fig"]
                            st.plotly_chart(fig, width='stretch')
                            message_assistant["plot"] = fig
                        
                        if "df_resultat" in execution_env:
                            df_resultat = execution_env["df_resultat"]
                            st.dataframe(df_resultat, width='stretch')
                            message_assistant["dataframe"] = df_resultat
                            
                            csv = df_resultat.to_csv(index=False).encode('utf-8')
                            #st.download_button(
                            #    label="📥 Télécharger le résultat (CSV)",
                            #    data=csv,
                            #    file_name="donnees_traitees.csv",
                            #    mime="text/csv",
                            #    key=f"dl_{len(st.session_state.messages)}"
                            #)
                            
                            #st.session_state.dataframe = df_resultat
                        
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