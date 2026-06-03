# chat_ui.py
import json
import streamlit as st
import requests
import uuid
import time
import os

# --- CONFIGURATION GLOBALE ---
LOGO_PATH = "ressource/Eau_de_Paris_bleu.svg.png"
API_URL = os.environ.get("API_URL", "http://backend:8000")
DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "ministral-3:14b")
DEFAULT_VLM = os.environ.get("DEFAULT_VLM", "ministral-3:14b")
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 12288))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.3))
PAYLOAD_DEBUG = os.environ.get("PAYLOAD_DEBUG", "hide")


def init_session_state():
    """Initialise ou restaure les variables de session pour résister aux rechargements (F5)."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "think_mode" not in st.session_state:
        st.session_state.think_mode = False

def reset_and_rerun():
    """Réinitialise le chat et recharge la page."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.processed_files = []
    st.rerun()

def render_chat(title="Chatbot EDP"):
    """
    Fonction principale pour générer l'interface de chat.
    Peut être appelée depuis n'importe quelle page Streamlit.
    """
    init_session_state()

    if os.path.exists(LOGO_PATH):
        st.logo(LOGO_PATH) 

    # --- SIDEBAR ---
    with st.sidebar:
        st.divider()
        st.session_state.think_mode = st.toggle(
            "Mode raisonnement",
            value=st.session_state.think_mode,
            help="Active le mode 'raisonnement' des modèles pour une réflexion approfondie"
        )
        st.divider()
        if st.button("Nouvelle session", use_container_width=True):
            reset_and_rerun()

    st.title(title)

    # 1. AFFICHAGE DE L'HISTORIQUE
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["display_content"])

    user_input = st.chat_input(
        "Votre message...",
        accept_file=True,
        file_type=["pdf", "txt", "md", "docx", "pptx", "jpg", "webp", "png"],
    )

    # 2. TRAITEMENT DE L'ENTRÉE UTILISATEUR
    if user_input:
        file_list = ""
        conversation_contexte = ""
        nom_fichiers = []
        contenu_fichiers = []
        
        # --- A. GESTION DES FICHIERS ---
        if hasattr(user_input, "files") and user_input.files:
            for fichier_joint in user_input.files:
                st.session_state.processed_files.append(fichier_joint.name)
                file_list += f"📎 **Fichier joint :** {fichier_joint.name}\n"
            
                files = {"file": (fichier_joint.name, fichier_joint.getvalue(), fichier_joint.type)}
                data = {"modele": DEFAULT_VLM}
                
                reponse = requests.post(f"{API_URL}/upload_fichier", files=files, data=data)
                
                if reponse.status_code == 200:
                    contenu_extrait = reponse.json().get("contenu", "Fichier vide.")
                    contenu_fichiers.append(contenu_extrait)
                    nom_fichiers.append(fichier_joint.name)
                    conversation_contexte += f"📄 **Contexte du fichier ({fichier_joint.name}) :**\n{contenu_extrait}\n\n---\n\n"
                else:
                    st.error(f"Erreur d'analyse pour {fichier_joint.name}")

        # --- B. GESTION DU TEXTE ---
        user_text = ""
        if hasattr(user_input, "text"):
            user_text = user_input.text
        elif isinstance(user_input, str):
            user_text = user_input

        instruction = user_text if user_text else "Peux-tu analyser ce document et m'en faire un résumé ?"

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

        # --- C. APPEL DU CHATBOT (STREAMING) ---
        with st.chat_message("assistant"):
            start_time = time.time()
            
            payload = {
                "messages": messages_pour_api,
                "modele": DEFAULT_LLM,
                "temperature": TEMPERATURE,
                "context_size": CONTEXT_SIZE,
                "think": st.session_state.think_mode,
            }

            with st.sidebar:
                st.info(f"Si la réponse indique 'Token prompt utilisateur : {CONTEXT_SIZE}', alors les informations les plus anciennes de la conversation ont été oubliées.")
                if PAYLOAD_DEBUG == "show": 
                    st.subheader("🔍 Debug — Payload")
                    st.json(payload)
                    st.caption(f"Nombre de messages : {len(messages_pour_api)}")

            mes_stats = {}

            def lire_flux_api(stats_container):
                with requests.post(f"{API_URL}/chat", json=payload, stream=True) as r:
                    r.raise_for_status() 
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            texte = chunk.decode("utf-8")
                            if "STATS_JSON:" in texte:
                                parties = texte.split("STATS_JSON:")
                                if parties[0]:
                                    yield parties[0]
                                stats_recues = json.loads(parties[1])
                                stats_container.update(stats_recues)
                            else:
                                yield texte

            full_response = st.write_stream(lire_flux_api(mes_stats))

            if mes_stats:
                st.caption(f"Tokens prompt utilisateur : {mes_stats.get('prompt_tokens')} --- "
                           f"Tokens de la sortie llm : {mes_stats.get('completion_tokens')} --- "
                           f"Durée de génération : {mes_stats.get('duration'):.2f} secondes")
            
            end_time = time.time()

        st.session_state.messages.append({
            "role": "assistant", 
            "display_content": full_response,
            "content": full_response
        })