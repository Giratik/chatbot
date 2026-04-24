import json

import streamlit as st
import requests
import uuid
import time
import os

# --- CONFIGURATION ---
LOGO_PATH = "ressource/Eau_de_Paris_bleu.svg.png"
API_URL = os.environ.get("API_URL", "http://backend:8000")


DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "ministral-3:14b")
DEFAULT_VLM = os.environ.get("DEFAULT_VLM", "ministral-3:14b")
CONTEXT_SIZE = os.environ.get("CONTEXT_SIZE", 12288)
TEMPERATURE = os.environ.get("TEMPERATURE", 0.3)
PAYLOAD_DEBUG = os.environ.get("PAYLOAD_DEBUG", "hide")


#selected_model = "mistral:latest"
#model_vlm_choix = "mistral-small3.2:24b"


# --- FONCTIONS DE SESSION ---
# Vérifie que le fichier n'a pas déjà été traité
#if "processed_files" not in st.session_state:
if "processed_files" :
    st.session_state.processed_files = []

def new_session():
    # Création d'une session pour l'utilisateur à laquelle est rattaché la conversation du chat
    # On force un identifiant fixe pour le test de redis (qui n'est pas utilisé ici)
    #st.session_state.session_id = "utilisateur_test_absolu"
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.processed_files = []
    st.session_state.nom_fichiers = []
    st.session_state.contenu_fichiers = []
    st.instruction_user = ""

def reset_and_rerun():
    # fonction pour réinitialiser le chat, équivaut à recharger sa page avec f5
    new_session()
    st.rerun()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------------



# --- INTERFACE ---
st.set_page_config(page_title="Chatbot EDP", page_icon="💧", layout="wide")

if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH) 

if "session_id" not in st.session_state:
    new_session()

if st.sidebar.button("Nouvelle session", use_container_width=True):
    reset_and_rerun()
st.title("Chatbot EDP")

# 1. AFFICHAGE DE L'HISTORIQUE
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["display_content"])

user_input = st.chat_input(
    "Votre message...",
    accept_file=True,
    #accept_file="multiple",
    file_type= None,
    accept_audio = False,
)

# 2. TRAITEMENT DE L'ENTRÉE UTILISATEUR
if user_input :
    file_list = ""
    conversation_contexte = ""
    nom_fichiers = []
    contenu_fichiers = []
    
    # --- A. GESTION DES FICHIERS (REQUÊTE HTTP) ---
    if hasattr(user_input, "files") and user_input.files:
        i=0
        
        for fichier_joint in user_input.files:
            #if fichier_joint.name not in st.session_state.processed_files:
            st.session_state.processed_files.append(fichier_joint.name)
            fichier_joint = user_input.files[i]
            file_list += f"📎 **Fichier joint :** {fichier_joint.name}\n"
        
            # On prépare le fichier et les paramètres pour l'API
            files = {"file": (fichier_joint.name, fichier_joint.getvalue(), fichier_joint.type)}
            data = {"modele": DEFAULT_VLM}
            
            # On envoie le fichier à FastAPI
            reponse = requests.post(f"{API_URL}/upload_fichier", files=files, data=data)
            
            if reponse.status_code == 200:
                contenu_extrait = reponse.json().get("contenu", "Fichier vide.")
                contenu_fichiers.append(contenu_extrait)
                nom_fichiers.append(fichier_joint.name)
                conversation_contexte += f"📄 **Contexte du fichier ({fichier_joint.name}) :**\n{contenu_extrait}\n\n---\n\n"

            else:
                st.error(f"Erreur d'analyse pour {fichier_joint.name}")

            i+=1

    # --- B. GESTION DU TEXTE ---
    user_text = ""
    if hasattr(user_input, "text"):
        user_text = user_input.text
    elif isinstance(user_input, str):
        user_text = user_input

    #st.text_area(conversation_contexte)
    instruction = user_text if user_text else "Peux-tu analyser ce(s) document(s) et m'en faire un résumé ?"

    # display_content : vu par l'utilisateur / content : vu par l'IA
    display_text = f"{file_list}\n{instruction}" if file_list else instruction
    #instruction_v2 = f"{instruction} **Attention :** si {fichier_joint.name} est un PDF, n'effectue absolument aucun résumé, retourne le texte tel quel."
    llm_text = f"{conversation_contexte} **Instruction de l'utilisateur :**\n{instruction}"

    #payload_build = {
    #        "nom_fichiers":nom_fichiers,
    #        "contenu_fichiers":contenu_fichiers,
    #        "instruction_user":instruction,
    #        "context_size": 12000,
    #    }

    #with requests.post(f"{API_URL}/création_prompt_user", json=payload_build) as r:
    #    print("Status code:", r.status_code)
    #    print("Réponse brute:", r.text)  # ← montre ce que le backend a vraiment renvoyé
    #    print("hehehe")
    #    print("Status code:", r.status_code)
    #    print("hehehe")
    #    print("Headers:", r.headers.get("content-type"))
    #    reponse = r.json()
    st.session_state.messages.append({
        "role": "user", 
        "display_content": display_text,     
        "content": llm_text      
    })
        #prompt = reponse["system_prompt"]

    messages_pour_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] # ==> tableau donc chaque entrée c'est rôle + content

    with st.chat_message("user"):
        st.markdown(display_text)

    # --- C. APPEL DU CHATBOT (REQUÊTE HTTP EN STREAMING) ---
    with st.chat_message("assistant"):
        start_time = time.time()

        
        payload = {
            "messages": messages_pour_api,
            "modele": DEFAULT_LLM,
            "temperature": TEMPERATURE,
            "context_size": CONTEXT_SIZE,
            #"prompt": prompt
            #"nom_fichiers": nom_fichiers,
            #"contenu_fichiers": contenu_fichiers,
            #"instruction_user": instruction
        }
        with st.sidebar:
            if PAYLOAD_DEBUG == "show" : st.subheader("🔍 Debug — Payload")
            if PAYLOAD_DEBUG == "show" : st.json(messages_pour_api)
            #if PAYLOAD_DEBUG == "show" : st.text(prompt)
            #if PAYLOAD_DEBUG == "show" : st.json(payload["messages"])
            #if PAYLOAD_DEBUG == "show" : st.json(payload["messages"]) # comment récupérer le dernier message du json ?
            # on agrandit spécifiquement le payload pour y mettre le dernier message ?
            # ça n'a pas de sens car ça prendrait en compte tous les messages, même ceux qui ne sont pas des fichiers
            # il faudrait vraiment effectuer le traitement du texte avant, spécifiquement pour les fichiers uploadé
            # donc réfléchir à comment faire le chunking, où le faire, quand le traiter avec l'ia, pourquoi pas avec une route supplémentaire et un prompt spécifique
            # peut-être modifier message pour api pour avoir une option instruction utilisateur séparé du texte envoyé ==> ne résout pas le problème que ça concernerait tous les messages, quoique si
            # il faut séparer llm_texte en deux car il contient le texte extrait des pièces jointes et les instructions de l'utilisateur
            # on pourrait même séparer le texte extrait du nom du fichier
            if PAYLOAD_DEBUG == "show" : st.caption(f"Nombre de messages : {len(messages_pour_api)}")
#
        # Fonction génératrice pour lire le flux venant de FastAPI
        stats_container = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}  # Dictionnaire pour stocker les stats du LLM
        def lire_flux_api(stats_container):
            # On se connecte à FastAPI avec l'option stream=True
            with requests.post(f"{API_URL}/chat", json=payload, stream=True) as r:
                r.raise_for_status() # Lève une erreur si la connexion échoue
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        #yield chunk.decode("utf-8")
                        texte = chunk.decode("utf-8")

        # On vérifie si le morceau contient nos stats
                        if "STATS_JSON:" in texte:
                            parties = texte.split("STATS_JSON:")
                            # On yield le texte avant les stats s'il y en a
                            if parties[0]:
                                print("yeald")
                                yield parties[0]
                            # On parse le JSON et on met à jour notre dict
                            stats_recues = json.loads(parties[1])
                            stats_container.update(stats_recues)
                            print("Stats mises à jour dans le container : ", stats_container)
                        else:
                            yield texte


        ## Streamlit affiche le texte en temps réel
        #full_response = st.write_stream(lire_flux_api())
        # Utilisation
        mes_stats = {}
        full_response = st.write_stream(lire_flux_api(mes_stats))

        # Maintenant mes_stats est rempli !
        if mes_stats:
            st.caption(f"Tokens prompt utilisateur : {mes_stats.get('prompt_tokens')} --- "
            f"Tokens de la sortie llm : {mes_stats.get('completion_tokens')} --- "
            f"Durée de génération : {mes_stats.get('duration'):.2f} secondes"
            )
        
        
        end_time = time.time() # Arrêt du chronomètre
        elapsed_time = end_time - start_time # Calcul de la durée


    # On sauvegarde la réponse de l'IA dans l'historique
    st.session_state.messages.append({
        "role": "assistant", 
        "display_content": full_response,
        "content": full_response
    })