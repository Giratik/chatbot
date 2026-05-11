#backend/main.py

import json
import shutil
import tempfile



from fastapi import FastAPI, UploadFile, File, Form, Query, APIRouter, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
#from prometheus_fastapi_instrumentator import Instrumentator
from typing import List, Dict, Any
import asyncio
import io
import sys
import os
import time
import threading
import chromadb
import json

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


# Import de tes fonctions locales existantes
from ollama_client import inferring_ollama
from file_type_action import analyser_contenu_fichier
from csv_rag import process_csv_file, get_csv_collection_name, delete_csv_session, get_csv_client
from newer_rag_engine import remplir_database_chroma, recherche_lexique, recherche_depuis_texte, get_collection
from traitement_long_fichier import identification_cas, map_reducing
from excel_parser_robust import find_tables_in_sheet
import robuster

CONTEXT_SIZE = os.environ.get("CONTEXT_SIZE", 12288)
URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")



SYSTEM_PROMPT = """
Tu es "EDP-IA", l'assistant IA officiel de l'entreprise Eau de Paris. 

--- TON IDENTITÉ ET TON RÔLE ---
* Tu es un expert technique, professionnel, mais toujours amical et concis.
* Ton but est d'aider les salariés de EDP à analyser leurs documents et à répondre à leurs questions.
* Tu ne dois jamais inventer d'informations (pas d'hallucinations). Si tu ne sais pas, dis-le simplement.

--- TES CONNAISSANCES DE BASE ---
* L'entreprise se spécialise dans la distribution de l'eau dans la ville de Paris.

--- RÈGLES DE FORMATAGE ---
* Réponds toujours en français.
* Utilise le format Markdown pour structurer tes réponses (listes à puces, texte en gras pour mettre en évidence les éléments clés).
* Ne sois pas trop bavard : va droit au but.
* Répond avec un minimum de déférence.
"""




# Création de l'application FastAPI
app = FastAPI(title="API EDP Chatbot")
#Instrumentator().instrument(app).expose(app)

# On autorise 1 seule analyse d'image à la fois
verrou_vlm_image = asyncio.Semaphore(1)

# --- FORMAT ATTENDU (Retour à la liste complète des messages) ---

#class build_prompt(BaseModel):
#    nom_fichiers: List[str] = []
#    contenu_fichiers: List[Any] = []
#    instruction_user: str
#    context_size: int

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int 
    #nom_fichiers: List[str] = []
    #contenu_fichiers: List[Any] = []
    #instruction_user: str
    #prompt: str = ""

class ChatRequest_csv(BaseModel):
    messages: List[Dict[str, Any]]
    modele: str
    temperature: float
    context_size: int
    colonnes_info: str = ""  # Information sur les colonnes du DataFrame
    csv_knowledge: str = ""  # Connaissance supplémentaire issue du CSV
    session_id: str = "default"

class SheetFile(BaseModel):
    file: UploadFile  # ← le type FastAPI pour les uploads
    onglet_choisi: str

class KnowledgeRequest(BaseModel):
    session_id: str
    tableaux: List[Dict[str, Any]]  # Format: [{"titre": "...", "donnees": [...]}]


def routine_demarrage():
    remplir_database_chroma()    # On lance l'injection du JSON dans ChromaDB

@app.on_event("startup")
def startup_event():
    # On lance ça en tâche de fond pour que FastAPI puisse démarrer tout de suite
    threading.Thread(target=routine_demarrage).start()

@app.get("/lexique")
def lexique():
    """Affiche le contenu actuel du lexique pour vérification."""
    client = chromadb.PersistentClient(path="./chromadb")
    
    try:
        # Pensez à vérifier que c'est bien le bon nom de collection !
        # (Si vous avez utilisé le script mis à jour précédemment, 
        # le nom est peut-être "base_connaissances_globale")
        collection = client.get_collection(name="base_connaissances_globale_acronymes")
        
        # Un seul appel suffit pour tout récupérer
        tous_les_docs = collection.get() 
        
        # On vérifie si la base est vide pour éviter un crash
        if not tous_les_docs['documents']:
            return {"message": "La base est actuellement vide."}

        # On renvoie la liste des documents
        return {
            "total_acronymes": len(tous_les_docs['documents']),
            "donnees": tous_les_docs['documents']
        }

    except ValueError:
        return {"erreur": "La collection n'existe pas. Avez-vous lancé le script d'ingestion ?"}

# --- ROUTE 1 : ANALYSE DE FICHIER ---
@app.post("/upload_fichier")
async def traiter_fichier(file: UploadFile = File(...), modele: str = Form(...)):
    """Reçoit un fichier de Streamlit, l'analyse, et renvoie le texte extrait."""
    #active_sessions.inc()
    start_time = time.time()
    
    try:
        file_bytes = await file.read()
        faux_fichier_streamlit = io.BytesIO(file_bytes)
        faux_fichier_streamlit.name = file.filename
        faux_fichier_streamlit.type = file.content_type
        
        async with verrou_vlm_image:
            contenu = analyser_contenu_fichier(faux_fichier_streamlit, modele)
        
        # Enregistrer la métrique
        elapsed = time.time() - start_time
        #upload_file_latency.observe(elapsed)
        #files_processed_total.inc()
        
        return {"nom_fichier": file.filename, "contenu": contenu}
        
    except Exception as e:
        print(f"Erreur backend lors de l'analyse : {str(e)}")
        elapsed = time.time() - start_time
    #    upload_file_latency.observe(elapsed)
    #    endpoint_errors.labels(endpoint="/upload_fichier", error_type=type(e).__name__).inc()
        return {"erreur": str(e)}
    #finally:
    #    active_sessions.dec()



#@app.post("/création_prompt_user")
#async def creation_prompt_user(requete: build_prompt):
#    resultat = identification_cas(
#            nom_fichiers=requete.nom_fichiers,
#            contenu_fichiers=requete.contenu_fichiers,
#            instruction_user= requete.instruction_user,
#            context_size=requete.context_size,
#        )
#    
#    conversation_contexte = ""
#    
#    # Cas sans fichier : on envoie juste l'instruction utilisateur
#    if len(requete.nom_fichiers) == 0:
#        llm_text = requete.instruction_user
#    elif resultat["necessite_map_reduce"] == True:
#        print(">>> CAS MAP REDUCE")  # ← ajoute ça
#        conversation_contexte = map_reducing(requete.contenu_fichiers[0]["compressed_prompt"])
#        llm_text = f"Nom du fichier : {resultat['nom_fichier']}: {conversation_contexte} **Instruction de l'utilisateur :**\n{requete.instruction_user}"
#    else:
#        print(">>> CAS DIRECT")      # ← et ça
#        llm_text = f"Nom du fichier : {resultat['nom_fichier']}: {resultat['contenu_fichier']} **Instruction de l'utilisateur :**\n{requete.instruction_user}"
#
#
#
#    return {
#        "role": "user",
#        "content": llm_text,
#        "nom_fichiers":requete.nom_fichiers,
#        "contenu_fichiers":requete.contenu_fichiers,
#        "instruction_user":requete.instruction_user,
#        "system_prompt":resultat["system_prompt"]
#    }




# --- ROUTE 2 : GÉNÉRATION DU CHAT (STREAMING) ---
@app.post("/chat")
async def generer_chat(requete: ChatRequest):
    """Reçoit l'historique complet, injecte le prompt système et renvoie un flux texte."""
    #active_sessions.inc()
    start_time = time.time()
    #s'il y a des fichiers on commence le traitement
    # on envoie le nom des fichiers (liste) ainsi que leur contenu
    # on fait le résumé de la partie i, i+1, etc. plus ou moins long
    # on peut aussi récupérer le compressed_token pour dire:
    # - si le fichier est assez petit (+ que la taille du chunk) on saute l'opération et on passe au fichier suivant

    try:
        stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}  # Dictionnaire pour stocker les stats du LLM

        #system_prompt = """Tu es un assistant professionnel qui répond aux utilisateurs de manière claire, concise et informative en français. Tu fournis des réponses précises et utiles en fonction des questions posées.
        #Réponds directement sans introduction ni formule de politesse."""

        # On place le prompt système au tout début, suivi de l'historique envoyé par Streamlit
        messages_pour_ollama = [{"role": "system", "content": SYSTEM_PROMPT}] + requete.messages

        # 2. Création du générateur (Streaming)
        def stream_generator():
            # Appel direct à ton modèle local
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                stats_dict = stats_dict,
                context_size=requete.context_size,
            ):
                yield chunk
        
        # Mesurer la latence au moment du streaming
            elapsed = time.time() - start_time
            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"
        # 3. Renvoi des données en temps réel à Streamlit
        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        print(f"Erreur backend lors de la rédaction par llm : {str(e)}")
        elapsed = time.time() - start_time


# --- ROUTE 2 : GÉNÉRATION DU CHAT (STREAMING) ---
@app.post("/chat_with_rag")
async def generer_chat(requete: ChatRequest):
    """Reçoit l'historique complet, injecte le prompt système et renvoie un flux texte."""
    # 1. On cherche si la question contient des acronymes connus
    #contexte_lexique = recherche_lexique(requete.messages[-1]["content"])
    texte_user = requete.messages[-1]["content"]
    acronymes_resolus = recherche_depuis_texte(texte_user, get_collection())
    contexte = "\n".join(
    f"{acr} = {sig}" for acr, sig in acronymes_resolus.items()
)
    #active_sessions.inc()
    start_time = time.time()

    try:

        stats_dict = {"prompt_tokens": 0, "completion_tokens": 0, "duration": 0}  # Dictionnaire pour stocker les stats du LLM
        
        system_prompt = f"""{SYSTEM_PROMPT} \nVoici le lexique des acronymes de Eau de Paris détectés dans la question :
        {contexte}. 
        ATTENTION : n'invente aucune information, si l'acronyme apparait dans le lexique tu ne peux utiliser dans ta réponse que la définition associée et pas provenant de tes connaissances.
        Si la requête de l'utilisateur comporte un acronyme qui n'apparait pas dans ton lexique, tu dois l'en informer et à ce moment utilise tes connassances pour répondre.
        Si la requête de l'utilisateur n'a rien à voir avec les acronymes détectés, tu ignores les acronymes détectés et converse normalement avec l'utilisateur. Requête de l'utilisateur : """
        # 2. On enrichit le message système avec ces définitions
        prompt_lexique = ""
        #if contexte_lexique:
        #    print(f"🔍 Contexte de lexique trouvé pour l'acronyme : {contexte_lexique}")
        #    system_prompt = f"""Tu es un assistant professionnel qui répond aux utilisateurs de manière claire, concise et informative en français. Tu fournis des réponses précises et utiles en fonction des questions posées.
        #Réponds directement sans introduction ni formule de politesse.\n\nLexique utile pour répondre : {contexte_lexique}. """
        #    print(system_prompt)
        #else:
        #    print("🔍 Aucun contexte de lexique trouvé pour la question posée.")
        ## On place le prompt système au tout début, suivi de l'historique envoyé par Streamlit
        messages_pour_ollama = [{"role": "system", "content": system_prompt}] + requete.messages

        # 2. Création du générateur (Streaming)
        def stream_generator():
            # Appel direct à ton modèle local
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                stats_dict = stats_dict,
                context_size=requete.context_size,
                #max_tokens=600  # Limite la longueur des réponses
            ):
                yield chunk
        
        # Mesurer la latence au moment du streaming
            elapsed = time.time() - start_time
        #chat_latency.observe(elapsed)
            yield f"\nSTATS_JSON:{json.dumps(stats_dict)}"
        # 3. Renvoi des données en temps réel à Streamlit
        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        print(f"Erreur backend lors de la rédaction par llm : {str(e)}")
        elapsed = time.time() - start_time









# --- ROUTE 3 : ASSISTANT DATA ANALYST (STREAMING) ---
@app.post("/chat_data_analyst")
async def generer_chat_data_analyst(requete: ChatRequest_csv):
    """Reçoit l'historique complet pour un assistant data analyst, génère du code Python."""
    start_time = time.time()

    try:
        # 1. NOUVEAU PROMPT : Le LLM manipule un dictionnaire 'dfs'
        system_prompt = f"""{SYSTEM_PROMPT} \n Mais ici, tu es un expert en data science Python. 
Tu as à ta disposition un dictionnaire Python nommé `dfs` contenant plusieurs DataFrames pandas.
Les clés de ce dictionnaire sont les noms des tableaux. Voici les tableaux disponibles dans `dfs` et leurs colonnes :
{requete.colonnes_info}

L'utilisateur va te faire une demande. Selon la demande, tu dois générer du code Python pour faire l'une de ces deux actions (ou les deux) :

ACTION A - GRAPHIQUES : Si on te demande un graphique, utilise `plotly.express` (importé sous `px`). Extraie le bon DataFrame depuis le dictionnaire (ex: `df = dfs["Nom_du_tableau"]`) et stocke le résultat OBLIGATOIREMENT dans une variable nommée `fig`.

ACTION B - FILTRAGE / MODIFICATION : Si on te demande de manipuler les données, extraie le bon DataFrame depuis `dfs`, fais les opérations et stocke le DataFrame final OBLIGATOIREMENT dans une variable nommée `df_resultat`.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec le code Python, entouré par ```python et ```.
2. UTILISE TOUJOURS les noms de colonnes et les clés du dictionnaire EXACTS fournis ci-dessus.
3. N'utilise JAMAIS de guillemets simples (') pour les noms de colonnes ou clés. Utilise TOUJOURS des guillemets doubles (").
4. Ne recrée pas les imports de pandas ou plotly.
5. IMPORTANT - GESTION DES ACCENTS : Le DataFrame a été pré-traité. Les noms de colonnes et données n'ont plus d'accents. Retire les accents dans la valeur de tes filtres.
6. Donne UNIQUEMENT le code, sans explications."""
        
        messages_pour_ollama = [{"role": "system", "content": system_prompt}] + requete.messages

        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                context_size=requete.context_size,
                max_tokens=800 
            ):
                yield chunk
            
        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        print(f"Erreur backend lors de la rédaction par llm : {str(e)}")
        elapsed = time.time() - start_time


@app.post("/ajouter_au_savoir_csv")
async def ajouter_au_savoir_csv(file: UploadFile = File(...), session_id: str = Form("default")):
    """Reçoit un fichier csv, crée la base chroma_db et la sauvegarde."""
    try:
        # Création d'un fichier physique temporaire pour le CSVLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Appel à csv_rag.py avec session_id
        nombre_docs = process_csv_file(temp_file_path, session_id)
        
        # Nettoyage
        os.remove(temp_file_path)

        # On retourne une réponse JSON propre
        return {"statut": "succès", "documents_ajoutes": nombre_docs, "session_id": session_id}

    except Exception as e:
        print(f"Erreur : {str(e)}")
        return {"erreur": str(e)}
    

class SessionRequest(BaseModel):
    session_id: str = "default"


@app.post("/cleanup_csv_session")
async def cleanup_csv_session(request: SessionRequest):
    """Supprime la collection CSV associée à une session utilisateur."""
    delete_csv_session(request.session_id)
    return {"statut": "supprimé", "session_id": request.session_id}


# ==========================================
# ROUTE 3 : LE CERVEAU "DISCUSSION" (RAG)
# ==========================================
@app.post("/chat_csv_rag")
async def generer_chat_csv_rag(requete: ChatRequest_csv):
    """Fouille dans ChromaDB pour discuter du contenu textuel du CSV."""
    try:
        try:
            # On récupère le client en mémoire
            client = robuster.get_csv_client(requete.session_id)
            session_collection_name = robuster.get_csv_collection_name(requete.session_id)
            
            # Vérification si la collection existe nativement
            collection_native = client.get_collection(session_collection_name)
        except ValueError:
            # Si get_collection échoue, c'est que la base est vide
            def erreur_stream():
                yield "❌ La base de données est vide. Veuillez indexer le fichier avant de poser une question."
            return StreamingResponse(erreur_stream(), media_type="text/plain")

        # 1. Reconnexion de LangChain à la base existante
        # (N'oubliez pas l'URL forcée qui nous a sauvé tout à l'heure !)
        embeddings = OllamaEmbeddings(
            model="embeddinggemma",
            base_url=URL_OLLAMA
        )
        
        vectorstore = Chroma(
            client=client,
            collection_name=session_collection_name,
            embedding_function=embeddings
        )

        # 2. On récupère la dernière question
        derniere_question = requete.messages[-1]["content"]
        
        # 3. Recherche via LangChain (similarity_search)
        resultats_docs = vectorstore.similarity_search(derniere_question, k=5)
        
        # Extraction du texte des documents LangChain
        if resultats_docs:
            contexte_extrait = "\n\n---\n\n".join([doc.page_content for doc in resultats_docs])
        else:
            contexte_extrait = "Je n'ai rien trouvé d'utile dans le document."

        # 4. Création du Prompt Système (inchangé, c'est parfait)
        system_prompt = f"""Tu es un assistant expert pour analyser un document.
Ton but est de répondre aux questions de l'utilisateur de manière naturelle et conversationnelle.

Voici les informations extraites du document pour répondre à la question :
{contexte_extrait}

RÈGLES :
1. Base-toi UNIQUEMENT sur les informations fournies ci-dessus.
2. Si la réponse ne se trouve pas dans ces informations, dis simplement que tu n'as pas l'information dans le document.
3. Ne propose JAMAIS de code Python. Formule une réponse claire et textuelle."""

        messages_pour_ollama = [{"role": "system", "content": system_prompt}] + requete.messages

        # 5. Appel au LLM en streaming (inchangé)
        def stream_generator():
            for chunk in inferring_ollama(
                messages=messages_pour_ollama,
                model=requete.modele,
                temperature=requete.temperature,
                stream=True,
                context_size=requete.context_size,
                max_tokens=1500 
            ):
                yield chunk

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
            message_erreur = str(e) 
            print(f"Erreur backend RAG : {message_erreur}")
            
            def erreur_fatale():
                yield f"❌ Erreur critique lors de la discussion : {message_erreur}"
                
            return StreamingResponse(erreur_fatale(), media_type="text/plain")
    



@app.post("/parse_excel")
async def parse_excel_route(
    file: UploadFile = File(...),
    sheet_name: str = Query("Sheet1")
):
    """
    Parse un Excel et retourne UN tableau spécifique.
    """
    try:
        tableaux = find_tables_in_sheet(file.file, sheet_name=sheet_name)
        
        if not tableaux:
            return {"status": "error", "message": "Aucun tableau trouvé"}
        
        # Retourne le premier tableau détecté
        df = tableaux[0]["dataframe"]
        
        return {
            "status": "success",
            "nom_fichier": file.filename,
            "tableau": df.to_dict(orient="records"),  # ← format JSON
            "colonnes": list(df.columns),
            "shape": df.shape
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@app.post("/get_tableau")
async def get_tableau(file: UploadFile = File(...), tableau_idx: int = 0):
    """
    Récupère un tableau spécifique en tant que CSV/JSON.
    """
    try:
        tableaux = find_tables_in_sheet(file.file, sheet_name="Sheet1")
        
        if tableau_idx >= len(tableaux):
            return {"status": "error", "message": f"Tableau {tableau_idx} inexistant"}
        
        df = tableaux[tableau_idx]["dataframe"]
        
        return {
            "status": "success",
            "tableau": df.to_dict(orient="records"),
            "nom": tableaux[tableau_idx]["nom"]
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    


@app.post("/parse_every_tab_excel")
async def parse_excel_route(
    file: UploadFile = File(...),
    sheet_name: str = Query("Sheet1")
):
    try:
        # 1. Read the uploaded file into memory
        contents = await file.read()
        
        # 2. Wrap it in a BytesIO object (this gives openpyxl the 'seekable' attribute it needs)
        excel_data = io.BytesIO(contents)
        
        # 3. Pass the BytesIO object instead of file.file
        tableaux = robuster.find_tables_in_sheet(excel_data, sheet_name=sheet_name)
        
        if not tableaux:
            return {"status": "error", "message": "Aucun tableau trouvé"}
        
        # 4. Convert the list of tuples into a clean list of JSON objects (from the previous fix)
        formatted_tableaux = [
            {"titre": title, "donnees": data} 
            for title, data in tableaux
        ]
        
        return {
            "status": "success",
            "nom_fichier": file.filename,
            "tableau": formatted_tableaux
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
    


@app.post("/knowledge_graphe")
async def knowledge_graphe(request: KnowledgeRequest):
    """Reçoit les tableaux extraits au format JSON, crée la base chroma_db et la sauvegarde."""
    try:
        # 1. Convertir la liste de dictionnaires JSON en liste de tuples 
        # pour être compatible avec robuster.prepare_rag_documents()
        tables_for_robuster = [
            (tab.get("titre", "Sans titre"), tab.get("donnees", [])) 
            for tab in request.tableaux
        ]

        # 2. Préparation des documents RAG (conversion en DataFrame puis Markdown)
        docs = robuster.prepare_rag_documents(tables_for_robuster)

        # 3. Création du Vector Store (On passe bien le session_id à la fonction !)
        #robuster.create_vector_store(docs, request.session_id, embeddings)
        vector_index = robuster.create_vector_store(docs, request.session_id)
        
        msg = f"{len(docs)} document(s) indexé(s) — base vectorielle prête !"
        print(msg)
        
        # On s'assure de renvoyer le statut attendu par le front ("succès")
        return {"statut": "succès", "message": msg}

    except Exception as e:
        print(f"Erreur lors de l'indexation : {str(e)}")
        # On renvoie aussi un statut d'erreur explicite
        return {"statut": "erreur", "erreur": str(e)}