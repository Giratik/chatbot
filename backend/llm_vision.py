#import os
#import ollama
import os
import httpx
from ollama import Client
import time
#from metrics import llm_latency, llm_tokens_generated

# Utilise la même variable d'env que ollama_client.py
URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CONTEXT_SIZE = os.environ.get("CONTEXT_SIZE", 12288)
client = Client(
    host=URL_OLLAMA,
    timeout=httpx.Timeout(
        connect=5.0,
        read=600.0,  # Pour les images volumineuses
        write=10.0,
        pool=5.0
    )
)
def analyse_image(image_bytes, prompt, model):
    """
    Envoie une image au modèle vlm via Ollama et retourne l'analyse.
    """

    print(f"⏳ Analyse de l'image en cours avec {model}...")
    start = time.time()
    
    try:
        #with llm_latency.time():
        response = client.chat(
            #qwen3-vl:8b est plus rapide que llava:13b, mais moins performant sur les tâches complexes, un peu la flemme de recopier du code mais reconnaît bien les formes. ==> du mal avec le texte 
            # utile pour du global understanding, mais pas pour des analyses très détaillées.
            # ministral-3:14b pour l'instant le plus performant pour reconnaître du texte mais très susceptible aux hallucinations pour les formes.
            # llama3.2-vision:11b à oublier
            # llava:13b à oublier pour l'instant, il est très lent et ne semble pas plus performant que les autres sur les tâches simples.
            # llava:7b à oublier, trop faible.
            # gemma4:e4b très performant globalement sur la reconnaissance de textes et formes, par contre confond des fraises et framboises
            model=model,
            options={
                "temperature": 0.4,
                "num_ctx":CONTEXT_SIZE,
            },
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }
            ]
        )
        duration = time.time() - start
        #if hasattr(response, 'eval_count'):
        #    llm_tokens_generated.inc(response.eval_count)   
        
        return response['message']['content']

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Erreur Ollama (Vision): {error_msg}")
        
        # Messages d'aide spécifiques
        if "refused" in error_msg.lower() or "connection" in error_msg.lower():
            return f"❌ Ollama Vision ne répond pas. Vérifie:\n- Ollama est lancé sur {URL_OLLAMA}\n- Le firewall autorise la connexion\nDétail: {e}"
        elif "timeout" in error_msg.lower():
            return f"❌ Timeout Ollama (image trop grande?). Essaie une image plus petite.\nDétail: {e}"
        else:
            return f"❌ Erreur Ollama Vision: {e}"
