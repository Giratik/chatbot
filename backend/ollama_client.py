import os
import time
import httpx
#import ollama
from ollama import Client

#from metrics import llm_latency, llm_tokens_generated

URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = Client(
    host=URL_OLLAMA,
    timeout=httpx.Timeout(
        connect=5.0,    # Connexion au serveur
        read=600.0,     # Attente de la réponse (le plus important)
        write=10.0,     # Envoi du prompt
        pool=5.0        # Attente d'une connexion disponible
    )
)

def inferring_ollama(messages, model, temperature=0.4, stream=False, stats_dict=None, context_size=32768, **kwargs):
    # Appel à l'API avec le paramètre stream
    start = time.time()
    #with llm_latency.time():
    response = client.chat(
        model=model,
        messages=messages,
        options={
            "temperature": temperature,
            "num_ctx": context_size,
            #"num_predict": max_tokens  # Limite les tokens générés
        },
        stream=stream 
    )
    duration = time.time() - start
    #if hasattr(response, 'eval_count'):
    #            llm_tokens_generated.inc(response.eval_count)

    if stream==False:
        return response['message']['content']
    else:
        return _stream_response(response, stats_dict)


def _stream_response(response, stats_dict=None):
    """Générateur pour streamer les réponses."""
    for chunk in response:
        # Si c'est le dernier morceau et qu'on a fourni un dictionnaire
        if chunk.get('done') and stats_dict is not None:
            # On récupère les tokens du prompt (entrée)
            stats_dict['prompt_tokens'] = chunk.get('prompt_eval_count', 0)
            # On récupère les tokens de la génération (sortie)
            stats_dict['completion_tokens'] = chunk.get('eval_count', 0)

            stats_dict['duration'] = chunk.get('total_duration', 0) / 1e9  # Convertir de nanosecondes à secondes
            
        # On yield uniquement le texte pour ne pas perturber Streamlit
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']