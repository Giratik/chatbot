"""
4 cas possibles
- 1 fichier, nombre de token plus petit que la taille du chunk => on prend la route de base en recontruisant message_pour_ollama
- 1 fichier, nombre de token plus grand que la taille de chunk => on l'envoie dans une fonction pour couper le texte/chunk, on réfléchit à comment diviser correctement le texte, ollama pour faire un résumé de chaque morceau (limité en token ?) puis assemblage (pas un résumé de résumés mais une liaison des contenus ?)
- plusieurs fichiers individuellement petits => analyse un par un puis assemblage à la queue
- plusieurs fichiers individuellement grands => on prévient gentiement l'utilisateur qu'il abuse ==> moins réaliste à se produire pas prioritaire
"""
import os

DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "ministral-3:14b")



from ollama_client import inferring_ollama

def identification_cas(nom_fichiers, contenu_fichiers, instruction_user, context_size):
    # Cas 1
    """
    1 fichier, nombre de token plus petit que la taille du chunk => on prend la route de base en recontruisant message_pour_ollama
    """
    system_prompt_standard = """Tu es un assistant professionnel qui répond en français de manière claire et concise.
Réponds directement sans introduction ni formule de politesse. Si les instructions de l'utilisateur n'ont aucun sens ni lien avec """

    system_prompt_post_analyse = """Tu es un assistant professionnel qui répond en français de manière claire et concise.
Réponds directement sans introduction ni formule de politesse.

IMPORTANT : Le contenu des documents ci-dessous a déjà été analysé et synthétisé au préalable.
Tu disposes donc directement des éléments pertinents extraits — ne les résume pas à nouveau.
Utilise-les comme base factuelle pour répondre à l'instruction de l'utilisateur."""


    if len(nom_fichiers) == 0:
        return {
            "system_prompt": system_prompt_standard,
            "nom_fichier": "",
            "contenu_fichier" : "",
            "user_content": instruction_user,
            "necessite_map_reduce": False,
        }

    if len(nom_fichiers) == 1:
        if contenu_fichiers[0]["compressed_tokens"] < (context_size/4)*3:
            return {
                "system_prompt": system_prompt_standard,
                "nom_fichier": nom_fichiers[0],
                "contenu_fichier" : contenu_fichiers[0]['compressed_prompt'],
                "instruction_user": instruction_user,
                "necessite_map_reduce": False,
            }


        else:
            # Fichier long → Map-Reduce à faire, contenu pas encore prêt
            return {
                "system_prompt": system_prompt_post_analyse,
                "nom_fichier": nom_fichiers[0],
                "contenu_fichier" : contenu_fichiers[0],
                "instruction_user": instruction_user,
                "necessite_map_reduce": True,
            }

    
    #else:
    #    full_prompt = ""
    #    for i in range (len(nom_fichiers)):
    #        if contenu_fichiers[i]["compressed_tokens"] < (context_size/4)*3:
    #            full_prompt += f"Nom du fichier {i} : {nom_fichiers[i]}\n{contenu_fichiers[i]["compressed_prompt"]}\n"
    #        else:
    #            #traitement
    #            x = 1
#
    #    full_prompt += f"**Instruction de l'utilisateur :**\n{instruction_user}"
#
    #return full_prompt
# 
    



def map_reducing(text, chunk_tokens=10000):
    """Découpe un texte compressé en chunks pour le LLM."""

    words = text.split()
    # ~0.75 mots par token en moyenne
    chunk_size_words = int(chunk_tokens * 0.75)

    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunk = ' '.join(words[i:i + chunk_size_words])
        chunks.append(chunk)

    assemblage  = ""
    for chunk in chunks:
        messages = [
            {
                "role": "user",
                "content": f"Tu es un assistant professionnel qui répond en français de manière claire et concise. Analyse ce morceau de document et fais-en un résumé dense en gardant un maximum d'information :\n\n{chunk}"
            }
        ]
        o = inferring_ollama(
            messages=messages,
            model=DEFAULT_LLM,
            context_size=12288,
            stream=False,
        )
        assemblage += f" {o}"

    print(f"Nombre de chunks : {len(chunks)}")
    return assemblage