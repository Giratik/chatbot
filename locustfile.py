from locust import HttpUser, task, between
import json

class ChatbotUser(HttpUser):
    # Simule le temps de lecture et de réflexion d'un humain
    # L'utilisateur attendra entre 5 et 15 secondes avant de reposer une question
    wait_time = between(5, 15)

    @task
    def parler_au_bot(self):
        # Le format exact attendu par ton Pydantic BaseModel (ChatRequest)
        payload = {
            "messages": [
                {"role": "user", "content": "Bonjour, donne-moi trois faits intéressants sur l'espace. Sois concis."}
            ],
            "modele": "ministral-3:14b", # ⚠️ Mets ici le vrai nom du modèle que tu utilises
            "temperature": 0.4,
            #"context_size": 16384
            "context_size": 8192
        }

        # On simule l'envoi du message vers ton backend
        # On met stream=True car ton API utilise une StreamingResponse
        with self.client.post("/chat", json=payload, stream=True, catch_response=True) as response:
            if response.status_code == 200:
                # On consomme le flux (comme le ferait Streamlit) pour mesurer le temps total
                try:
                    for chunk in response.iter_content(chunk_size=1024):
                        pass
                    response.success()
                except Exception as e:
                    response.failure(f"La connexion a été coupée en cours de route : {e}")
            else:
                response.failure(f"Le serveur a répondu avec l'erreur : {response.status_code}")