#backend/routers/files.py

"""
Routeur API : Gestion des Fichiers
Description : Gère les points d'entrée pour l'importation (upload), la suppression et le suivi 
              des documents soumis par l'utilisateur pour alimenter le chatbot.
"""

import io
import time
import asyncio
from fastapi import APIRouter, UploadFile, File, Form
from utils.file_type_action import analyser_contenu_fichier

router = APIRouter(tags=["Files"])
verrou_vlm_image = asyncio.Semaphore(1)

@router.post("/upload_fichier")
async def traiter_fichier(file: UploadFile = File(...), modele: str = Form(...)):
    try:
        file_bytes = await file.read()
        faux_fichier_streamlit = io.BytesIO(file_bytes)
        faux_fichier_streamlit.name = file.filename
        faux_fichier_streamlit.type = file.content_type
        async with verrou_vlm_image:
            contenu = analyser_contenu_fichier(faux_fichier_streamlit, modele)
        return {"nom_fichier": file.filename, "contenu": contenu}
    except Exception as e:
        print(f"Erreur backend lors de l'analyse : {str(e)}")
        return {"erreur": str(e)}