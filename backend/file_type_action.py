import filetype
from llm_vision import analyse_image
from paddle_ocr_processor import process_file_with_ocr  # Utilise PaddleOCR avec GPU

# --- 1. TES FONCTIONS SPÉCIALISÉES (Les ouvriers) ---

def traiter_image(fichier_joint, model_vlm):
    # Appelle ta fonction LLaVA ici
    image_bytes = fichier_joint.getvalue()
    return analyse_image(image_bytes, prompt=f"Voici l'image: Rédige un description détaillée de cette image en français.", model=model_vlm)

def traiter_pdf(fichier_bytes, model_vlm):
    # Appelle ta fonction d'OCR pour PDF
    print(f"Analyse du PDF avec OCR...")
    return process_file_with_ocr(fichier_bytes, file_type="application/pdf")

def traiter_texte(fichier_objet, model_vlm):
    # Le texte brut se décode simplement
    fichier_bytes = fichier_objet.read()
    if hasattr(fichier_objet, 'seek'):
        fichier_objet.seek(0)

        # 2. Tentative de décodage avec gestion d'erreurs
    print(f"Analyse du contenu...")
        # 'errors=replace' évite de crasher sur les fichiers .py avec accents
    contenu_texte = fichier_bytes.decode('utf-8', errors='replace')
        
        # Vérification simple : si le contenu est "propre", on renvoie le texte
        # On peut tester si le caractère de remplacement () est trop présent
    if contenu_texte.count('\ufffd') > (len(contenu_texte) * 0.1):
        raise ValueError("Probablement un fichier binaire")

    return contenu_texte

def format_non_supporte(fichier_bytes, model_vlm):
    return "⚠️ Ce type de contenu n'est pas encore pris en charge."


# --- 2. LE DICTIONNAIRE DE ROUTAGE ---
# On associe un type général à la fonction qui sait s'en occuper
ROUTEUR = {
    "image": traiter_image,
    "pdf": traiter_pdf,
    "texte": traiter_texte
}


# --- 3. LA FONCTION MAGIQUE (Le chef d'orchestre) ---

def analyser_contenu_fichier(uploaded_file, model_vlm):
    """
    Détecte la vraie nature du fichier et applique le bon traitement.
    """
    fichier_bytes = uploaded_file.getvalue()
    
    # filetype analyse les premiers octets (Magic Numbers)
    kind = filetype.guess(fichier_bytes)
    
    # On détermine la catégorie du fichier
    categorie = None
    
    if kind is None:
        # filetype ne détecte pas le texte brut car il n'a pas d'en-tête spécifique.
        # Si c'est None, on tente de le lire comme du texte.
        try:
            fichier_bytes.decode('utf-8')
            categorie = "texte"
        except UnicodeDecodeError:
            categorie = "inconnu"
            
    else:
        # kind.mime renvoie par ex: "image/jpeg" ou "application/pdf"
        mime_principal = kind.mime.split('/')[0]
        extension_reelle = kind.extension
        
        if mime_principal == "image":
            categorie = "image"
        elif extension_reelle == "pdf":
            categorie = "pdf"
        elif mime_principal == "audio":
            categorie = "audio" # Pour plus tard ?

    # --- L'EXÉCUTION DYNAMIQUE ---
    # On va chercher la bonne fonction dans le dictionnaire. 
    # Si la catégorie n'existe pas, on utilise la fonction 'format_non_supporte' par défaut.
    fonction_a_executer = ROUTEUR.get(categorie, format_non_supporte)
    
    # On exécute la fonction et on retourne le résultat
    return fonction_a_executer(uploaded_file, model_vlm)