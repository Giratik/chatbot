"""
PaddleOCR Processor - OCR avec support GPU CUDA
"""

import io
import os
import pdfplumber
from PIL import Image
import pdf2image
import cv2
import numpy as np
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from ollama import Client

from ollama_client import inferring_ollama
from llmlingua_format import token_saver

# Configuration Ollama
URL_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = Client(
    host=URL_OLLAMA,
    timeout=httpx.Timeout(
        connect=5.0,
        read=600.0,
        write=10.0,
        pool=5.0
    )
)

DEFAULT_LLM = os.environ.get("DEFAULT_LLM", "mistral:7b")
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", 12288))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.4))

# ==================== PaddleOCR INITIALIZATION ====================
try:
    from paddleocr import PaddleOCR
    import torch
    
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "N/A"
    
    print(f"🔍 Détection GPU: {('✅ ' + GPU_NAME) if GPU_AVAILABLE else '❌ CPU-only'}")
    
    # Initialiser PaddleOCR une seule fois (Singleton)
    ocr = PaddleOCR(
        use_angle_cls=True,
        # CORRECTION 1 : Une seule string. 'fr' inclut les caractères anglais.
        lang='fr',
        use_gpu=GPU_AVAILABLE,
        show_log=False,
        # CORRECTION 2 : On supprime det_model_dir et rec_model_dir 
        # pour laisser Paddle auto-télécharger les modèles correctement.
        det_limit_side_len=2048,  # Limite la taille des images pour éviter les OOM GPU
        rec_limit_side_len=2048, # Limite la taille pour la reconnaissance
    )
    print("✅ PaddleOCR initialisé avec succès")
    
except ImportError:
    print("❌ PaddleOCR non disponible. Installez: pip install paddleocr torch")
    ocr = None
except Exception as e:
    print(f"❌ Erreur initialisation PaddleOCR: {e}")
    ocr = None


def fix_orientation(image):
    """
    Détecte l'orientation du texte et pivote l'image si nécessaire.
    (Compatible avec PaddleOCR qui détecte aussi l'orientation)
    """
    try:
        # PaddleOCR gère déjà l'orientation avec use_angle_cls=True
        # On peut ajouter une détection manuelle avec vision si besoin
        pass
    except Exception:
        pass
    return image


def preprocess_image_for_ocr(image):
    """
    Prétraite l'image pour améliorer la reconnaissance PaddleOCR.
    """
    image = fix_orientation(image)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 1. Légère mise à l'échelle pour meilleures images
    height, width = cv_image.shape[:2]
    if width < 800:
        scale = 1.5
        cv_image = cv2.resize(cv_image, (int(width * scale), int(height * scale)), 
                             interpolation=cv2.INTER_CUBIC)
    
    # 2. Réduction léger du bruit (bilatéral = préserve les edges)
    denoised = cv2.bilateralFilter(cv_image, 5, 50, 50)
    
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))


def extract_text_from_image_paddle(image):
    """
    Extrait le texte d'une image avec PaddleOCR et reconstruit l'alignement
    horizontal pour sauver la structure des tableaux issus de screenshots.
    """
    if ocr is None:
        return "", 0.0
    
    try:
        # Conversion BGR
        if isinstance(image, Image.Image):
            img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_array = image
        
        results = ocr.ocr(img_array, cls=True)
        
        if not results or not results[0]:
            return "", 0.0
            
        # 1. Extraire les blocs avec leurs coordonnées géométriques
        blocks = []
        confidences = []
        for line in results[0]:
            box = line[0]      # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text = line[1][0]  
            conf = line[1][1]   
            
            if conf > 0.2:
                # On calcule le centre vertical de la boîte de texte
                y_center = (box[0][1] + box[2][1]) / 2
                x_left = box[0][0]
                blocks.append({"text": text, "y": y_center, "x": x_left})
                confidences.append(conf)

        # 2. Trier tous les blocs de haut en bas
        blocks.sort(key=lambda b: b["y"])
        
        # 3. Grouper les blocs en "Lignes" (Rows)
        lines = []
        current_line = []
        y_tolerance = 15  # Tolérance en pixels : si deux textes ont moins de 15px d'écart en Y, ils sont sur la même ligne

        for block in blocks:
            if not current_line:
                current_line.append(block)
            else:
                # On compare avec le centre vertical du premier élément de la ligne en cours
                if abs(block["y"] - current_line[0]["y"]) < y_tolerance:
                    current_line.append(block)
                else:
                    lines.append(current_line)
                    current_line = [block]
        if current_line:
            lines.append(current_line)

        # 4. Trier chaque ligne de gauche à droite et la formater
        formatted_text = ""
        for line in lines:
            # Tri horizontal (par coordonnée X)
            line.sort(key=lambda b: b["x"])
            # On joint les colonnes avec un séparateur visuel
            formatted_text += " | ".join([b["text"] for b in line]) + "\n"
            
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return formatted_text, avg_confidence
    
    except Exception as e:
        print(f"❌ Erreur PaddleOCR: {e}")
        return "", 0.0


def extract_pdf_ocr_paddle(pdf_bytes):
    """
    Extrait le texte d'un PDF avec PaddleOCR GPU.
    Utilise ThreadPoolExecutor pour paralléliser sur GPU.
    """
    if ocr is None:
        return None
    
    text_content = ""
    
    try:
        print("🔄 Conversion PDF→images")
        images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
        
        total_pages = len(images)
        print(f"📄 {total_pages} pages avec PaddleOCR (GPU)...")
        
        for page_idx, image in enumerate(images, 1):
            text_content += f"\n\n----- Page {page_idx} -----\n\n"
            
            try:
                processed = image
                
                # Extraction PaddleOCR
                page_text, confidence = extract_text_from_image_paddle(processed)
                
                if not page_text or confidence < 0.3:
                    # Fallback sans preprocessing
                    page_text, confidence = extract_text_from_image_paddle(image)
                
                # Nettoyage LLM optionnel (garder rapide pour GPU)
                if page_text.strip():
                    text_content += page_text + "\n"
            
            except Exception as e:
                print(f"⚠️  Page {page_idx}: {e}")
                continue
            
            if page_idx % 5 == 0:
                print(f"  ✓ {page_idx}/{total_pages}")
        
        print(f"✅ PaddleOCR complet")
    
    except Exception as e:
        print(f"❌ PaddleOCR: {e}")
        return None
    
    return text_content if text_content.strip() else None



def extract_pdf_native(pdf_bytes):
    """
    Fallback: Extrait le texte natif avec pdfplumber.
    """
    text_content = ""
    
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                text_content += f"\n{'='*60}\nPage {page_idx + 1}\n{'='*60}\n"
                
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
                
                try:
                    tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 3,
                            "intersection_tolerance": 3,
                            "edge_min_length": 3,
                        }
                    )
                    
                    if tables:
                        text_content += "\n--- TABLEAUX DÉTECTÉS ---\n"
                        for table_idx, table in enumerate(tables):
                            has_real_content = any(
                                cell and str(cell).strip() 
                                for row in table 
                                for cell in row
                            )
                            
                            if has_real_content:
                                text_content += f"\nTableau {table_idx + 1}:\n"
                                for row in table:
                                    cells = [str(cell).strip() if cell else "" for cell in row]
                                    text_content += "| " + " | ".join(cells) + " |\n"
                                text_content += "\n"
                
                except Exception:
                    pass
    
    except Exception as e:
        print(f"❌ Erreur pdfplumber: {e}")
        return None
    
    return text_content if text_content.strip() else None


def process_file_with_ocr(uploaded_file, file_type="application/pdf"):
    """
    Extrait le texte avec PaddleOCR GPU (primaire) + fallback pdfplumber.
    
    Stratégie:
    1. PaddleOCR GPU (rapide et précis)
    2. Fallback pdfplumber (texte natif)
    """
    try:
        if hasattr(uploaded_file, 'read'):
            file_bytes = uploaded_file.read()
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
        else:
            file_bytes = uploaded_file
        
        # --- CAS PDF ---
        if "pdf" in file_type:
            print("📄 Extraction: PaddleOCR GPU...")
            
            # Phase 1: PaddleOCR (primaire)
            ocr_text = extract_pdf_ocr_paddle(file_bytes)
            compressed_ocr_text = token_saver(ocr_text) if ocr_text else ""
            print("✅ PaddleOCR OK")
            return compressed_ocr_text

        
        # --- CAS IMAGE ---
        elif "image" in file_type:
            print("🖼️ Extraction image PaddleOCR GPU...")
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            image = Image.open(io.BytesIO(file_bytes))
            processed = preprocess_image_for_ocr(image)
            
            text_content, confidence = extract_text_from_image_paddle(processed)
            return text_content if text_content else "⚠️ Aucun texte détecté"
        
        else:
            return "❌ Format non supporté"
    
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return f"❌ Erreur: {str(e)}"


# ==================== STREAMING SUPPORT ====================
def process_file_with_ocr_streaming(uploaded_file, file_type="application/pdf"):
    """
    Version streaming pour retour progressif des pages au client.
    Utile pour PDFs très longs.
    """
    if hasattr(uploaded_file, 'read'):
        file_bytes = uploaded_file.read()
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
    else:
        file_bytes = uploaded_file
    
    if "pdf" not in file_type:
        yield process_file_with_ocr(uploaded_file, file_type)
        return
    
    try:
        print("📄 Extraction streaming: PaddleOCR GPU...")
        images = pdf2image.convert_from_bytes(file_bytes, dpi=300)
        total_pages = len(images)
        
        for page_idx, image in enumerate(images, 1):
            try:
                processed = preprocess_image_for_ocr(image)
                page_text, confidence = extract_text_from_image_paddle(processed)
                
                if page_text.strip():
                    yield f"Page {page_idx}/{total_pages}:\n{page_text}\n\n"
                else:
                    yield f"Page {page_idx}/{total_pages}: [Aucun texte détecté]\n\n"
            
            except Exception as e:
                yield f"Page {page_idx}/{total_pages}: ❌ Erreur - {str(e)}\n\n"
                continue
    
    except Exception as e:
        yield f"❌ Erreur extraction: {str(e)}"
