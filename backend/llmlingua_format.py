from llmlingua import PromptCompressor
import re

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True
)


def clean_ocr_text(text):
    """Remet en forme un texte brut issu de PaddleOCR."""

    # Ajoute un saut de ligne avant les majuscules qui suivent un point
    text = re.sub(r'\. ([A-Z])', r'.\n\n\1', text)

    # Ajoute un saut de ligne après les signes de ponctuation forts
    text = re.sub(r'([.!?;]) ', r'\1\n', text)

    # Supprime les espaces multiples
    text = re.sub(r' +', ' ', text)

    # Supprime les lignes vides multiples
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def compress_long_text(text, rate=0.5, chunk_size=400):
    """Découpe le texte en chunks et compresse chaque chunk."""

    # Découpe en paragraphes d'abord
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Regroupe les paragraphes en chunks de ~chunk_size mots
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        word_count = len(para.split())
        if current_size + word_count > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = word_count
        else:
            current_chunk.append(para)
            current_size += word_count

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    # Compresse chaque chunk
    compressed_parts = []
    total_origin = 0
    total_compressed = 0

    for i, chunk in enumerate(chunks):
        result = compressor.compress_prompt(
            chunk,
            rate=rate,
            force_tokens=['\n', '?', '.', ',']
        )
        compressed_parts.append(result['compressed_prompt'])
        total_origin += result['origin_tokens']
        total_compressed += result['compressed_tokens']
        print(f"Chunk {i+1}/{len(chunks)} compressé")

    # ✅ Retourne un dictionnaire avec le texte compressé en string
    return {
        'compressed_prompt': '\n\n'.join(compressed_parts),
        'origin_tokens': total_origin,
        'compressed_tokens': total_compressed,
        'ratio': f"{total_origin / total_compressed:.1f}x"
    }


def process_for_llm(text, chunk_tokens=10000):
    """Découpe un texte compressé en chunks pour le LLM."""

    words = text.split()
    # ~0.75 mots par token en moyenne
    chunk_size_words = int(chunk_tokens * 0.75)

    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunk = ' '.join(words[i:i + chunk_size_words])
        chunks.append(chunk)

    print(f"Nombre de chunks : {len(chunks)}")
    return chunks



def token_saver(raw_text):
    cleaned_text = clean_ocr_text(raw_text)
    result = compress_long_text(cleaned_text, rate=0.7)

    print(f"Tokens originaux : {result['origin_tokens']} → compressés : {result['compressed_tokens']} (ratio {result['ratio']})")

    # Redécouper le texte compressé en chunks pour le LLM
    #chunks = process_for_llm(result['compressed_prompt'], chunk_tokens=10000)

    return result