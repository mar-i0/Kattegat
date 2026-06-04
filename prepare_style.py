#!/usr/bin/env python3.10
import json
import warnings
from pathlib import Path

from config import (
    BASE_DIR, MEDIUMS_DIR, INVERSION_DIR, STYLE_INDEX_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

warnings.filterwarnings("ignore")

STYLE_INDEX_FILE = STYLE_INDEX_DIR / "index.npy"
STYLE_CHUNKS_FILE = STYLE_INDEX_DIR / "chunks.json"


def extract_text_from_pdf(path):
    import fitz
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_text_from_epub(path):
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(path)
    texts = []
    for item in book.get_items():
        if item.get_type() == 9:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text(separator="\n"))
    return "\n".join(texts)


def extract_text_from_txt(path):
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text(path):
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            return extract_text_from_pdf(path)
        elif ext == ".epub":
            return extract_text_from_epub(path)
        elif ext == ".txt":
            return extract_text_from_txt(path)
        else:
            return None
    except Exception as e:
        print(f"  ⚠  Error en {path.name}: {e}")
        return None


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def scan_files(directory, label):
    if not directory.exists():
        print(f"  ! {label}: directorio no encontrado")
        return []
    files = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in {".pdf", ".epub", ".txt"}:
            files.append((f, label))
        elif f.is_file():
            print(f"  ~ {label}: formato omitido: {f.name}")
    return files


def deduplicate_files(files):
    seen = {}
    deduped = []
    for fpath, label in files:
        key = fpath.stem.lower().replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u").replace("ñ","n").replace("Ã¡","a").replace("Â ","").replace("Ã","a").replace("","").replace("","").replace("â","a")
        key = key[:40]
        existing = seen.get(key)
        if existing is None:
            seen[key] = (fpath, label)
            deduped.append((fpath, label))
        else:
            # Preferir PDF sobre EPUB para mejor extracción
            if fpath.suffix.lower() == ".pdf" and existing[0].suffix.lower() != ".pdf":
                deduped.remove(existing)
                seen[key] = (fpath, label)
                deduped.append((fpath, label))
                print(f"  ~ reemplazado por: {fpath.name}")
            else:
                print(f"  ~ duplicado: {fpath.name}")
    return deduped


def main():
    STYLE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Preparando la biblioteca de estilo del oráculo")
    print("=" * 50)

    # --- Fase 1: cargar o extraer chunks ---
    if STYLE_CHUNKS_FILE.exists():
        print("\n📦 Cargando fragmentos existentes...")
        with open(STYLE_CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_chunks = data["chunks"]
        chunk_sources = data["sources"]
        print(f"   → {len(all_chunks)} fragmentos de {STYLE_CHUNKS_FILE.name}")
    else:
        files = []
        files.extend(scan_files(MEDIUMS_DIR, "Mediums"))
        files.extend(scan_files(INVERSION_DIR, "Inversión"))
        files = deduplicate_files(files)
        print(f"\nArchivos a procesar (tras dedup): {len(files)}")

        all_chunks = []
        chunk_sources = []

        for fpath, label in files:
            print(f"\n  📖 {label}: {fpath.name}")
            text = extract_text(fpath)
            if not text or len(text.strip()) < 100:
                print(f"     → texto insuficiente, saltando")
                continue
            chunks = chunk_text(text)
            print(f"     → {len(chunks)} fragmentos")
            all_chunks.extend(chunks)
            chunk_sources.extend([{"file": fpath.name, "label": label}] * len(chunks))

        if not all_chunks:
            print("\n❌ No se encontró texto para indexar.")
            return

        print(f"\n📦 Total fragmentos: {len(all_chunks)}")
        with open(STYLE_CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump({"chunks": all_chunks, "sources": chunk_sources}, f, ensure_ascii=False)
        print(f"  → Fragmentos guardados en {STYLE_CHUNKS_FILE.name}")

    print("\n✨ Fragmentación completada.")
    print("   Ejecuta: python embed.py para generar los embeddings")
    print("   Luego:   python oracle.py \"tu pregunta\"")


if __name__ == "__main__":
    main()
