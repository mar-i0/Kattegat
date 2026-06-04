#!/usr/bin/env python3.10
"""Genera embeddings desde chunks.json → index.npy usando transformers."""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).parent
STYLE_INDEX_DIR = BASE_DIR / "style_index"
CHUNKS_FILE = STYLE_INDEX_DIR / "chunks.json"
INDEX_FILE = STYLE_INDEX_DIR / "index.npy"

BATCH_SIZE = 200
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not CHUNKS_FILE.exists():
        print("❌ Ejecuta prepare_style.py primero")
        sys.exit(1)

    print("📖 Cargando fragmentos...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]
    print(f"   → {len(chunks)} fragmentos")

    print(f"\n⚡ Cargando modelo {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    print("   Modelo listo")

    print("\n⚡ Generando embeddings...")
    all_embs = []
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).numpy()
        all_embs.append(emb)
        del inputs, outputs, emb
        gc.collect()
        done = min(start + BATCH_SIZE, total)
        print(f"   [{done}/{total}] ({done/total*100:.0f}%)")

    embeddings = np.concatenate(all_embs, axis=0).astype("float32")
    np.save(str(INDEX_FILE), embeddings)

    mb = embeddings.nbytes / 1024 / 1024
    print(f"\n✅ Índice guardado: {len(embeddings)} vectores ({mb:.1f} MB)")
    print(f"   → {INDEX_FILE}")
    print('\n✨ Ejecuta: python oracle.py "tu pregunta"')


if __name__ == "__main__":
    main()
