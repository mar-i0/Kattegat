#!/usr/bin/env python3.10
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def _silence_tf():
    null = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    os.dup2(null, 2)
    return old

def _restore_stderr(old):
    os.dup2(old, 2)
    os.close(old)

_saved_err = _silence_tf()

import click
import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer

_restore_stderr(_saved_err)

from config import (
    BASE_DIR, STYLE_INDEX_DIR, SYSTEM_PROMPT_FILE,
    CLOUD_API_URL, CLOUD_MODEL,
    LOCAL_API_URL, LOCAL_MODEL,
    TOP_K_STYLE, load_api_key,
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

STYLE_INDEX_FILE = STYLE_INDEX_DIR / "index.npy"
STYLE_CHUNKS_FILE = STYLE_INDEX_DIR / "chunks.json"


def load_style_library():
    if not STYLE_INDEX_FILE.exists():
        return None, None, None
    embeddings = np.load(str(STYLE_INDEX_FILE))
    with open(STYLE_CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return embeddings, data["chunks"], data["sources"]


_tokenizer = None
_model = None


def _load_embed_model():
    global _tokenizer, _model
    if _tokenizer is None:
        old = _silence_tf()
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _model.eval()
        _restore_stderr(old)
    return _tokenizer, _model


def get_local_embedding(text):
    tokenizer, model = _load_embed_model()
    inputs = tokenizer(
        [text[:8000]], padding=True, truncation=True,
        max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return emb


def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a_norm, b_norm)


def query_style(query, index_embeddings, chunks, sources, k=TOP_K_STYLE):
    query_emb = get_local_embedding(query)
    if query_emb is None:
        return []
    query_emb = np.array(query_emb).astype("float32")

    sims = np.array([cosine_similarity(query_emb, e) for e in index_embeddings])
    top_indices = np.argsort(sims)[-k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx],
            "source": sources[idx]["file"],
            "label": sources[idx]["label"],
            "score": float(sims[idx]),
        })
    return results


def extract_pdf_text(pdf_path):
    import fitz
    doc = fitz.open(str(pdf_path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def call_ollama_cloud(api_key, system_prompt, style_context, doc_summary, question):
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if style_context:
        messages.append({
            "role": "user",
            "content": f"[CONTEXTO DE ESTILO]\n{style_context}"
        })
    if doc_summary:
        messages.append({
            "role": "user",
            "content": f"[DOCUMENTO ADJUNTO]\n{doc_summary}"
        })
    messages.append({
        "role": "user",
        "content": f"[CONSULTA]\n{question}"
    })

    resp = requests.post(
        CLOUD_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": CLOUD_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 512},
        },
        timeout=120,
    )

    if resp.status_code == 200:
        data = resp.json()
        return data.get("message", {}).get("content", "")
    else:
        error_body = resp.text[:300]
        print(f"  ⚠  Cloud API error ({resp.status_code}): {error_body}", file=sys.stderr)
        return None


def call_ollama_local(system_prompt, style_context, doc_summary, question):
    parts = [system_prompt]
    if style_context:
        parts.append(f"\n[CONTEXTO DE ESTILO]\n{style_context}")
    if doc_summary:
        parts.append(f"\n[DOCUMENTO ADJUNTO]\n{doc_summary}")
    parts.append(f"\n[CONSULTA]\n{question}")
    prompt = "\n".join(parts)

    resp = requests.post(
        LOCAL_API_URL,
        json={
            "model": LOCAL_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 512},
        },
        timeout=180,
    )

    if resp.status_code == 200:
        return resp.json().get("response", "")
    else:
        print(f"  ⚠  Local API error ({resp.status_code}): {resp.text[:200]}", file=sys.stderr)
        return "Las runas están mudas. Los dioses no responden."


def render_response(response):
    if not response:
        return
    print()
    print("  ☠  Invoco a los dioses...")
    print()
    for line in response.strip().split("\n"):
        line = line.strip()
        if line:
            print(f"     {line}")
        else:
            print()
    print()


def answer(question, doc_summary, api_key, system_prompt, index_embeddings, chunks, sources):
    style_results = []
    if index_embeddings is not None:
        style_results = query_style(question, index_embeddings, chunks, sources)

    style_context = ""
    if style_results:
        style_parts = []
        for r in style_results:
            style_parts.append(f"[{r['label']}: {r['source']}]\n{r['text'][:400]}")
        style_context = "\n\n---\n\n".join(style_parts)

    response = call_ollama_cloud(api_key, system_prompt, style_context, doc_summary, question)
    if response is None:
        print(f"  ↻ Fallback a modelo local ({LOCAL_MODEL})...")
        response = call_ollama_local(system_prompt, style_context, doc_summary, question)

    render_response(response)


@click.command()
@click.argument("question", required=False, default=None)
@click.option("--file", "-f", "files", multiple=True,
              help="PDF(s) con información económica/empresarial")
def oracle(question, files):
    """Consulta al oráculo nórdico. Pulsa Ctrl+C para salir.

    QUESTION: pregunta inicial (opcional).
    """
    api_key = load_api_key()
    if not api_key:
        print("❌ No se encontró OLLAMA_API_KEY en .env ni key.txt")
        sys.exit(1)

    print("\n  🔮 El oráculo de Kattegat está despierto.")
    print("     Pulsa Ctrl+C para cerrar el velo.\n")

    index_embeddings, chunks, sources = load_style_library()
    if index_embeddings is None:
        print("  ℹ️  Biblioteca de estilo no encontrada.")
        print("     Ejecuta 'python prepare_style.py' y 'python embed.py' primero.\n")

    system_prompt = ""
    if SYSTEM_PROMPT_FILE.exists():
        system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")

    doc_summary = None
    if files:
        doc_parts = []
        for fpath in files:
            p = Path(fpath)
            if not p.exists():
                print(f"  ⚠  Archivo no encontrado: {fpath}")
                continue
            print(f"  📄 Leyendo: {p.name}")
            text = extract_pdf_text(p)
            if text:
                doc_parts.append(f"-- {p.name} --\n{text[:3000]}")
        if doc_parts:
            doc_summary = "\n\n".join(doc_parts)

    # Responde la pregunta inicial si se pasó como argumento
    if question:
        answer(question, doc_summary, api_key, system_prompt, index_embeddings, chunks, sources)

    # Bucle interactivo
    try:
        while True:
            try:
                question = input("  >> ").strip()
            except EOFError:
                break
            if not question:
                continue
            answer(question, None, api_key, system_prompt, index_embeddings, chunks, sources)
    except KeyboardInterrupt:
        print("\n\n  Las runas se cierran. Hasta la próxima visión.\n")


if __name__ == "__main__":
    oracle()
