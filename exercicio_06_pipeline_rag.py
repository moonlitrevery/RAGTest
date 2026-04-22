#!/usr/bin/env python3
"""
Exercício 6 — Pipeline RAG completo: ingestão TXT, chunking 500, ChromaDB,
retrieval dos 2 melhores chunks e geração com LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from exercicio_01_extracao import ler_txt
from exercicio_02_chunking import dividir_texto
from llm_client import chat_completion


def main() -> None:
    raiz = Path(__file__).resolve().parent
    caminho_txt = raiz / "data" / "regras_empresa.txt"
    if not caminho_txt.is_file():
        print(f"Arquivo não encontrado: {caminho_txt}", file=sys.stderr)
        sys.exit(1)

    # 1) Ingestão
    texto = ler_txt(caminho_txt)
    # 2) Chunking
    chunks = dividir_texto(texto, tamanho_chunk=500)
    if not chunks:
        print("Nenhum chunk gerado.")
        return

    # 3) Indexação (Chroma em memória + mesmo modelo do curso)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.Client()
    nome_colecao = "regras_empresa"
    collection = client.create_collection(name=nome_colecao, embedding_function=ef)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)

    pergunta = input("Sua pergunta sobre o regulamento: ").strip()
    if not pergunta:
        print("Nenhuma pergunta informada.")
        return

    # 4) Busca
    n = min(2, len(chunks))
    resultados = collection.query(query_texts=[pergunta], n_results=n)
    docs = resultados["documents"][0] if resultados["documents"] else []
    contexto = "\n\n".join(docs)

    # 5) Geração
    system_prompt = (
        "Você é um assistente que responde com base APENAS no contexto abaixo.\n"
        "Se a resposta não estiver no contexto, diga que as regras fornecidas não especificam isso.\n"
        "Seja objetivo e cite implicitamente as regras (não invente números ou políticas).\n\n"
        f"CONTEXTO:\n{contexto}"
    )

    try:
        resposta = chat_completion(system_prompt, pergunta, temperature=0.2)
    except Exception as e:  # noqa: BLE001
        print("Erro ao chamar o LLM (API key / Ollama / modelo).", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    print("\n--- Trechos usados (retrieval) ---")
    for i, d in enumerate(docs, start=1):
        print(f"\n[Chunk {i}]\n{d[:400]}{'...' if len(d) > 400 else ''}")

    print("\n--- Resposta ---")
    print(resposta)


if __name__ == "__main__":
    main()
