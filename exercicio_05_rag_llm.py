#!/usr/bin/env python3
"""
Exercício 5 — Retrieval (como no Ex. 4) + Generation via API LLM.
Configure uma das variáveis: GROQ_API_KEY, OPENAI_API_KEY, ou use Ollama local (OLLAMA_BASE_URL).
"""

from __future__ import annotations

import sys

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from llm_client import chat_completion


def fato_mais_semelhante(pergunta: str, fatos: list[str], modelo: SentenceTransformer) -> str:
    vetores_fatos = modelo.encode(fatos, convert_to_numpy=True)
    v_pergunta = modelo.encode([pergunta], convert_to_numpy=True)[0]
    melhor_idx = 0
    melhor_dist = float("inf")
    for i, v_fato in enumerate(vetores_fatos):
        dist = cosine(v_pergunta, v_fato)
        if dist < melhor_dist:
            melhor_dist = dist
            melhor_idx = i
    return fatos[melhor_idx]


def main() -> None:
    fatos = [
        "O pão de queijo é originário de Minas Gerais.",
        "A Terra gira em torno do Sol.",
        "A capital da França é Paris.",
        "Python é uma linguagem de programação popular em ciência de dados.",
        "O Rio Amazonas é o maior rio em volume de água do mundo.",
    ]

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    pergunta = input("Digite sua pergunta: ").strip()
    if not pergunta:
        print("Nenhuma pergunta informada.")
        return

    contexto = fato_mais_semelhante(pergunta, fatos, modelo)
    print(f"\n[Fato mais relevante recuperado]\n{contexto}\n")

    system_prompt = (
        "Você é um assistente prestativo.\n"
        "Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE neste contexto:\n"
        f"CONTEXTO: {contexto}"
    )

    try:
        resposta = chat_completion(system_prompt, pergunta)
    except Exception as e:  # noqa: BLE001 — exemplo didático
        print("Erro ao chamar o LLM. Verifique API keys ou Ollama.", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    print("--- Resposta do modelo ---")
    print(resposta)


if __name__ == "__main__":
    main()
