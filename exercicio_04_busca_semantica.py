#!/usr/bin/env python3
"""
Exercício 4 — Banco de conhecimento em memória: similaridade por cosseno.
"""

from __future__ import annotations

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


def main() -> None:
    fatos = [
        "O pão de queijo é originário de Minas Gerais.",
        "A Terra gira em torno do Sol.",
        "A capital da França é Paris.",
        "Python é uma linguagem de programação popular em ciência de dados.",
        "O Rio Amazonas é o maior rio em volume de água do mundo.",
    ]

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    vetores_fatos = modelo.encode(fatos, convert_to_numpy=True)

    pergunta = input("Digite sua pergunta: ").strip()
    if not pergunta:
        print("Nenhuma pergunta informada.")
        return

    v_pergunta = modelo.encode([pergunta], convert_to_numpy=True)[0]

    melhor_idx = 0
    melhor_dist = float("inf")
    for i, v_fato in enumerate(vetores_fatos):
        dist = cosine(v_pergunta, v_fato)
        if dist < melhor_dist:
            melhor_dist = dist
            melhor_idx = i

    print("\nFato mais semelhante à pergunta:")
    print(fatos[melhor_idx])
    print(f"(distância cosseno menor = mais similar; valor: {melhor_dist:.4f})")


if __name__ == "__main__":
    main()
