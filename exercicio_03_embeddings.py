#!/usr/bin/env python3
"""
Exercício 3 — Embeddings locais com sentence-transformers (all-MiniLM-L6-v2).
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


def main() -> None:
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    frases = [
        "O gato dorme no sofá.",
        "A programação em Python é versátil.",
        "Embeddings convertem texto em vetores densos.",
    ]
    vetores = modelo.encode(frases, convert_to_numpy=True)
    primeiro = vetores[0]
    print(f"Dimensão do vetor da primeira frase: {len(primeiro)}")
    print("Vetor da primeira frase (amostra dos primeiros 12 valores):")
    print(primeiro[:12].tolist())
    print("\nVetor completo da primeira frase:")
    print(primeiro.tolist())


if __name__ == "__main__":
    main()
