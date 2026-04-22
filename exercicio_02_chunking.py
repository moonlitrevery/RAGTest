#!/usr/bin/env python3
"""
Exercício 2 — Chunking: divide o texto do Exercício 1 em blocos de tamanho fixo.
"""

from __future__ import annotations

from pathlib import Path

from exercicio_01_extracao import ler_pdf, ler_txt


def dividir_texto(texto: str, tamanho_chunk: int = 500) -> list[str]:
    """Divide o texto em blocos consecutivos de no máximo `tamanho_chunk` caracteres."""
    if not texto:
        return []
    texto = texto.strip()
    if not texto:
        return []
    return [texto[i : i + tamanho_chunk] for i in range(0, len(texto), tamanho_chunk)]


def main() -> None:
    caminho = Path(__file__).resolve().parent / "data" / "historia.txt"
    if caminho.suffix.lower() == ".txt":
        texto = ler_txt(caminho)
    else:
        texto = ler_pdf(caminho)

    tamanho_chunk = 500
    chunks = dividir_texto(texto, tamanho_chunk=tamanho_chunk)
    print(f"Quantidade de chunks: {len(chunks)}")
    print(f"Tamanho máximo por chunk: {tamanho_chunk} caracteres\n")
    if chunks:
        print("--- Primeiro chunk ---")
        print(chunks[0])
        print("\n--- Último chunk ---")
        print(chunks[-1])


if __name__ == "__main__":
    main()
