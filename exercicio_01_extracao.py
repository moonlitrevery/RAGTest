#!/usr/bin/env python3
"""
Exercício 1 — Extração de texto de um documento (TXT ou PDF).
Uso:
  python exercicio_01_extracao.py data/historia.txt
  python exercicio_01_extracao.py caminho/para/artigo.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ler_txt(caminho: Path) -> str:
    return caminho.read_text(encoding="utf-8")


def ler_pdf(caminho: Path) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(caminho)
    partes: list[str] = []
    for pagina in doc:
        partes.append(pagina.get_text())
    doc.close()
    return "\n".join(partes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lê e imprime o texto de um .txt ou .pdf")
    parser.add_argument(
        "arquivo",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "data" / "historia.txt",
        help="Caminho do arquivo (padrão: data/historia.txt)",
    )
    args = parser.parse_args()
    caminho: Path = args.arquivo
    if not caminho.is_file():
        print(f"Arquivo não encontrado: {caminho}", file=sys.stderr)
        sys.exit(1)

    sufixo = caminho.suffix.lower()
    if sufixo == ".txt":
        texto = ler_txt(caminho)
    elif sufixo == ".pdf":
        texto = ler_pdf(caminho)
    else:
        print("Use um arquivo .txt ou .pdf.", file=sys.stderr)
        sys.exit(1)

    print(texto)


if __name__ == "__main__":
    main()
