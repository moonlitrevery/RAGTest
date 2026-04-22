"""Cliente LLM compatível com OpenAI API (OpenAI, Groq, ou servidor local Ollama)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def create_llm_client() -> tuple[OpenAI, str]:
    """
    Retorna (cliente, model_id).
    Prioridade: GROQ_API_KEY > OPENAI_API_KEY > OLLAMA (URL local).
    """
    groq = os.getenv("GROQ_API_KEY")
    if groq:
        client = OpenAI(api_key=groq, base_url="https://api.groq.com/openai/v1")
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return client, model

    oai = os.getenv("OPENAI_API_KEY")
    if oai:
        client = OpenAI(api_key=oai)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return client, model

    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    client = OpenAI(api_key="ollama", base_url=base)
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    return client, model


def chat_completion(system_prompt: str, user_message: str, *, temperature: float = 0.3) -> str:
    client, model = create_llm_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    choice = resp.choices[0]
    if choice.message.content is None:
        return ""
    return choice.message.content.strip()
