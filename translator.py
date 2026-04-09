"""
translator.py — Japanese → Brazilian Portuguese translation
via the Ollama Python library.
"""

import logging
from ollama import chat, ChatResponse

log = logging.getLogger(__name__)

_MODEL = "gemma4:e4b"

_SYSTEM_PROMPT = """You are a professional manga localisation translator specialising in Japanese to Brazilian Portuguese.

STRICT OUTPUT RULES - never violate these:
1. Output ONLY the translated text. No preface, no labels, no quotes, no "Translation:", no explanation of any kind.
2. Never add punctuation that is not present in the original Japanese.
3. If the bubble contains a sound effect (onomatopoeia), transliterate or adapt it to a PT-BR equivalent - do not translate it literally.
4. Preserve the emotional register exactly: shouting stays shouting, whispers stay soft, questions keep their interrogative feel.
5. Use informal Brazilian Portuguese (voce, girias when appropriate) unless the source is clearly formal/keigo, in which case mirror that formality.
6. Keep honorifics (-san, -kun, -chan, -senpai, etc.) only when no natural PT-BR equivalent exists; otherwise drop or adapt them.
7. Manga bubbles are spatially constrained - prefer shorter phrasing that carries the same meaning over a longer literal rendering.
8. Never hallucinate content. If the input is empty or illegible, return an empty string and nothing else."""


def translate_text(japanese: str) -> str:
    japanese = japanese.strip()
    if not japanese:
        return ""

    response: ChatResponse = chat(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": japanese},
        ],
        options={"temperature": 0.3},
    )
    return response.message.content.strip()


def get_backend_name() -> str:
    return f"Ollama ({_MODEL})"
