"""Optional Ollama-backed tutor explanations.

The core music system does not require Ollama. This module calls a local Ollama
server when available and falls back cleanly when it is not running.
"""

from __future__ import annotations

from urllib import error, request
import json


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"


def ollama_chat(
    lesson: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 20.0,
) -> str:
    prompt = (
        "You are a concise Indian classical music tutor. Rewrite the structured "
        "lesson below as friendly teaching guidance for a beginner. Do not invent "
        "new swaras. Keep it practical and under 180 words.\n\n"
        f"{lesson}"
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You teach sargam practice clearly and carefully."},
            {"role": "user", "content": prompt},
        ],
    }
    raw = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama is not reachable at {base_url}") from exc

    message = data.get("message", {})
    content = str(message.get("content", "")).strip()
    if not content:
        raise RuntimeError("Ollama returned an empty tutor response")
    return content
