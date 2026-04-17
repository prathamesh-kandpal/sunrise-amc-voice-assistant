from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


def _post_json(base_url: str, path: str, payload: dict) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {base_url}. Start the local server with `ollama serve`."
        ) from exc


def generate_answer(base_url: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    response = _post_json(
        base_url=base_url,
        path="/api/generate",
        payload={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    if "response" not in response:
        raise OllamaError(f"Unexpected Ollama response: {response}")
    return str(response["response"]).strip()
