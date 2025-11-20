# src/models/ollama_client.py

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(model: str, prompt: str, timeout: float = 90.0) -> str:
    """
    Sends a prompt to an Ollama model and returns the generated text.
    Includes:
      - timeout protection
      - response error catching
      - model memory errors
      - missing 'response' field fallback
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise RuntimeError("❌ Ollama is not running. Start Ollama before calling the API.")
    except requests.exceptions.Timeout:
        raise RuntimeError("❌ Ollama request timed out (model too slow or CPU overloaded).")

    # HTTP-level error
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"❌ Ollama HTTP Error: {e}")

    data = resp.json()

    # Model-level errors
    if "error" in data:
        raise RuntimeError(f"❌ Ollama Model Error: {data['error']}")

    # Normal output
    if "response" in data:
        return data["response"]

    # Unexpected format
    raise RuntimeError(f"❌ Unexpected Ollama response: {data}")



if __name__ == "__main__":
    print(ollama_generate("phi:latest", "Hello!"))