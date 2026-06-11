from __future__ import annotations

import requests


class OllamaClient:
    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")

    def health(self) -> dict:
        """
        Returns:
          {
            "ok": bool,
            "models": list[str],
            "error": str
          }
        """
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            body = r.json()
            models = [m.get("name", "") for m in body.get("models", []) if m.get("name")]
            return {"ok": True, "models": models, "error": ""}
        except Exception as e:
            return {"ok": False, "models": [], "error": str(e)}

    def generate(self, prompt: str, timeout: int = 60, num_ctx: int = 2048, num_gpu: int = -1) -> str:
        options = {"num_ctx": num_ctx, "num_gpu": num_gpu, "num_thread": 4}

        # 1) Ollama native generate endpoint
        try:
            url = f"{self.host}/api/generate"
            payload = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
            res = requests.post(url, json=payload, timeout=timeout)
            res.raise_for_status()
            body = res.json()
            text = str(body.get("response", "")).strip()
            if text:
                return text
        except Exception:
            pass

        # 2) Ollama chat endpoint fallback
        try:
            url = f"{self.host}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": options,
            }
            res = requests.post(url, json=payload, timeout=timeout)
            res.raise_for_status()
            body = res.json()
            return str(body.get("message", {}).get("content", "")).strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed on both /api/generate and /api/chat: {e}")
