"""
Minimal LLM client adapters and factory.

Uniform interface:
  client = get_llm_client(provider_name)
  text = client.chat_completion(messages, model=..., max_tokens=..., temperature=...)
"""
from typing import List, Dict, Optional
import os
import requests


class BaseLLMClient:
    def chat_completion(self, messages: List[Dict], model: str, max_tokens: int = 3000, temperature: float = 0) -> str:
        raise NotImplementedError()


class OpenAIAdapter(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("OpenAI SDK not available (install `openai`).") from e
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs) if kwargs else OpenAI()

    def chat_completion(self, messages, model: str, max_tokens: int = 3000, temperature: float = 0) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            return resp.choices[0].message.content.strip()
        except Exception:
            return getattr(resp, "text", str(resp))


class TogetherAIAdapter(BaseLLMClient):
    """Together.ai via its OpenAI-compatible endpoint."""

    def __init__(self, api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("OpenAI SDK not available (install `openai`).") from e
        self.client = OpenAI(
            api_key=api_key or os.getenv("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
        )

    def chat_completion(self, messages, model: str, max_tokens: int = 3000, temperature: float = 0) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


class AnthropicAdapter(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")

    def chat_completion(self, messages, model: str, max_tokens: int = 3000, temperature: float = 0) -> str:
        system_text = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        user_messages = [m for m in messages if m.get("role") != "system"]
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_text,
            "messages": user_messages,
        }
        resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()


class GeminiAdapter(BaseLLMClient):
    def __init__(self):
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("google.generativeai (Gemini) SDK not available.") from e
        self.client = genai

    def chat_completion(self, messages, model: str, max_tokens: int = 3000, temperature: float = 0) -> str:
        contents = [{"type": "text", "text": m.get("content", "")} for m in messages]
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return getattr(response, "text", str(response)).strip()


class OllamaAdapter(BaseLLMClient):
    """Ollama via its OpenAI-compatible endpoint (default: http://localhost:11434).

    To use a remote server, either:
      - Set OLLAMA_BASE_URL=http://<host>:11434  (if Ollama is exposed externally)
      - Or set up an SSH tunnel first:
          ssh -L 11434:localhost:11434 user@server
        then use OLLAMA_BASE_URL=http://localhost:11434
    """

    def __init__(self, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("OpenAI SDK not available (install `openai`).") from e
        resolved_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = OpenAI(
            base_url=f"{resolved_url.rstrip('/')}/v1",
            api_key="ollama",          # Ollama ignores this but SDK requires a value
        )

    def chat_completion(self, messages, model: str, max_tokens: int = 3000, temperature: float = 0) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()


class DemoAdapter(BaseLLMClient):
    def chat_completion(self, messages, model: str, max_tokens: int = 3000, temperature: float = 0) -> str:
        joined = "\n".join(m.get("content", "") for m in messages[-2:])
        return f"[demo response] {joined}"[: max(0, max_tokens)]


def get_llm_client(provider: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """Factory. Reads LLM_PROVIDER env var if provider is None (defaults to 'openai')."""
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai")
    provider = provider.lower()

    if provider in ("openai", "openai.com"):
        return OpenAIAdapter(
            api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY"),
            base_url=kwargs.get("base_url"),
        )
    if provider in ("together", "togetherai", "together.ai"):
        return TogetherAIAdapter(api_key=kwargs.get("api_key") or os.getenv("TOGETHER_API_KEY"))
    if provider in ("anthropic", "claude"):
        return AnthropicAdapter(api_key=kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY"))
    if provider in ("gemini", "google"):
        return GeminiAdapter()
    if provider in ("ollama",):
        return OllamaAdapter(base_url=kwargs.get("base_url") or os.getenv("OLLAMA_BASE_URL"))
    if provider in ("openrouter", "openrouter.ai"):
        # OpenRouter is OpenAI-compatible and reuses OpenAIAdapter with its base_url.
        return OpenAIAdapter(
            api_key=kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
    if provider == "demo":
        return DemoAdapter()

    raise RuntimeError(
        f"LLM provider '{provider}' not supported. "
        "Set LLM_PROVIDER to one of: openai, together, anthropic, gemini, ollama, openrouter, demo."
    )
