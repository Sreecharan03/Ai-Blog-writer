"""
app/services/vertex_gemini_client.py

Gemini API client via google-generativeai.

Requires env (recommended):
- GEMINI_API_KEY (or GOOGLE_API_KEY)

Note: This uses the public Gemini API, not Vertex AI.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import google.generativeai as genai
import requests


class VertexGeminiError(RuntimeError):
    pass


class VertexGeminiPermissionError(VertexGeminiError):
    pass


class VertexGeminiRetryableError(VertexGeminiError):
    pass


@dataclass
class VertexUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _gemini_api_key() -> Optional[str]:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GENAI_API_KEY")
    )


def _classify_error(err: Exception) -> VertexGeminiError:
    msg = str(err)
    low = msg.lower()
    if "permission" in low or "unauthorized" in low or "api key" in low or "403" in low:
        return VertexGeminiPermissionError(msg)
    if "429" in low or "rate" in low or "timeout" in low or "temporarily" in low:
        return VertexGeminiRetryableError(msg)
    return VertexGeminiError(msg)


GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _groq_api_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY")


class VertexGeminiClient:
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        timeout_s: int = 60,
    ) -> None:
        # project_id/location kept for backward compatibility (unused for Gemini API)
        _ = project_id, location

        self._api_key = api_key or _gemini_api_key()
        self._groq_key = groq_api_key or _groq_api_key()
        self._groq_model = groq_model or GROQ_DEFAULT_MODEL
        if not self._api_key and not self._groq_key:
            raise VertexGeminiError("Missing GEMINI_API_KEY / GOOGLE_API_KEY and GROQ_API_KEY")

        if self._api_key:
            genai.configure(api_key=self._api_key)
        self._timeout_s = timeout_s
        self._model_cache: Dict[str, Any] = {}

    def _get_model(self, model: str):
        m = self._model_cache.get(model)
        if m is None:
            m = genai.GenerativeModel(model)
            self._model_cache[model] = m
        return m

    def _usage_from_metadata(self, um: Any) -> VertexUsage:
        def _get(obj: Any, attr: str, key: str) -> int:
            if obj is None:
                return 0
            if hasattr(obj, attr):
                return int(getattr(obj, attr) or 0)
            if isinstance(obj, dict):
                return int(obj.get(key) or 0)
            return 0

        return VertexUsage(
            prompt_tokens=_get(um, "prompt_token_count", "prompt_token_count"),
            output_tokens=_get(um, "candidates_token_count", "candidates_token_count"),
            total_tokens=_get(um, "total_token_count", "total_token_count"),
        )

    def generate_content(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        response_mime_type: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> Tuple[str, VertexUsage, Dict[str, Any]]:
        # Prefer Gemini if available; otherwise use Groq
        if self._api_key:
            prompt_text = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            gen_cfg = genai.types.GenerationConfig(
                temperature=float(temperature),
                max_output_tokens=int(max_output_tokens),
            )
            if response_mime_type:
                gen_cfg.response_mime_type = response_mime_type

            try:
                resp = self._get_model(model).generate_content(
                    prompt_text,
                    generation_config=gen_cfg,
                    request_options={"timeout": self._timeout_s},
                )
            except Exception as e:
                # fall back to Groq if configured
                if self._groq_key:
                    return self._groq_generate_content(
                        prompt=prompt,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        response_mime_type=response_mime_type,
                    )
                raise _classify_error(e)

            text = (getattr(resp, "text", "") or "").strip()
            usage = self._usage_from_metadata(getattr(resp, "usage_metadata", None))

            raw: Dict[str, Any] = {}
            to_dict = getattr(resp, "to_dict", None)
            if callable(to_dict):
                try:
                    raw = to_dict()
                except Exception:
                    raw = {}

            return text, usage, raw

        # Gemini unavailable, use Groq
        return self._groq_generate_content(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
        )

    def generate_json(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        system_instruction: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], VertexUsage, str]:
        raw_text, usage, _raw = self.generate_content(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            system_instruction=system_instruction,
        )

        try:
            obj = json.loads(raw_text)
        except Exception as e:
            raise VertexGeminiError(f"Model returned non-JSON. Parse error={e}. Raw={raw_text[:500]}")

        return obj, usage, raw_text

    def _groq_generate_content(
        self,
        *,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_output_tokens: int,
        response_mime_type: Optional[str],
    ) -> Tuple[str, VertexUsage, Dict[str, Any]]:
        if not self._groq_key:
            raise VertexGeminiError("Missing GROQ_API_KEY")

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": self._groq_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_output_tokens),
        }
        if response_mime_type == "application/json":
            body["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self._groq_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=self._timeout_s)
        except Exception as e:
            raise _classify_error(e)

        if resp.status_code != 200:
            raise _classify_error(RuntimeError(f"Groq {resp.status_code}: {resp.text[:500]}"))

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception:
            text = ""

        um = data.get("usage") or {}
        usage = VertexUsage(
            prompt_tokens=int(um.get("prompt_tokens") or 0),
            output_tokens=int(um.get("completion_tokens") or 0),
            total_tokens=int(um.get("total_tokens") or 0),
        )
        return text, usage, data

    def generate_with_fallback(
        self,
        primary_model: str,
        fallback_model: str,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        json_mode: bool = False,
        system_instruction: Optional[str] = None,
        max_retries: int = 4,
    ) -> Tuple[Any, VertexUsage, str, str]:
        models = [primary_model, fallback_model]

        last_err: Optional[Exception] = None
        for model in models:
            for attempt in range(max_retries + 1):
                try:
                    if json_mode:
                        obj, usage, raw_text = self.generate_json(
                            model=model,
                            prompt=prompt,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            system_instruction=system_instruction,
                        )
                        return obj, usage, raw_text, model
                    txt, usage, _raw = self.generate_content(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        system_instruction=system_instruction,
                    )
                    return txt, usage, txt, model

                except VertexGeminiPermissionError:
                    raise
                except VertexGeminiRetryableError as e:
                    last_err = e
                    sleep_s = min(8.0, (2 ** attempt) + random.random())
                    time.sleep(sleep_s)
                    continue
                except VertexGeminiError as e:
                    last_err = e
                    break

        raise VertexGeminiError(f"All models failed. Last error: {last_err}")
