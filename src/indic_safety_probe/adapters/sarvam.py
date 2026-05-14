"""Sarvam Cloud API adapter. Wraps the OpenAI-compatible endpoint."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from ..core import ModelAdapter


@dataclass
class SarvamCloudAdapter(ModelAdapter):
    """Adapter for Sarvam Cloud models via their OpenAI-compatible endpoint.

    Parameters
    ----------
    model : str
        Sarvam model identifier, e.g. "sarvam-m", "sarvam-30b", "sarvam-105b".
    system_prompt : str, optional
        System prompt that frames the agent context. Used for the
        "you are a customer service agent for Indian Railways" demos.
    api_key : str, optional
        Defaults to SARVAM_API_KEY env var.
    base_url : str
        Sarvam's OpenAI-compatible endpoint.
    temperature : float
        Default 0.0 for reproducibility of safety evals.
    max_tokens : int
        Cap on response length. BEC outputs that exceed this are still
        scoreable; we just need enough to see refusal vs compliance.
    max_retries : int
        Retries on transient errors (rate limit, timeout).
    request_timeout : float
        Per-call timeout in seconds.
    """

    model: str
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    base_url: str = "https://api.sarvam.ai/v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    max_retries: int = 3
    request_timeout: float = 60.0

    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("SARVAM_API_KEY")
        if not key:
            raise ValueError(
                "Sarvam API key not provided. Pass api_key=... or set SARVAM_API_KEY."
            )
        self._client = OpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=self.request_timeout,
        )

    @property
    def name(self) -> str:
        return f"sarvam:{self.model}"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Single-turn completion. Returns the assistant text or raises after retries."""
        effective_system = system_prompt if system_prompt is not None else self.system_prompt

        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError) as e:
                last_err = e
                # exponential backoff: 2s, 4s, 8s
                time.sleep(2 ** (attempt + 1))
            except APIError as e:
                # non-transient API error — surface immediately
                raise
        raise RuntimeError(
            f"Sarvam API failed after {self.max_retries} retries: {last_err}"
        )