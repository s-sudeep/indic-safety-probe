"""Core abstractions for indic-safety-probe."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import statistics


@dataclass
class Prompt:
    """A single prompt sent to the model."""
    text: str
    language: str            # ISO code: "en", "hi", "kn", "ta", "te", "bn"
    category: str            # e.g., "harmful", "benign", "control"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeScore:
    """Score for one prompt-response pair. Higher value = worse failure."""
    value: float             # 0.0 = ideal, 1.0 = worst
    label: str               # "refused", "complied", "partial", "error", etc.
    explanation: str = ""
    raw_response: str = ""


@dataclass
class ProbeResult:
    """Aggregated output of one probe."""
    probe_name: str
    scores: list[ProbeScore]
    prompts: list[Prompt]
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        valid = [s.value for s in self.scores if s.label != "error"]
        return statistics.mean(valid) if valid else 0.0

    @property
    def failure_rate(self) -> float:
        valid = [s for s in self.scores if s.label != "error"]
        if not valid:
            return 0.0
        return sum(1 for s in valid if s.value > 0.5) / len(valid)

    def by_language(self) -> dict[str, list[ProbeScore]]:
        out: dict[str, list[ProbeScore]] = {}
        for prompt, score in zip(self.prompts, self.scores):
            out.setdefault(prompt.language, []).append(score)
        return out


class ModelAdapter(ABC):
    """Interface for any model the probes test."""
    name: str

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Send prompt, return response text."""
        ...


class Probe(ABC):
    """Base class for all safety probes."""
    name: str
    description: str
    languages: list[str]

    @abstractmethod
    def generate_prompts(self) -> list[Prompt]:
        ...

    @abstractmethod
    def score(self, prompt: Prompt, response: str) -> ProbeScore:
        ...

    def run(self, model: ModelAdapter, verbose: bool = False) -> ProbeResult:
        prompts = self.generate_prompts()
        scores: list[ProbeScore] = []
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"  [{self.name}] {i+1}/{len(prompts)} ({prompt.language})")
            try:
                response = model.generate(prompt.text)
                score = self.score(prompt, response)
                score.raw_response = response
            except Exception as e:
                score = ProbeScore(
                    value=float("nan"),
                    label="error",
                    explanation=f"{type(e).__name__}: {e}",
                )
            scores.append(score)
        return ProbeResult(probe_name=self.name, scores=scores, prompts=prompts)


@dataclass
class SuiteResult:
    model_name: str
    probe_results: list[ProbeResult]
    timestamp: datetime = field(default_factory=datetime.now)


class Suite:
    """Runs a collection of probes against a model."""
    def __init__(self, probes: list[Probe]):
        self.probes = probes

    def run(self, model: ModelAdapter, verbose: bool = True) -> SuiteResult:
        results = []
        for probe in self.probes:
            if verbose:
                print(f"Running probe: {probe.name}")
            results.append(probe.run(model, verbose=verbose))
        return SuiteResult(model_name=model.name, probe_results=results)