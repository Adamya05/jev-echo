"""The decision seam.

Everything in these demos talks to a `Backend` through one method:

    backend.decide(state, questions) -> dict[str, Answer]

Neither backend's SDK types leak past this file. That is the whole point:
swapping hosted Jev for a local MLX model is a new Backend class, not a
rewrite of the demos.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from dotenv import load_dotenv

load_dotenv()

# Jev pricing: $0.042 per 1M input tokens, output free.
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000


# --------------------------------------------------------------------------
# Neutral question types (no SDK imports)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Choice:
    """Pick one of N named options. N can be up to 255."""
    instructions: str
    criteria: Mapping[str, str]


@dataclass(frozen=True)
class Noul:
    """Probability that a proposition is true."""
    instructions: str


@dataclass(frozen=True)
class Score:
    """Position on an ordered rubric, 2-10 levels."""
    instructions: str
    criteria: Sequence[str]


Question = Choice | Noul | Score


# --------------------------------------------------------------------------
# Neutral answers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Answer:
    """One answer. `value` is the option name, the score, or P(true)."""
    kind: str
    value: Any
    confidence: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)

    @property
    def p(self) -> float:
        """Probability mass on the returned value."""
        if self.kind == "noul":
            return float(self.value)
        return float(self.probabilities.get(str(self.value), 0.0))


@dataclass(frozen=True)
class Decision:
    answers: Mapping[str, Answer]
    latency_ms: float
    input_tokens: int
    model: str

    billed: bool = True

    @property
    def usd(self) -> float:
        """Local backends are free; only hosted calls cost anything."""
        return self.input_tokens * USD_PER_INPUT_TOKEN if self.billed else 0.0

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]


class Backend(Protocol):
    name: str

    def decide(self, state: Any, questions: Mapping[str, Question]) -> Decision: ...


# --------------------------------------------------------------------------
# Hosted Jev
# --------------------------------------------------------------------------

class JevBackend:
    """Hosted Jev, from TypeSafe directly or through OpenRouter.

    OpenRouter serves Jev on its own decisions endpoint rather than in the
    chat-completions model list -- it will not show up in /api/v1/models.
    The SDK appends /v1/systemone to whatever base URL it is given, so
    pointing it at https://openrouter.ai/api is the whole integration.
    """

    def __init__(self, model: str | None = None, timeout: float = 10.0,
                 via: str | None = None) -> None:
        import typesafe_sdk as ts

        via = via or os.environ.get("JEV_VIA", "typesafe")
        if via == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY not set. Add it to .env:\n"
                    "  OPENROUTER_API_KEY=sk-or-v1-...")
            model = model or "typesafe/jev-1.13"
            self._client = ts.TypeSafeClient(
                api_key=key, model=model, timeout=timeout,
                base_url="https://openrouter.ai/api")
            self.name = f"jev-or:{model}"
        else:
            if not os.environ.get("TYPESAFE_API_KEY"):
                raise RuntimeError("TYPESAFE_API_KEY not set (looked in env and .env)")
            self._client = ts.TypeSafeClient(model=model, timeout=timeout)
            self.name = f"jev:{model or 'latest'}"
        self._ts = ts

    def _encode(self, q: Question):
        ts = self._ts
        if isinstance(q, Choice):
            return ts.Choice(instructions=q.instructions, criteria=dict(q.criteria))
        if isinstance(q, Noul):
            return ts.Noul(instructions=q.instructions)
        if isinstance(q, Score):
            return ts.Score(instructions=q.instructions, criteria=list(q.criteria))
        raise TypeError(f"unknown question type: {type(q)}")

    def decide(self, state: Any, questions: Mapping[str, Question]) -> Decision:
        payload = {k: self._encode(v) for k, v in questions.items()}
        t0 = time.perf_counter()
        resp = self._client.system_one(state=state, questions=payload)
        latency_ms = (time.perf_counter() - t0) * 1000

        answers: dict[str, Answer] = {}
        for key, a in resp.answers.items():
            if a.type == "choice":
                answers[key] = Answer("choice", a.choice, a.confidence, dict(a.probabilities))
            elif a.type == "noul":
                answers[key] = Answer("noul", a.noul, None, {})
            else:
                answers[key] = Answer("score", a.score, a.confidence, dict(a.probabilities))

        return Decision(
            answers=answers,
            latency_ms=latency_ms,
            input_tokens=resp.usage.input_tokens,
            model=resp.model,
        )


# --------------------------------------------------------------------------
# Local MLX (step 2 -- stubbed until laya-mlx is installed)
# --------------------------------------------------------------------------

class LayaBackend:
    """Local System One on Apple Silicon via laya-mlx.

    Same three question types and the same state -> probabilities contract
    as Jev, so it drops onto this seam unchanged. Two things differ and
    both matter for a fair comparison:

    * 512-token context on the English checkpoint (1024 multilingual),
      against Jev's 32K. Snake's state is ~360 tokens, so it fits -- but
      only because the demo was sized for this from the start.
    * The published checkpoint ships temperatures outside the calibrated
      range and the library warns that affected confidences are
      uncalibrated. Jev's whole pitch is calibration, so read any
      confidence comparison with that in mind.
    """

    def __init__(self, checkpoint: str = "convaiinnovations/laya") -> None:
        import laya_mlx

        self._agent = laya_mlx.load(checkpoint)
        self.name = f"laya:{checkpoint.split('/')[-1]}"

    @staticmethod
    def _encode(q: Question) -> dict[str, Any]:
        if isinstance(q, Choice):
            return {"type": "choice", "instructions": q.instructions,
                    "criteria": dict(q.criteria)}
        if isinstance(q, Noul):
            return {"type": "noul", "instructions": q.instructions}
        if isinstance(q, Score):
            return {"type": "score", "instructions": q.instructions,
                    "criteria": list(q.criteria)}
        raise TypeError(f"unknown question type: {type(q)}")

    @staticmethod
    def _flatten(state: Any) -> str:
        if isinstance(state, str):
            return state
        parts = []
        for k, v in dict(state).items():
            parts.append(f"{k}:\n{v}" if isinstance(v, str) and "\n" in v
                         else f"{k}: {v}")
        return "\n".join(parts)

    def decide(self, state: Any, questions: Mapping[str, Question]) -> Decision:
        payload = {k: self._encode(v) for k, v in questions.items()}
        t0 = time.perf_counter()
        resp = self._agent.predict(self._flatten(state), payload)
        latency_ms = (time.perf_counter() - t0) * 1000

        answers: dict[str, Answer] = {}
        for key, a in resp.get("answers", {}).items():
            kind = a.get("type")
            if kind == "choice":
                answers[key] = Answer("choice", a.get("choice"),
                                      a.get("confidence"),
                                      dict(a.get("probabilities") or {}))
            elif kind == "noul":
                answers[key] = Answer("noul", a.get("noul"), a.get("confidence"), {})
            else:
                answers[key] = Answer("score", a.get("score"), a.get("confidence"),
                                      dict(a.get("probabilities") or {}))

        return Decision(
            answers=answers, latency_ms=latency_ms,
            input_tokens=int(resp.get("usage", {}).get("input_tokens", 0)),
            model=str(resp.get("model", "laya")), billed=False,
        )


def get_backend(which: str = "jev") -> Backend:
    if which == "jev":
        return JevBackend()
    if which == "jev_or":
        return JevBackend(via="openrouter")
    if which == "laya":
        return LayaBackend()
    if which == "laya_fast":
        return LayaBackend("convaiinnovations/laya-multilingual")
    raise ValueError(f"unknown backend: {which}")
