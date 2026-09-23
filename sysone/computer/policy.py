"""Turning a screen into questions, and answers into safe actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sysone.core import Answer, Backend, Choice, Noul
from sysone.computer.ax import Element, Screen

BATCH = 20           # elements per tournament heat
MAX_DIRECT = 24      # above this, run a tournament instead of one big Choice

OPERATIONS = {
    "CLICK": "Press the chosen element.",
    "TYPE": "Write text into the chosen editable field.",
    "WAIT": "The screen is still loading or mid-transition; observe again.",
    "DONE": "The goal is already satisfied by what is on screen.",
    "BLOCKED": "The goal cannot be advanced from this screen.",
}

# Words that make an action expensive to get wrong.
CONSEQUENTIAL = {
    "send", "delete", "remove", "buy", "purchase", "pay", "confirm",
    "submit", "publish", "post", "share", "trash", "erase", "discard",
    "sign out", "log out", "unsubscribe", "transfer", "move to trash",
}


@dataclass(frozen=True)
class Gate:
    ok: bool
    reason: str


def is_consequential(label: str) -> bool:
    low = label.lower()
    return any(w in low for w in CONSEQUENTIAL)


def build_state(screen: Screen, goal: str, history: list[str]) -> dict[str, Any]:
    return {
        "goal": goal,
        "application": screen.app,
        "window": screen.window,
        "steps_taken": history[-4:] or ["(nothing yet)"],
        "elements": [e.for_model() for e in screen.elements],
    }


def batch_state(screen: Screen, goal: str, batch: list[Element]) -> dict[str, Any]:
    return {
        "goal": goal,
        "application": screen.app,
        "window": screen.window,
        "elements": [e.for_model() for e in batch],
    }


def element_choice(batch: list[Element], goal: str) -> Choice:
    return Choice(
        instructions=(
            f"Which single element on this screen best advances the goal: {goal!r}? "
            "Judge by label, role and whether it is enabled. If none of them help, "
            "pick the least wrong one -- a later round decides whether to act at all."
        ),
        criteria={e.id: e.describe() for e in batch},
    )


def choose_element(backend: Backend, screen: Screen, goal: str
                   ) -> tuple[Element | None, Answer, int]:
    """Pick one element, using a tournament when the screen is crowded.

    Tournament sampling (Goedecke): models judge *relatively* far better
    than they rate absolutely, so narrow by comparison in heats rather
    than scoring hundreds of options at once. Each round is a single
    call -- Jev answers every heat against its state in one parallel
    pass -- so a 200-element screen costs two calls, not two hundred.
    """
    live = [e for e in screen.elements if e.enabled]
    if not live:
        return None, Answer("choice", "", 0.0, {}), 0

    if len(live) <= MAX_DIRECT:
        d = backend.decide(build_state(screen, goal, []), {"pick": element_choice(live, goal)})
        return screen.by_id(str(d["pick"].value)), d["pick"], 1

    heats = [live[i:i + BATCH] for i in range(0, len(live), BATCH)]
    questions = {f"heat{i}": element_choice(h, goal) for i, h in enumerate(heats)}
    round1 = backend.decide({"goal": goal, "application": screen.app,
                             "window": screen.window,
                             "note": "each question covers a different slice of the screen"},
                            questions)

    winners: list[Element] = []
    for i, heat in enumerate(heats):
        ans = round1.answers.get(f"heat{i}")
        if ans is None:
            continue
        won = next((e for e in heat if e.id == str(ans.value)), None)
        if won is not None:
            winners.append(won)

    if not winners:
        return None, Answer("choice", "", 0.0, {}), 1
    if len(winners) == 1:
        only = next(a for k, a in round1.answers.items() if k.startswith("heat"))
        return winners[0], only, 1

    final = backend.decide(build_state(screen, goal, []),
                           {"pick": element_choice(winners, goal)})
    return screen.by_id(str(final["pick"].value)), final["pick"], 2


def step_questions(screen: Screen, goal: str, candidate: Element | None) -> dict[str, Any]:
    """The gates, all evaluated against one state in a single call."""
    label = candidate.label if candidate else "(none)"
    qs: dict[str, Any] = {
        "operation": Choice(
            instructions=f"Goal: {goal!r}. The element under consideration is "
                         f"{candidate.describe() if candidate else 'nothing usable'}. "
                         "What should happen next?",
            criteria=dict(OPERATIONS),
        ),
        "complete": Noul(
            instructions=f"Is the goal {goal!r} already fully satisfied by what is "
                         "currently on screen, with nothing left to do?"
        ),
        "right_element": Noul(
            instructions=f"Is {label!r} genuinely the right thing to act on for "
                         f"the goal {goal!r}?"
        ),
    }
    if candidate and is_consequential(candidate.label):
        qs["authorized"] = Noul(
            instructions=f"Acting on {label!r} is hard to undo. Does the goal "
                         f"{goal!r} explicitly and unambiguously ask for this, "
                         "rather than merely implying it?"
        )
    return qs


def gate(op: str, answers: dict[str, Answer], candidate: Element | None) -> Gate:
    """Thresholds before anything touches the machine.

    Deliberately strict, and strictest where mistakes are permanent.
    Calibrated probabilities are the point of a System One model: this
    is the part you cannot write against an LLM that only emits JSON.
    """
    operation = answers["operation"]
    if operation.p < 0.55:
        return Gate(False, f"operation {op} only p={operation.p:.2f} (need 0.55)")
    if (operation.confidence or 0) < 0.35:
        return Gate(False, f"operation confidence {operation.confidence:.2f} (need 0.35)")

    if op in {"WAIT", "DONE", "BLOCKED"}:
        return Gate(True, "no mutation")

    if candidate is None:
        return Gate(False, "no element selected")
    if answers["right_element"].p < 0.55:
        return Gate(False, f"element only p={answers['right_element'].p:.2f} right (need 0.55)")

    if is_consequential(candidate.label):
        if operation.p < 0.85:
            return Gate(False, f"consequential: operation p={operation.p:.2f} (need 0.85)")
        if (operation.confidence or 0) < 0.75:
            return Gate(False, f"consequential: confidence {operation.confidence:.2f} (need 0.75)")
        auth = answers.get("authorized")
        if auth is None or auth.p < 0.90:
            got = f"{auth.p:.2f}" if auth else "missing"
            return Gate(False, f"consequential: authorization {got} (need 0.90)")

    return Gate(True, "cleared")
