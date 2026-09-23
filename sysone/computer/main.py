"""Computer use on macOS, decided by a System One model.

Reads the screen through Accessibility -- no screenshots, no vision
model, no coordinates. Dry run unless you pass --execute.

    uv run python -m sysone.computer.main --fixture mail --goal "send the email"
    uv run python -m sysone.computer.main --goal "open the Displays settings"
    uv run python -m sysone.computer.main --goal "..." --execute
"""

from __future__ import annotations

import argparse
import sys
import time

from sysone.computer import ax, policy
from sysone.computer.fixture import SCREENS
from sysone.core import get_backend

BOLD, DIM, RESET = "\x1b[1m", "\x1b[2m", "\x1b[0m"
GREEN, YELLOW, RED, CYAN = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[36m"

MAX_DECISIONS = 30
MAX_MUTATIONS = 20
TIME_BUDGET_S = 90


def bar(p: float, width: int = 18) -> str:
    f = round(p * width)
    c = GREEN if p > 0.6 else YELLOW if p > 0.3 else DIM
    return f"{c}{'█' * f}{DIM}{'·' * (width - f)}{RESET}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", required=True)
    ap.add_argument("--app", default=None, help="target app (default: frontmost)")
    ap.add_argument("--fixture", choices=sorted(SCREENS), default=None)
    ap.add_argument("--backend", default="jev")
    ap.add_argument("--execute", action="store_true",
                    help="actually act on the machine (default: dry run)")
    ap.add_argument("--steps", type=int, default=1,
                    help="how many decision steps to run")
    args = ap.parse_args()

    if args.execute and args.fixture:
        print("--execute is meaningless against a fixture; fixtures have no live refs.")
        return 2

    backend = get_backend(args.backend)
    mode = f"{RED}EXECUTE{RESET}" if args.execute else f"{GREEN}dry run{RESET}"
    print(f"\n{BOLD}Computer use via {backend.name}{RESET}   mode: {mode}")
    print(f"goal: {CYAN}{args.goal}{RESET}\n")

    history: list[str] = []
    calls = tokens = mutations = 0
    usd = 0.0
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        if calls >= MAX_DECISIONS or mutations >= MAX_MUTATIONS:
            print(f"{YELLOW}loop budget reached{RESET}")
            break
        if time.perf_counter() - t_start > TIME_BUDGET_S:
            print(f"{YELLOW}time budget reached{RESET}")
            break

        try:
            screen = SCREENS[args.fixture]() if args.fixture else ax.capture(args.app)
        except PermissionError as exc:
            print(f"{RED}{exc}{RESET}")
            return 1
        except LookupError as exc:
            print(f"{RED}{exc}{RESET}")
            return 1

        live = [e for e in screen.elements if e.enabled]
        print(f"{BOLD}step {step}{RESET}  {screen.app} / {screen.window}   "
              f"{DIM}{len(screen.elements)} elements, captured in "
              f"{screen.capture_ms:.1f} ms{RESET}")

        t0 = time.perf_counter()
        candidate, pick, rounds = policy.choose_element(backend, screen, args.goal)
        calls += rounds

        if len(live) > policy.MAX_DIRECT:
            print(f"  {DIM}tournament: {len(live)} elements -> "
                  f"{rounds} call(s){RESET}")

        if candidate is None:
            print(f"  {RED}no usable element{RESET}")
            break

        print(f"  element  {BOLD}{candidate.label}{RESET} "
              f"{DIM}({candidate.role.removeprefix('AX')}, {candidate.position}){RESET}  "
              f"p={pick.p:.2f}")
        runners = sorted(pick.probabilities.items(), key=lambda kv: -kv[1])[1:4]
        for eid, p in runners:
            el = screen.by_id(eid)
            if el and p > 0.01:
                print(f"    {DIM}vs {el.label:<28}{RESET} {bar(p)} {p:.2f}")

        qs = policy.step_questions(screen, args.goal, candidate)
        d = backend.decide(policy.build_state(screen, args.goal, history), qs)
        calls += 1
        tokens += d.input_tokens
        usd += d.usd
        elapsed = (time.perf_counter() - t0) * 1000

        op = str(d["operation"].value)
        print(f"  operation {BOLD}{op}{RESET} {bar(d['operation'].p)} "
              f"p={d['operation'].p:.2f} conf={d['operation'].confidence:.2f}")
        print(f"  {DIM}goal complete{RESET} {d['complete'].p:.2f}   "
              f"{DIM}right element{RESET} {d['right_element'].p:.2f}"
              + (f"   {DIM}authorized{RESET} {d['authorized'].p:.2f}"
                 if "authorized" in d.answers else ""))

        g = policy.gate(op, dict(d.answers), candidate)
        consequential = policy.is_consequential(candidate.label)
        tag = f" {RED}[consequential]{RESET}" if consequential else ""

        if not g.ok:
            print(f"  {YELLOW}BLOCKED{RESET}{tag}: {g.reason}")
            print(f"  {DIM}total {elapsed:.0f} ms{RESET}\n")
            break

        action = f"{op} {candidate.label!r}"
        if args.execute and op in {"CLICK", "TYPE"}:
            done = ax.press(candidate) if op == "CLICK" else False
            mutations += 1
            print(f"  {GREEN}executed{RESET}{tag} {action} -> {'ok' if done else 'failed'}")
            time.sleep(0.4)
        elif op in {"DONE", "BLOCKED"}:
            print(f"  {GREEN}{op}{RESET}: {policy.OPERATIONS[op]}")
            history.append(op)
            print(f"  {DIM}total {elapsed:.0f} ms{RESET}\n")
            break
        else:
            print(f"  {GREEN}would {action}{RESET}{tag} {DIM}(dry run){RESET}")

        history.append(action)
        print(f"  {DIM}total {elapsed:.0f} ms{RESET}\n")

    print(f"{DIM}{calls} calls   {tokens:,} tokens   ${usd:.5f}   "
          f"{mutations} mutations{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
