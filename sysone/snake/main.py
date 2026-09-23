"""Snake, played by a System One model. One decision per tick.

    uv run python -m sysone.snake.main
    uv run python -m sysone.snake.main --tick 0        # model-paced
    uv run python -m sysone.snake.main --backend laya  # step 2
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from sysone.core import Decision, get_backend
from sysone.snake import policy
from sysone.snake.game import Snake

HOME = "\x1b[H\x1b[J"
DIM, BOLD, RESET = "\x1b[2m", "\x1b[1m", "\x1b[0m"
GREEN, YELLOW, RED, CYAN = "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[36m"


def bar(p: float, width: int = 22) -> str:
    filled = round(p * width)
    colour = GREEN if p > 0.6 else YELLOW if p > 0.25 else DIM
    return f"{colour}{'█' * filled}{DIM}{'·' * (width - filled)}{RESET}"


def colour_board(board: str) -> str:
    out = board.replace("@", f"{BOLD}{GREEN}@{RESET}")
    out = out.replace("*", f"{BOLD}{RED}*{RESET}")
    return out.replace("o", f"{GREEN}o{RESET}").replace(".", f"{DIM}·{RESET}")


def render(game: Snake, strategy: str, d: Decision | None, stats: dict, note: str) -> None:
    lines = [f"{HOME}{BOLD}Snake, decided by {stats['backend']}{RESET}", ""]
    for row in colour_board(game.ascii_board()).split("\n"):
        lines.append("  " + row)

    lines += [
        "",
        f"  score {BOLD}{game.score}{RESET}   length {len(game.body)}   "
        f"step {game.steps}   strategy {CYAN}{strategy}{RESET}",
        "",
    ]

    if d is not None:
        mv = d["move"]
        lines.append(f"  {BOLD}move{RESET}  -> {BOLD}{mv.value}{RESET}  "
                     f"(confidence {mv.confidence:.2f})")
        for option, p in sorted(mv.probabilities.items(), key=lambda kv: -kv[1]):
            mark = "→" if option == mv.value else " "
            lines.append(f"    {mark} {option:<6} {bar(p)} {p:0.2f}")
        lines.append("")
        lines.append(f"  {DIM}P(trapping itself){RESET} {d['trapped'].value:0.2f}    "
                     f"{DIM}P(board crowded){RESET} {d['crowded'].value:0.2f}")

    lat = stats["latencies"]
    if lat:
        p50 = statistics.median(lat)
        lines += [
            "",
            f"  {DIM}latency{RESET} {lat[-1]:6.0f} ms   {DIM}p50{RESET} {p50:5.0f} ms   "
            f"{DIM}decisions/s{RESET} {1000 / p50:4.1f}",
            f"  {DIM}tokens {stats['tokens']:,}   cost ${stats['usd']:.5f}   "
            f"late ticks {stats['late']}{RESET}",
        ]
    if note:
        lines += ["", f"  {YELLOW}{note}{RESET}"]

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="jev")
    ap.add_argument("--tick", type=float, default=0.0,
                    help="seconds per tick; 0 means run at the model's own pace")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--compact", action="store_true",
                    help="drop the ASCII board from state (fits laya's 512-tok context)")
    args = ap.parse_args()

    backend = get_backend(args.backend)
    game = Snake(width=args.size, height=args.size)
    strategy = "CHASE_FOOD"
    stats = {"backend": backend.name, "latencies": [], "tokens": 0, "usd": 0.0, "late": 0}
    note = ""
    decision = None

    try:
        while game.alive and game.steps < args.max_steps:
            t_tick = time.perf_counter()
            survivable = game.survivable_moves()

            if not survivable:
                note = "no survivable move -- boxed in"
                game.alive = False
                render(game, strategy, decision, stats, note)
                break

            if len(survivable) == 1:
                # Nothing to decide. Don't spend a call on a forced move.
                only = next(iter(survivable))
                game.step(only)
                note = f"forced move {only} (no call made)"
                render(game, strategy, decision, stats, note)
                continue

            state = policy.build_state(game, strategy, compact=args.compact)
            questions = policy.per_tick_questions(survivable, strategy)

            # Tiered goal: re-plan the strategy occasionally, in the same call.
            replanning = game.steps % policy.STRATEGY_EVERY == 0
            if replanning:
                questions["strategy"] = policy.strategy_question(strategy)

            try:
                decision = backend.decide(state, questions)
            except Exception as exc:  # network blip, timeout, rate limit
                stats["late"] += 1
                note = f"{type(exc).__name__}: falling through on {game.direction}"
                game.step(game.direction)
                render(game, strategy, decision, stats, note)
                continue

            stats["latencies"].append(decision.latency_ms)
            stats["tokens"] += decision.input_tokens
            stats["usd"] += decision.usd
            note = ""

            if replanning and "strategy" in decision.answers:
                strategy = str(decision["strategy"].value)

            chosen = str(decision["move"].value)
            if chosen not in survivable:
                chosen = min(survivable, key=lambda m: survivable[m].food_distance)
                note = "model chose an unavailable move -- fell back to nearest-food"

            game.step(chosen)
            render(game, strategy, decision, stats, note)

            if args.tick:
                remaining = args.tick - (time.perf_counter() - t_tick)
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    stats["late"] += 1

    except KeyboardInterrupt:
        pass

    lat = stats["latencies"]
    print(f"\n{BOLD}game over{RESET}  score {game.score} in {game.steps} steps")
    if lat:
        print(f"  {len(lat)} decisions   p50 {statistics.median(lat):.0f} ms   "
              f"p95 {sorted(lat)[int(len(lat) * 0.95)]:.0f} ms   "
              f"min {min(lat):.0f} ms")
        print(f"  {stats['tokens']:,} input tokens   ${stats['usd']:.5f} total   "
              f"${stats['usd'] / max(game.steps, 1) * 1000:.4f} per 1000 steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
