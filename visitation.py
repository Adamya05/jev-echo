"""Which game states do policies actually arrive at?

The motivating question: if the hard states are rare, then being good at
them barely moves the score, and every policy looks the same. If they are
common, the differences should show. This measures that directly instead
of inferring it from the score.

A state is "tight" when at least one legal move leaves less open space
than the snake's own length -- i.e. taking it plausibly seals the snake in.
That is the only kind of position where judgement can save you, so it is
the only kind where a better model can pay for itself.

    uv run python visitation.py --games 300 --max-steps 5000
"""

from __future__ import annotations

import argparse
import collections
import random
import statistics

from sysone.snake import search
from sysone.snake.game import Snake

BOLD, DIM, RESET = "\x1b[1m", "\x1b[2m", "\x1b[0m"


def walk(config: str, seed: int, max_steps: int) -> dict:
    pol = search.build(config, None)
    game = Snake(seed=seed)
    rng = random.Random(seed * 7919 + 13)

    lengths, tight, forced, critical_len = [], 0, 0, []
    decisions = 0

    while game.alive and game.steps < max_steps:
        facts = game.survivable_moves()
        if not facts:
            break
        if len(facts) == 1:
            forced += 1
            game.step(next(iter(facts)))
            continue

        decisions += 1
        lengths.append(len(game.body))
        is_tight = any(f.open_space < len(game.body) for f in facts.values())
        if is_tight:
            tight += 1
            critical_len.append(len(game.body))

        game.step(pol.select(game, rng).move)

    return {"lengths": lengths, "tight": tight, "forced": forced,
            "decisions": decisions, "score": game.score, "steps": game.steps,
            "critical_len": critical_len, "final_len": len(game.body),
            "died": not game.alive or not game.survivable_moves()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--configs", nargs="*",
                    default=["student_prior", "student_improved"])
    ap.add_argument("--cap-note", type=int, default=110,
                    help="also report what fraction of play happens before this step")
    args = ap.parse_args()

    for cfg in args.configs:
        runs = [walk(cfg, s, args.max_steps) for s in range(args.games)]
        name = search.build(cfg, None).name

        all_len = [l for r in runs for l in r["lengths"]]
        dec = sum(r["decisions"] for r in runs)
        tight = sum(r["tight"] for r in runs)
        forced = sum(r["forced"] for r in runs)
        crit = [l for r in runs for l in r["critical_len"]]

        print(f"\n{BOLD}{name}{RESET}")
        print(f"  score {statistics.mean(r['score'] for r in runs):.1f}   "
              f"steps {statistics.mean(r['steps'] for r in runs):.0f}   "
              f"died {sum(r['died'] for r in runs)}/{len(runs)}")
        print(f"  {dec:,} real decisions, {forced:,} forced "
              f"({forced/(dec+forced):.0%} of moves need no model at all)")
        print(f"  {BOLD}tight states: {tight/max(dec,1):.1%}{RESET} of decisions "
              f"(some move leaves less room than the snake is long)")
        if crit:
            print(f"    they start at snake length {min(crit)}, "
                  f"median {statistics.median(crit):.0f}")

        print(f"  snake length at decision time: "
              f"median {statistics.median(all_len):.0f}, "
              f"p90 {sorted(all_len)[int(len(all_len)*.9)]}, "
              f"max {max(all_len)}")

        hist = collections.Counter(min(l // 6 * 6, 36) for l in all_len)
        print(f"  {DIM}where the game is spent:{RESET}")
        for k in sorted(hist):
            share = hist[k] / len(all_len)
            label = f"{k}-{k+5}" if k < 36 else "36+"
            print(f"    len {label:<7} {share:>6.1%}  {'#' * int(share * 90)}")

        short = sum(1 for l in all_len if l < 14)
        print(f"  {DIM}a 110-step game never gets past length ~17, so it only ever"
              f" sees the top {short/len(all_len):.0%} of this distribution{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
