# Jev and Echo

Echo is a 22,212-parameter network (87 KB) trained on the answers of
[Jev](https://typesafe.ai), TypeSafe's System One model. It reads the same
Snake board Jev does and runs on a laptop CPU, about a thousand times faster.

**[Watch them play, then try the same board yourself →](https://adamya05.github.io/jev-echo/)**

## Results

Ten games per player, all three on the same ten boards, each played to the end.

| | Safety net on | Safety net off |
|---|---|---|
| Jev | 21.5 | 8.9 |
| Echo | 17.8 · no clear difference | 7.7 · no clear difference |
| Echo + search | 29.1 · **+35%, won 9 of 10** | 17.2 · **+93%, won 9 of 10** |
| Time per move, Jev / Echo | 215 ms / 0.20 ms | 227 ms / 0.21 ms |

**Safety net on:** the models are only offered moves that won't crash them.
**Off:** they can pick any direction, so one bad move ends the game.

**Echo + search** is Echo given Jev's time per move (or a little less) and using
it to play each option forward before choosing.

## Run the page

```bash
python3 -m http.server 8000 --directory docs
```

Then open http://localhost:8000. The page is one self-contained file.

## Reproduce

Needs an Apple Silicon Mac (Echo runs on [MLX](https://github.com/ml-explore/mlx)),
[uv](https://docs.astral.sh/uv/), and an [OpenRouter](https://openrouter.ai) key with
access to `typesafe/jev-1.13`.

```bash
uv sync
cp .env.example .env        # then add OPENROUTER_API_KEY
```

Safety net on:

```bash
uv run python collect_raw.py --games 90        # Jev labels ~13.7k boards, ~5 min, ~$0.27
uv run python train_board.py                   # train Echo, ~30 s
uv run python -m sysone.snake.compare --games 10 --max-steps 2000 \
    --configs jev_raw board_student board_budget_197 --save results/fair_uncapped.json
uv run python record_replay.py                 # the replay on the page, ~$0.01
```

Safety net off. Copying Jev's games isn't enough here: one slip ends the game, and
Echo never saw the boards its own slips lead to. So after the first round, Echo
drives and Jev labels where it ends up (DAgger), twice. That took Echo from 6.0 to
7.7 for another ~$0.48.

```bash
export SNAKE_ALLOW_CRASH=1
uv run python collect_raw.py --games 90 --out results/dataset_board_crash.npz
uv run python train_board.py --data results/dataset_board_crash.npz --out results/student_board_crash_v1.npz
uv run python collect_raw.py --driver echo --echo results/student_board_crash_v1.npz \
    --games 250 --max-steps 2000 --seed-offset 5000 --out results/dataset_board_crash_dagger1.npz
uv run python train_board.py --data results/dataset_board_crash.npz results/dataset_board_crash_dagger1.npz \
    --out results/student_board_crash_d1.npz
uv run python collect_raw.py --driver echo --echo results/student_board_crash_d1.npz \
    --games 250 --max-steps 2000 --seed-offset 6000 --out results/dataset_board_crash_dagger2.npz
uv run python train_board.py --data results/dataset_board_crash.npz results/dataset_board_crash_dagger1.npz \
    results/dataset_board_crash_dagger2.npz --out results/student_board_crash.npz
uv run python -m sysone.snake.compare --games 10 --max-steps 2000 \
    --configs jev_raw board_student board_budget_181 --save results/fair_crash_uncapped.json
uv run python record_replay.py
unset SNAKE_ALLOW_CRASH
```

Then build the page from both:

```bash
uv run python viz/build.py                     # -> docs/index.html
```

Set the `board_budget_…` number to Jev's measured time per move, which varies
a little from run to run, so the search really does get the same time.

## What's where

| | |
|---|---|
| `sysone/snake/game.py` | the game, and the `SNAKE_ALLOW_CRASH` switch |
| `sysone/snake/policy.py` | the prompts Jev sees |
| `sysone/snake/boardnet.py` | Echo: a small conv net over the raw board |
| `sysone/snake/search.py` | every player compared, including Echo + search |
| `sysone/snake/compare.py` | paired ten-game comparisons |
| `sysone/core.py` | one `decide()` interface for Jev (TypeSafe or OpenRouter) and Laya |
| `viz/` | page template, build script, and a JS port of Python's RNG so "Your turn" deals the same board |
| `docs/` | the built page |

## Earlier experiments

Kept in the repo; the page doesn't need them.

- **A student on precomputed features** (`features.py`, `student.py`, `train.py`).
  It matched Jev, but on 28 numbers worked out by code rather than the raw board.
- **[Laya](https://github.com/mizorewww/laya-mlx)** as an off-the-shelf local System One
  model (`collect_laya.py`). It scored near zero here, which is why Echo exists.
- **MCTS and best-of-N search** on Jev's answers, in `search.py`.
- **`bench.py`**: Jev's token economics. There's a fixed ~268-token overhead per call,
  so ask several questions per call.
- **`visitation.py`**: which board states games actually reach.
- **`sysone/computer`**: macOS computer use through the accessibility tree, driven by
  Jev. `--fixture` screens run without any permissions.

## Caveats

- One small task: a 10×10 grid with four moves.
- Thinking ahead works because Snake's rules predict the next board exactly.
  Most real tasks can't be simulated that cheaply.
- Ten games per player is a small sample.
- The page replays recordings made on an Apple M4. Nothing runs live.

## Credits and reading

- [sorrycc/typesafe-snake](https://github.com/sorrycc/typesafe-snake), the first Jev Snake
- [laya-mlx](https://github.com/mizorewww/laya-mlx)
- Sean Goedecke, [Two techniques for working with System One models](https://www.seangoedecke.com/two-techniques-for-working-with-system-one-models/)
- [Monte Carlo Tree Search, a beginner's guide](https://int8.io/monte-carlo-tree-search-beginners-guide/)
- [AlphaZero](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

## License

MIT
