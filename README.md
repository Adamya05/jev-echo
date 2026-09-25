# Jev and Echo

Echo is a 22,212-parameter network (87 KB) trained on the answers of
[Jev](https://typesafe.ai), TypeSafe's System One model, for one narrow task:
Snake. It reads the same board Jev does and runs on a laptop CPU, about
800× faster. Given a tree search, it scores about 5× what Jev does, still
taking less time per move than Jev.

**[See the results, then play the same board yourself →](https://adamya05.github.io/jev-echo/)**

## Results

20 boards that no training or tuning touched (boards 100–119), each played to
the end. A crash ends the game.

| | Average score | Time per move |
|---|---|---|
| Jev | 7.5 | 159 ms |
| Echo | 7.9 · no clear difference from Jev | 0.20 ms |
| Echo + search | 38.8 · won 20 of 20 | 103 ms |

On games it never trained on, Echo picks the same move as Jev 93% of the time
(KL divergence 0.013 nats; a uniform guess is 0.66).

Playing the way Echo + search does would take Jev about 361 calls per move:
roughly 6.1 hours and $2.76 a game.

## How it works

**Jev** gets the board as text, the head and food positions, and one
instruction: head for the food without getting trapped. It returns a
probability for each move. See `sysone/snake/policy.py`.

**Echo** gets the same board as three 10×10 grids (body, head, food) and
returns a probability for each move. It is trained to match Jev's full
probabilities (KL divergence), with each board weighted by how sure Jev was.
Copying Jev's games alone isn't enough when one slip ends the game, because
Echo never sees the positions its own mistakes lead to. So Echo then drives,
Jev labels the boards it actually reaches, and Echo retrains, twice (DAgger).
38,199 boards in all, $0.77 of Jev calls.

**Echo + search** looks several moves ahead (usually 7–8, as deep as the time
budget allows), expanding every move at every level and asking Echo about a
whole level at once. A leaf is worth the pellets eaten on the way, or
−(score + 10) if the snake died: it loses everything so far, plus a fixed cost
for dying. Each node weighs its children by Echo's probability times
exp(value), so moves the tree sees are fatal drop out without any hard-coded
rule, and when everything is equally good it just follows Echo. Simulated
food lands on a random empty cell, never where the real game will put it.
See `TreeSearch` in `sysone/snake/search.py`.

## Run the page

```bash
python3 -m http.server 8000 --directory docs
```

Then open http://localhost:8000. The page is static: `docs/index.html` and `docs/landing-assets/`.

## Reproduce

Needs an Apple Silicon Mac (Echo runs on [MLX](https://github.com/ml-explore/mlx)),
[uv](https://docs.astral.sh/uv/), and an [OpenRouter](https://openrouter.ai) key with
access to `typesafe/jev-1.13`. Everything below costs about $1 of Jev calls.

```bash
uv sync
cp .env.example .env        # then add OPENROUTER_API_KEY
export SNAKE_ALLOW_CRASH=1  # crashes end the game, as on the page
```

Train Echo:

```bash
uv run python collect_raw.py --games 90 --out results/dataset_board_crash.npz   # 14,391 boards, $0.29
uv run python train_board.py --data results/dataset_board_crash.npz --out results/student_board_crash_v1.npz

# Let Echo drive and have Jev label where it ends up, twice (23,808 boards, $0.48)
uv run python collect_raw.py --driver echo --echo results/student_board_crash_v1.npz \
    --games 250 --max-steps 2000 --seed-offset 5000 --out results/dataset_board_crash_dagger1.npz
uv run python train_board.py --out results/student_board_crash_d1.npz \
    --data results/dataset_board_crash.npz results/dataset_board_crash_dagger1.npz
uv run python collect_raw.py --driver echo --echo results/student_board_crash_d1.npz \
    --games 250 --max-steps 2000 --seed-offset 6000 --out results/dataset_board_crash_dagger2.npz
uv run python train_board.py --out results/student_board_crash.npz \
    --data results/dataset_board_crash.npz results/dataset_board_crash_dagger1.npz \
           results/dataset_board_crash_dagger2.npz
```

Race them on the held-out boards, record the replay and the example position,
and build the page:

```bash
# One worker, so nobody's timing (or the tree's depth) is sharing the CPU
uv run python -m sysone.snake.compare --games 20 --seed-offset 100 --max-steps 2000 --workers 1 \
    --configs jev_raw board_student board_tree_181 --save results/heldout_crash.json
uv run python record_replay.py      # the race on the page, under $0.01
uv run python example_move.py       # the "How they play" position, one Jev call
uv run python viz/build.py          # -> docs/race.html (the race replay; the front page is static)
# docs/og.png, the link preview, is the top of the front page at 1200x630:
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --window-size=1200,630 \
    --screenshot=docs/og.png http://localhost:8000/
node viz/video.mjs echo.mp4         # optional: a 15-second video of the race
```

## What's where

| | |
|---|---|
| `sysone/snake/game.py` | the game; `SNAKE_ALLOW_CRASH=1` lets a crash end it; `sim_clone()` for search |
| `sysone/snake/policy.py` | the prompts Jev sees |
| `sysone/snake/boardnet.py` | Echo: a small conv net over the raw board |
| `sysone/snake/search.py` | every player compared; `TreeSearch` is Echo + search |
| `sysone/snake/compare.py` | paired comparisons on the same boards |
| `sysone/core.py` | one `decide()` interface for Jev (TypeSafe or OpenRouter) and Laya |
| `collect_raw.py`, `train_board.py` | Jev labels boards; Echo trains on them |
| `record_replay.py`, `example_move.py` | the race replay (`docs/race.html`, not deployed) and the video's example position |
| `viz/` | the race replay's template and build script, and `snake_core.js`, a JS port of Python's RNG so "Your turn" deals the same board (copied to `docs/landing-assets/`) |
| `docs/` | the site: the front page, its assets, the step-through search explainer, and the four-minute video (`video.html`, `landing-assets/walkthrough.mp4` with captions) |
| `results/` | the trained Echo (`student_board_crash.npz`) and the held-out runs (`heldout_crash*.json`); everything else there is regenerated by the scripts |

## Earlier experiments

Kept in the repo; the page doesn't need them.

- **A version where the game never offers a fatal move** (tag `v1-no-crash`).
  There, plain copying was enough: Echo matched Jev without the extra DAgger
  rounds. Check out that tag to run it.
- **A flat Monte Carlo search** (`BudgetSearch`): about 50 random 8-move
  futures per move. On boards 0–9 it beat Jev by +69%; the tree does 5× on
  the held-out boards.
- **A leak.** Search used to copy the game's random generator, so it knew
  where the next food would appear. `sim_clone()` fixed it; on boards 0–9 the
  flat search's lead over Jev fell from +93% to +69%.
- **Other tree backups** (plain expectation, max, fixed death penalties, Echo's
  vote at the root) are in commit `142ef1e`. The one kept beat them on the
  tuning boards (0–9), and beat the first tree on the held-out boards.
- **A student on precomputed features** (`features.py`, `student.py`, `train.py`),
  **[Laya](https://github.com/mizorewww/laya-mlx)** as an off-the-shelf local
  model (`collect_laya.py`, scored near zero), **MCTS and best-of-N** on Jev's
  answers, **`bench.py`** (Jev's fixed ~268-token overhead per call),
  **`visitation.py`** and **`sysone/computer`** (macOS computer use via the
  accessibility tree).

Other search classes in `search.py` still copy the game with `clone()`, so
they can see one future pellet; only the tree and the flat search were fixed.

## Caveats

- One small task: a 10×10 grid with four moves.
- Looking ahead works because Snake's rules predict the next board exactly.
  Most real tasks can't be simulated that cheaply.
- 20 boards is a small sample.
- Jev's time per move varies between runs (about 160–230 ms), so the speed-up
  is anywhere from about 800× to 1,100×. The page uses the held-out run.
- The playable board's recorded scores are board 0, which the tree was tuned
  on; the averages are not. The step-through explainer is board 9000, move 176,
  chosen to illustrate a delayed trap.
- Scores and timings on the page are recordings made on an Apple M4. Only the
  playable board runs live, in your browser.

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
