# System One demos

Two demos on TypeSafe's Jev, built so the model can be swapped for a local
one without touching the demos themselves.

Everything talks to one seam, in `sysone/core.py`:

```python
backend.decide(state, questions) -> {name: Answer}
```

Three question types, which is the entire API surface:

| type     | returns                                      |
|----------|----------------------------------------------|
| `Choice` | distribution over up to 255 named options     |
| `Noul`   | P(true) for a proposition                     |
| `Score`  | distribution over an ordered 2-10 level rubric |

## Setup

```bash
uv sync
echo 'TYPESAFE_API_KEY=sk-...' > .env
```

## Read this first: what later measurements overturned

Several conclusions below were measured with games capped at **110 steps**, and
that cap turned out to change the task rather than just shorten it. Score is
*exactly* 110 / steps-per-food for every config -- travel efficiency, with no
skill term. Uncapped, the same players score **34 instead of 14** and die in 249
of 250 games.

A capped game never gets the snake past length ~17, the bottom 10-27% of the
real state distribution, and the part where **0% of decisions are tight** (a
tight state is one where some legal move leaves less room than the snake is
long -- the only kind where judgement can pay for itself). Uncapped, 14-38% of
decisions are tight.

So the capped sections below measure a problem that required no judgement.
Specifically:

| stated below | actually |
|--------------|----------|
| "the prior is the ceiling, search is worth ~5%" | the *step cap* was the ceiling; search is worth **+53%** uncapped |
| "three-tier latency: cloud 165ms -> laya ~10ms -> student 0.43ms" | laya runs **269-448 ms** here, slower than the cloud |
| "a 6k-param MLP matches hosted Jev" | true, but **only measured capped**; untested where the snake dies |

The uncapped numbers and the full reasoning are in the session notes.

## Snake

One decision per tick. Code computes flood fill, dead ends and distances;
the model only chooses among moves that are already known to be legal and
survivable.

```bash
uv run python -m sysone.snake.main              # model-paced
uv run python -m sysone.snake.main --tick 0.25  # fixed clock
uv run python -m sysone.snake.main --compact    # drop the board from state
```

Measured on `jev-1.13.0`: **p50 160 ms, ~6.4 decisions/sec, score 15 in 150
steps, $0.024 per 1000 decisions.**

Two techniques are built in:

- **Tiered goals.** A strategy (`CHASE_FOOD` / `PLAY_SAFE` / `FOLLOW_TAIL`)
  is re-chosen every 8 ticks *in the same call* as the move, and injected
  into the next 8 per-tick prompts. Planning behaviour, no planner.
- **Forced moves cost nothing.** When only one move survives, no call is
  made. Decide whether to call before deciding what to ask.

## Computer use

Reads the screen through macOS Accessibility. No screenshots, no vision
model, no coordinates. Capture is **2-6 ms** and a few hundred tokens.

```bash
# safe, no permissions needed -- synthetic screens
uv run python -m sysone.computer.main --fixture mail     --goal "reply to this and send it"
uv run python -m sysone.computer.main --fixture settings --goal "open the Displays settings"
uv run python -m sysone.computer.main --fixture confirm  --goal "get rid of these files"

# live, read-only
uv run python -m sysone.computer.main --goal "open the Displays settings"

# live, actually acts
uv run python -m sysone.computer.main --goal "..." --execute
```

**Dry run is the default.** `--execute` is the only thing that touches the
machine, and consequential labels (send, delete, buy, publish...) must clear
a separate authorization gate first.

Live capture needs Accessibility permission: System Settings > Privacy &
Security > Accessibility, enable whichever app runs this. Grant it yourself;
nothing here changes security settings.

### Tournament sampling

Above 24 enabled elements the screen is split into heats of 20. Because Jev
answers every question against one state in a single parallel pass, **a whole
tournament round is one call** — a 38-element screen resolves in 2 calls, not
38. Models judge relatively far better than they rate absolutely, so narrowing
by comparison beats scoring everything at once.

### The gates

Calibrated probabilities are the thing you cannot get from an LLM emitting
JSON, so the safety policy is plain arithmetic:

| action                        | operation p | confidence | authorization |
|-------------------------------|-------------|------------|---------------|
| ordinary                      | >= 0.55     | >= 0.35    | --            |
| consequential (send, delete)  | >= 0.85     | >= 0.75    | >= 0.90       |

Authorization tracks how explicitly the goal asked for it:

```
clean up my downloads folder                   auth 0.15   blocked
I want these gone                              auth 0.73   blocked
get rid of these files                         auth 0.77   blocked
delete the 3 selected files                    auth 0.93   PASS
click Delete to permanently remove ... certain auth 0.95   PASS
```

## Search on the prior (phase 1)

Jev's `Choice` distribution is a categorical policy head, pi(a|s). It is a
usable **prior** and nothing else:

- RLCD trains for **calibration**, not return maximization. When it says
  0.7 it is right 70% of the time -- that is not a value estimate, and
  `confidence` must never be read as one.
- **No credit assignment.** Every call is a stateless one-step judgment,
  so there is no learning loop to inherit.
- The **action space is an input, not an architecture.** `survivable_moves()`
  hands it a different option set with different natural-language facts
  every tick. That is why zero-shot works at all, and why this is unlike a
  fixed 4-way softmax head.

You cannot improve the black box. You improve a copy of it (phase 2), or
you improve its output with search:

```bash
uv run python -m sysone.snake.compare --games 4 --max-steps 110
```

4 seeds, 110-step cap, 160 MCTS iterations. Every game hit the step cap and
none died, so **score is food per 110 steps -- a rate, not a ceiling**:

| config              | score | ms/decision | $ / 4 games | greedy | open space |
|---------------------|-------|-------------|-------------|--------|------------|
| JEV                 | 12.2  |     179     |   0.0104    |  72%   |     91     |
| JEV+BESTOF          | 12.8  |     186     |   0.0101    |  77%   |     92     |
| JEV+MCTS(160)       | 13.2  |   6,619     |   0.0102    |  80%   |     91     |
| **UNIFORM+MCTS(160)** | **0.8** |   6,078 |   0.0000    |  44%   |     99     |

**The control is the result.** UNIFORM+MCTS scores 0.8. The identical search
with a Jev root prior scores 13.2. The prior is doing essentially all of the
work, and the search is worth about +1.0 over the prior alone -- which at
n=4 is inside the noise, and costs 37x the wall clock to obtain. The
100-game harness in phase 3 is what settles whether that +1.0 is real.

### At n=100 (seeds 0-99)

MCTS was retired after the n=4 run, so the 100-game eval covers the two
configs that survived. Corrected timings -- the earlier best-of-N figure
counted only Jev latency and not its own rollout compute:

| config      | score | +/- | ms/decision | $ / 100 games |
|-------------|-------|-----|-------------|---------------|
| JEV         | 13.8  | 0.18|     169     |    0.255      |
| JEV+BESTOF  | 14.4  | 0.18|     522     |    0.254      |

`JEV - JEV+BESTOF: -0.66 +/- 0.25` -> **distinguishable** (~2.6 sigma).

So search does buy something real, and best-of-N buys it for 522 ms rather
than MCTS's 6,619. But the margin is 4.8%, which is small enough to matter
for what can be distilled from it.

### Does more search help? No.

Best-of-N re-run with 48 rollouts instead of 16 -- three times the compute,
zero extra Jev calls -- against JEV on the same 60 seeds, paired:

| config           | score | +/- | ms/decision |
|------------------|-------|-----|-------------|
| JEV              | 13.7  | 0.19|     167     |
| JEV+BESTOF(16)   | 14.4  | 0.18|     522     |
| JEV+BESTOF(48)   | 14.4  | 0.24|   1,460     |

`paired JEV - JEV+BESTOF(48): -0.68 +/- 0.22 over 60 seeds (12W/24L/24T)`

Identical to the -0.66 measured at 16 rollouts. **Tripling the search budget
bought nothing.** The ceiling is the prior, not the amount of compute spent
on top of it.

The mechanism shows up during data collection: search changes the argmax on
only **~7% of states**. The prior already picks what the search would pick
93% of the time, so there is very little room for search to move the score,
and no amount of extra rollouts creates more.

`open space` and `greedy` are strategy-divergence telemetry, free from what
`analyse()` already computes. The prior-driven configs take the food-nearest
move far more often (72-80% vs 44%) and operate in tighter space, because
they are actually eating and growing; UNIFORM+MCTS wanders an empty board.

### Two traps this surfaced

**1. An under-shaped value function silently poisons the search.** The first
rollout value was food plus survival only. But ~45% of rollouts eat nothing,
so the value pinned to its survival floor and the between-move gap (0.085)
fell *below* the rollout noise (sd 0.14-0.19). Adding terminal
distance-to-food as a dense term moved separation to 0.107 against a
standard error of 0.022. This matters beyond phase 1: MCTS-improved
distillation targets are only as good as the value function, and an
under-shaped one would have quietly trained the student on noise.

**2. Best-of-N discards the prior when N covers the action space.** The
survivable action space here is 3 moves or fewer in **100%** of positions,
so "take the prior's top 3 and simulate" selects *every* move -- the prior
picks the candidate set, then rollouts rank them alone. That config is a
pure rollout policy wearing the prior's name, and it scored like one: **2.5**.
Keeping the prior in the ranking (`score = value + 0.25 * prior`, the thing
PUCT does at every visit) took it to **12.8** -- 97% of MCTS's score for 3%
of its wall clock.

## The student (phase 2)

Distil the prior into a local specialist. 28 scalars in (no board -- `analyse()`
already computes flood fill, dead ends and distances), 4 probabilities out,
6,276 parameters, MLX on CPU.

```bash
uv run python -m sysone.snake.collect --games 150      # ~16 min, $0.38
uv run python train.py                                 # ~2 s
uv run python -m sysone.snake.compare --games 100 \
    --configs jev bestof student_prior student_improved
```

100 games, seeds 0-99, all four configs on identical seeds:

| config            | score | +/- | ms/decision | $ / 100 games | wall  |
|-------------------|-------|-----|-------------|---------------|-------|
| JEV               | 14.0  | 0.17|    165      |    0.255      | 742 s |
| JEV+BESTOF(16)    | 14.4  | 0.18|    599      |    0.253      | —     |
| STUDENT(prior)    | 13.9  | 0.18|   **0.43**  |  **0.000**    | **2 s** |
| STUDENT(improved) | 14.1  | 0.17|     0.44    |    0.000      | 2 s   |

Paired, on matched seeds:

```
JEV            - STUDENT(prior)     +0.08 +/- 0.16   NOT distinguishable
JEV            - STUDENT(improved)  -0.05 +/- 0.16   NOT distinguishable
JEV            - JEV+BESTOF(16)     -0.41 +/- 0.18   distinguishable
JEV+BESTOF(16) - STUDENT(prior)     +0.49 +/- 0.20   distinguishable
STUDENT(prior) - STUDENT(improved)  -0.13 +/- 0.09   NOT distinguishable
```

**A 6,276-parameter MLP is statistically indistinguishable from hosted Jev on
this task**, at 384x lower latency and zero marginal cost. That is the whole
local-tradeoff argument in one row.

### What the student gives up

- **Robustness at the tail.** Same mean, worse worst case: the students boxed
  themselves in on 3 of 100 games. Jev never did. The mean hides this.
- **The action space.** Jev takes it as an *input* -- a different option set with
  different natural-language facts every tick. The student has a fixed 4-way
  head and a mask, and cannot generalize to a new action space at all.
- **Everything else.** Jev answers arbitrary typed questions about arbitrary
  state. The student answers one question about one game.

### The null, predicted before it was measured

Training the two students, before any gameplay eval:

| target                 | val KL | argmax | on decisive states |
|------------------------|--------|--------|--------------------|
| prior                  | 0.0267 | 92.2%  | 99.9%              |
| prior (unweighted)     | 0.0339 | 92.2%  | 99.9%              |
| improved               | 0.0644 | 88.4%  | 98.8%              |
| improved (unweighted)  | 0.0729 | 88.4%  | 98.6%              |

The two **teachers** disagreed on 7.7% of states. The two **students** disagree
on 1.6% -- distillation washed out ~79% of the teacher difference, predicting a
gameplay gap of 1.6/7.7 x 0.68 = 0.14 points. Measured: **0.13 +/- 0.09**.

So the honest result is not "search does not help" -- it does, by a
distinguishable 0.41. It is that **a 5% edge does not survive compression into
6k parameters.** The distillation channel is narrower than the signal.

Two things follow. Search-improved targets are measurably harder to fit (KL
0.064 vs 0.027) because they carry rollout noise the student cannot separate
from signal -- a calibrated prior is a *cleaner* distillation teacher than a
searched one. And entropy weighting helps exactly where predicted: better KL,
unchanged argmax.

## Economics

`uv run python bench.py`

- **268-token fixed envelope per call**, before any of your content. Small
  decisions are envelope-dominated.
- 8 questions in 1 call: **508 tokens, 158 ms**.
  The same 8 in 8 calls: **3,175 tokens, 1,159 ms** — 6.2x tokens, 7.3x time
  for identical information. Batch everything into one call.
- The 10x10 ASCII board costs 122 tokens and does earn them: same move, but
  confidence 0.83 with it versus 0.69 without.

## Step 2: going local

`LayaBackend` in `sysone/core.py` is the stub. `laya-mlx` implements the same
three question types against the same state-to-probabilities contract, so it
plugs into the existing seam.

Reported for laya-mlx on M3 Max: **7-13 ms p50**, under 1 GiB, 150-395 q/s
batched — against 160 ms here. Roughly 15x on latency, and the cost goes to
zero.

The constraint that decides which demo survives: **512-token total context**
(1024 on multilingual). Jev's 268-token envelope is TypeSafe-side framing, not
your content, so the comparison is against actual content:

- **Snake**: ~362 tokens of content. Fits. Straight swap.
- **Computer use**: a real accessibility tree does not fit. This is what
  tournament sampling is for — heats of 20 elements are individually small
  enough. Expect to shrink `BATCH` and run more rounds; more calls at 10 ms
  each is still far cheaper than one call at 160 ms.

Open the swap by installing `laya-mlx` and implementing `LayaBackend.decide`
to map `Choice`/`Noul`/`Score` onto `agent.predict(state, questions)`.
