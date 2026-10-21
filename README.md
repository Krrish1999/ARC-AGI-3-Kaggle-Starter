# ARC Prize 2026 — Local Dev Starter

**Go from zero to your first Kaggle submission in about 10 minutes, without
ever opening the Kaggle notebook editor.**

This is a starter kit for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
competition. You'll edit one Python file on your laptop, see it actually play
the real game environments locally, and push it to Kaggle as a submission with
a single command.

No Docker. No `submission.json` to hand-write. No copy-pasting between your
editor and a notebook.

## Sprint 2 local inference

The generated notebook now starts the retained 28-point notebook's Pennyroyal
v2.5.3 / SGLang 0.5.19 launcher, serving Qwen 3.8 AutoRound as `flashnext` on
`127.0.0.1:8001`. Kernel metadata attaches `dfranzen/pennyroyal-v253` and
`mirzamilanfarabi/intel-qwen3-8-flash-next-w4a16-autoround` plus
`mirzamilanfarabi/albucino-qwen3-8-flash-next-drafter` as datasets. Their metadata
identifies the same upstream target and draft models used by the baseline. It installs from
the attached wheelhouse into an isolated venv, leaving the framework interpreter
unchanged. The retained reference notebook is unchanged.
The launcher's Cell 7 wheel prewarm and Cell 13 model-prefetch start are retained
under `ENABLE_BASELINE_PREFETCH=1`; `0` selects the serial-read control. Prefetch
is best-effort I/O and does not replace any asset integrity/format check.

Installation, model loading and `/v1/models` readiness share a **2,400-second (40-minute)**
deadline before the gameplay runner starts. Failure stops the owned process
group and records exploration fallback. `ENABLE_OFFLINE_BACKEND=0` selects that
control explicitly; `ENABLE_SPECULATIVE_DRAFT=0` serves the same target without
the draft. Both default to `1`. The model endpoint is separate from
`gateway:8001`; Kaggle alone controls `KAGGLE_IS_COMPETITION_RERUN`.

Complete prompt histories pass through PrefixLock unchanged between playbook
resets. Structured model proposals still pass replay, sandbox and forward
prediction gates. Inference requests use the existing level budgets and a
30-second maximum socket timeout. Diagnostics include `llm_status.json`,
`llm_assets.json`, `llm_launcher.log`, `llm_requests.jsonl`, and `serve.log`.
Cache-hit rates are recorded only when the server supplies cached-token counts.
The >95% rate is a performance target, not a measured result here.

`ENABLE_ZERO_ACTION_REPLAN=1` allows up to three proposals after a rejected
mechanic or invalid action plan, using appended counterexamples and validation
feedback before any live dispatch. The proposals share one reasoning allowance;
inference and worker timeouts shrink with the remaining allowance. Setting it
to `0` restores the single-proposal path. Exact replay and `expect` checks stay
mandatory, and infrastructure failures remain visible. Repair attempts and
outcomes are recorded in each perception directory's `world_model.jsonl`.

Local regression checks cover startup and adapter failures. Actual GPU loading,
VRAM, throughput and cache reuse await an offline Kaggle rehearsal with these
mounts. The 2,400-second allowance supersedes the earlier 900-second timeout
observed on Kaggle. Failed startup remains visible, and the Gen-3 agent halts
before unverified gameplay when the model is unavailable.
Asset provenance and validation limits are recorded in
`research_lab/offline_inventory.json`.

---

## What you need before you start

- **Python 3.12** (the competition's `arc-agi` package requires it)
  - macOS: `brew install python@3.12`
  - Ubuntu: `sudo apt install python3.12 python3.12-venv`
  - Windows: install from [python.org](https://www.python.org/downloads/)
- **git** (to clone the official agent framework)
- **A Kaggle account** with the competition rules accepted
  ([accept here](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules))

That's it. No GPU required for the starter agent.

---

## Quick start

```bash
# 1.  Clone this repo and step in
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git
cd ARC-AGI-3-Kaggle-Starter

# 2.  Drop your Kaggle API token (kaggle.com → Settings → Create New Token)
#     into the project-local .kaggle/ folder (NOT your home directory)
mkdir -p .kaggle && echo "KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .kaggle/access_token
chmod 600 .kaggle/access_token

# 3.  One-time setup: venv, dependencies, framework
make setup

# 4.  Open agent/my_agent.py to see the random-action starter, then edit
#     it to make a better submission. This is the only file you change.

# 5.  Run it locally against every game in the competition (takes seconds)
make play-local

# 6.  Push it to Kaggle as a submission notebook
make submit

# 7.  Watch the run
make status

# 8.  "complete" finishes the save run; diagnostics/status.json will say
#     NOT_EVALUATED. Start a competition rerun of that saved notebook version.
#     Only the gateway creates submission.parquet during that rerun.
#     See the code-submission command below if the UI requires a saved parquet.
```

That's the entire loop. Steps 4–7 are what you'll repeat as you iterate;
step 8 is the deliberate moment when you spend a daily submission.

---

## The one file you edit: `agent/my_agent.py`

This is the only file you normally touch. It defines a class called `MyAgent`
with two methods:

```python
class MyAgent(Agent):
    def is_done(self, frames, latest_frame) -> bool:
        """Return True when your agent wants to stop playing."""
        ...

    def choose_action(self, frames, latest_frame) -> GameAction:
        """Look at the game state and return the next action."""
        ...
```

The starter version picks random actions — a baseline that proves your whole
pipeline works end-to-end. Replace the body of `choose_action` with your
strategy. Everything else (Kaggle plumbing, submission file format, game
orchestration) is handled for you.

---

## What happens when you run `make submit`

The competition is a *code* competition: you submit a notebook, Kaggle runs it
twice.

```diagram
   make submit
       │
       ▼
   ┌─────────────────────────────────────┐
   │  Kaggle Phase A: Save & Run All     │
   │  ─ Runs your notebook in their      │
   │    real environment                 │
   │  ─ Validates that your code         │
   │    executes without errors          │
   │  ─ make status shows "complete"     │
   └─────────────────┬───────────────────┘
                     │
                     │  You click "Submit to Competition"
                     │  on the kernel page
                     ▼
   ┌─────────────────────────────────────┐
   │  Kaggle Phase B: Competition Rerun  │
   │  ─ Your agent actually plays the    │
   │    hidden game set                  │
   │  ─ Your leaderboard score appears   │
   └─────────────────────────────────────┘
```

`make submit` builds and uploads the notebook (Phase A). After
`make status` reports `complete`, run `make evaluate VERSION=<saved-version>`
to request evaluation of that exact saved notebook through the competition gateway.
This uses a competition submission slot; it does not upload a local parquet.

Kaggle's service requires the output filename to exist in the saved run. Save runs
therefore emit a **zero-row, schema-only** `submission.parquet`, with diagnostics
explicitly marked `NOT_EVALUATED`. This contains no gameplay records or scores.
The earlier fabricated dummy row is not restored. Only the gateway writes evaluated
rows; validation rejects an empty or unchanged file during evaluation.
`ENABLE_SUBMISSION_BOOTSTRAP=0` disables this artifact for local diagnostics, but
then Kaggle cannot select the file for competition submission.

```bash
make submit                         # Upload a new saved notebook version
make status                         # Wait for its save run to complete
make evaluate VERSION=7             # Use the version returned by the upload
make output OUTPUT_DIR=kaggle-output/latest
```

Do not set `KAGGLE_IS_COMPETITION_RERUN` yourself: Kaggle must supply it and start
the gateway. The CLI helper uses `KAGGLE_API_TOKEN`, the environment's token stored
in `KAGGLE_KEY`, or `.kaggle/access_token`, with project-local configuration.
`make output` saves API logs and a file listing before downloading output files;
a proxy restriction on Kaggle's file host can block downloads while retaining
those API diagnostics.

### Debugging a short run or failed evaluation

The notebook prints an execution-mode banner before dependency installation.
`SAVE_OR_INTERACTIVE` followed by `NOT_EVALUATED` explains a short save run;
it does not indicate that hidden games completed. A competition rerun must log
`COMPETITION_RERUN`, gateway readiness, containment, and framework execution.
Save runs now also copy the framework, create the isolated interpreter, run
containment, import the deployed agent, and exercise one synthetic action through
its real framework loop. These checks contact no gateway and produce no scored
gameplay. `FRAMEWORK_RUNTIME_OK` confirms that the mounted Kaggle runtime passed;
a failure stops the save with its traceback before an evaluation is requested.
The launcher must also use the ARC SDK's `competition` operation mode: `online`
omits `competition_mode: true` from the scorecard-open request. The generated
framework configuration and its child process both enforce competition mode.

Download these files from `/kaggle/working/diagnostics/`:

| File | What it records |
|---|---|
| `status.json` | `NOT_EVALUATED`, `RUNNING`, `FAILED`, or `COMPLETE`, with the last stage and error |
| `notebook.log`, `events.jsonl` | Build ID, mode, stage timing, GPU probe, command exits and heartbeat |
| `pip.log`, `gateway.log`, `sandbox_setup.log` | Installation, gateway connection and isolated interpreter output |
| `framework_imports.log` | Imports and a synthetic lifecycle check in Kaggle's actual Python/SDK environment |
| `containment.log`, `containment.json` | Audit output and the current behavioral report |
| `framework.log` | Complete, unbuffered framework stdout/stderr, including child tracebacks |
| `traceback.log` | Notebook failure traceback and failing stage |

Agent recovery tracebacks are in `/kaggle/working/scratch/choose_action_errors.log`.
Failures are re-raised after logs are saved. `COMPLETE` requires a fresh,
readable, nonempty gateway parquet with the expected columns; missing, unchanged,
corrupt and legacy dummy results fail visibly. Existing result files are never
overwritten by the save-run bootstrap. A failed rerun cannot pass validation
using the empty save artifact.

`ENABLE_NOTEBOOK_DIAGNOSTICS=0` disables the optional GPU probe, heartbeat and
console mirroring for controlled comparisons; disk logs, error propagation,
containment, output validation and deadlines remain enforced.

The regenerated `notebooks/submission.ipynb` is now checked in alongside its
builder. Rebuild with `make notebook` after source edits, upload this file, and
save a new Kaggle version. Its build ID appears in metadata and the first log
banner so a stale uploaded notebook can be identified.

> **You only get 5 official submissions per day**, so it pays to be
> confident before you submit: get `make play-local` passing, then submit.

> **Heads up:** Before your first `make submit`, open
> [`notebooks/kernel-metadata.json`](notebooks/kernel-metadata.json) and
> replace `REPLACE_WITH_YOUR_USERNAME` with your Kaggle handle. The Makefile
> will refuse to push until you do.

### Choosing an accelerator

The notebook is generated with a **T4 GPU** by default (matches Kaggle's
sample submission). To change it, open
[`scripts/build_notebook.py`](scripts/build_notebook.py) and edit **one
line** near the top:

```python
ACCELERATOR = "t4"     # change "t4" to one of: cpu, t4, p100, rtx6000
```

Then re-run `make submit`. That's it — both the notebook metadata and
[`notebooks/kernel-metadata.json`](notebooks/kernel-metadata.json) get
updated automatically.

| Value | Hardware | When to use |
|---|---|---|
| `"cpu"` | No GPU | The random starter, or any non-ML agent |
| `"t4"` | Nvidia T4 ×2 | **Default.** Small models, fast iteration |
| `"p100"` | Nvidia P100 | Single big-memory GPU |
| `"rtx6000"` | Nvidia RTX 6000 (`g4-standard-48`) | Heavy ML; **ARC-AGI-3 exclusive**, burns GPU quota faster |

RTX 6000 is reserved for ARC-AGI-3 notebooks only — don't use it for early
iteration. All accelerated Kaggle sessions have internet disabled, which is
already the default in this kit.

---

## All the commands

| Command | What it does |
|---|---|
| `make setup` | One-time install: Python venv, `arc-agi`, `kaggle` CLI, clones the framework |
| `make play-local` | Runs your agent against every game in the dataset, locally |
| `make play-local GAME=ls20` | Same, but only one game (faster while debugging) |
| `make verify-local` | 30-second smoke test on two games |
| `make list-games` | Print every game id available |
| `make pull-sample` | Download the official sample agent for reference |
| `make notebook` | Build the Kaggle notebook from your agent (no push) |
| `make submit` | Build the notebook **and** push it to Kaggle |
| `make status` | Check the status of your most recent Kaggle run |
| `make evaluate VERSION=N` | Submit saved notebook version N for gateway evaluation |
| `make output OUTPUT_DIR=path` | Retrieve latest notebook logs and output files |
| `make clean` | Remove the venv and downloads; keep the checked-in notebook |

---

## Why this setup, instead of editing in the Kaggle notebook?

Three reasons:

1. **Iteration speed.** Editing in your normal IDE, then `make play-local`,
   gives you a real-game-engine feedback loop in seconds. The Kaggle editor's
   loop is *minutes* per change.
2. **No environment surprises.** The local `arc-agi` PyPI package hosts the
   same game engine the Kaggle gateway runs. If it works locally, it works on
   Kaggle.
3. **Your code stays in git.** Notebooks are awful for diffs and code review.
   Here your real work lives in [`agent/my_agent.py`](agent/my_agent.py); the
   notebook is just an auto-generated deployment artifact.

---

## Project layout

```
.
├── agent/
│   └── my_agent.py             ★ The file you edit
├── scripts/
│   ├── play_local.py           Runs your agent against real games
│   ├── build_notebook.py       Packages your agent into a Kaggle notebook
│   └── slim_framework.py       Trims framework deps so install is light
├── notebooks/
│   ├── kernel-metadata.json    Edit once: your Kaggle username
│   └── submission.ipynb        Auto-generated, never edit by hand
├── vendor/                     Cloned framework (gitignored)
├── .venv/                      Python 3.12 venv (gitignored)
├── .kaggle/                    Your project-local Kaggle token (gitignored)
└── Makefile
```

---

## Troubleshooting

**`make setup` fails: `python3.12: command not found`**
Install Python 3.12 — the `arc-agi` package requires it. macOS:
`brew install python@3.12`.

**`make submit` says "edit kernel-metadata.json"**
You haven't replaced `REPLACE_WITH_YOUR_USERNAME` in
[`notebooks/kernel-metadata.json`](notebooks/kernel-metadata.json) yet.

**`make submit` says `401 Unauthorized`**
Your Kaggle token is missing or invalid. Generate a fresh one from your
[Kaggle Settings page](https://www.kaggle.com/settings) and overwrite
`.kaggle/access_token`.

**`make play-local` says "Could not create environment"**
Your machine couldn't reach the ARC-AGI API to download the game source on
first run. Check your internet, then try again — once downloaded, games are
cached in `environment_files/` and you're fully offline.

**My local score is 0.0**
That's expected for the random starter agent. Your job is to make it
non-zero. 🙂

---

## Where to go next

- Read the [ARC-AGI-3 docs](https://docs.arcprize.org/) to understand the
  benchmark.
- `make pull-sample` to study Kaggle's reference agent (the same one
  currently sitting on the leaderboard).
- The competition's [discussion forum](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion)
  for community Q&A.

Good luck. Looking forward to seeing what you build.
