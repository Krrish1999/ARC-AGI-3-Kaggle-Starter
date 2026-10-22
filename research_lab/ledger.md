# ARC-AGI-3 PhD Research Laboratory Ledger

## [EDIT] Bounded shard staging — systems ticket, 2026-10-22

The user explicitly answered **Develop bounded shard staging**, replacing the
all-shards-before-launch constraint with incremental copy/load of the **same
Qwen weights**. Continue PR #14 from engineering parent `b1afa97`; retain the
archived agent parent and all Gen-3 policy/security sources. Apply `phd-systems`.
Version 8's final startup outcome remains unconfirmed; no new diagnostics are
on main at this preflight. Its measured 20.94 GB working capacity motivates
this change independently of the unresolved fallback outcome.

**[EDIT] hypothesis.** Default-on `ENABLE_BOUNDED_MODEL_STAGING=1` prepares
verified local metadata and a manifest of indexed source shards. Integrate
with the exact pinned Pennyroyal loader (upstream `d00d88efc8d6`) through a
hash-checked, narrowly scoped adapter. Block each loader read until its shard
copy closes, fsyncs and matches the source-stream digest. Pipeline one next
shard with one current shard, then unlink the consumed file. Clone yielded
CPU tensors so surviving model-loader references cannot retain an evicted
file's mmap. Serialize target/draft iterators with an interprocess lock to
enforce the aggregate disk bound; reject unsupported loader modes or source
drift. Disable whole-checkpoint page-cache prefetch on this path. Toggle 0
retains the full-copy control. Keep the same checkpoints, quantization,
speculation, RadixAttention, 2,400-second startup and global deadlines.

**Acceptance.** Exercise actual safetensors across all 38+2 tiny fixture
shards, exact tensor equality, bounded physical storage, blocking read barrier,
copy/hash/size failures, cancellation, concurrent target/draft and restart
ownership. Test the generated model/draft handoff and pinned loader adapter;
fail closed on altered upstream source. Run pytest, ruff, pyright and
containment, rebuild and update the reviewable PR. Do not claim GPU readiness
or RHAE from synthetic fixtures. Record runtime evidence separately if another
Phase A rehearsal is possible within the existing deployment authorization.

### Bounded candidate verification

**282/282 pytest cases** pass, including 21 new bounded-staging cases with
real CPU torch/safetensors, all 38 target + 2 draft fixture shards, retained
tensor equality after eviction, physical-block bounds, cancellation, copy
barriers/failures and the exact pinned upstream default-loader methods.
Generated model arguments and the retained draft-view helper point to local
metadata without source-weight symlinks or placeholder shards. Both target and
draft use the adapter even when draft multithread configuration is omitted.
The native loader file hash is checked before any patch; unsupported modes
and altered source fail closed. Rerun of full pytest after rebuilding the
initially stale notebook passes; ruff and pyright pass, containment **20/20**
with `containment_passed=true`. One lint repair retained `safe_open.keys()`
with an API-specific suppression. No agent module changed.

Cache contents are bounded to two indexed shards (10,741,854,576 bytes at the
observed maximum shard size) plus metadata and 4 GiB of free headroom. The
underlying copy helper now refuses to write beyond the inventoried byte count,
including a growing source. Each yielded tensor is cloned on CPU to detach
its file mmap; the original quantization and model/draft weights are retained.
Startup diagnostics now persist elapsed startup time and emit a 30-second
wait heartbeat under the unchanged 2,400-second deadline. Real CPU tests used
torch 2.14.1+cpu / safetensors 0.8.0; the Kaggle serving wheels remain pinned
at torch 2.13.0+cu130 / safetensors 0.9.0rc1. Local success does not establish
GPU readiness, startup latency or RHAE lift.

The reviewed notebook build is **f68cb1c9e817f9fe**. Deployment receipt:
`research_lab/traces/bounded_staging_deployment.json`. Version 8 remains
COMPLETE with final startup diagnostics unavailable; this is a distinct,
explicitly authorized bounded-loading experiment. No Phase B request is made.

### Version 9 — loader source mismatch, zero weight reads

`make submit` saved version 9 from `0a6316c`, build **f68cb1c9e817f9fe**.
All deployed cells match, dataset membership matches (Kaggle reordered the
array), GPU-on/internet-off accepted. Local metadata and the owned working
cache passed preflight on ext4 with 20,937,408,512 bytes free. Pennyroyal
installed and CUDA initialized, but the installed loader's full-file hash
differs from the Git commit source. The adapter correctly refused to patch;
**SGLang never launched and zero weight shards were read**. The owned metadata
transaction was removed. Startup failed after **131.064 seconds**; readiness
latency remains null. COMPLETE save observed **2026-10-22 20:30:38.939
Asia/Calcutta**, separately from the startup failure. Full console evidence is
captured in `research_lab/traces/bounded_staging_runtime_evidence.json`.

**[EDIT] artifact-binding repair.** Bind the installed loader to the exact
approved wheel's SHA256 **d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225**,
already observed in version-7 input diagnostics, and require installed bytes
to equal that wheel's loader member before patching. Keep the upstream hash
as a reference rather than assume packaged source equality. Retain AST-scoped
method/boundary checks, record wheel/installed/patched hashes and extract the
actual method contract for auditing. A changed archive or installed file must
still reject before loading. Normal Kaggle wheel download is blocked at
storage.googleapis.com; perform archive verification inside the authorized
offline Kaggle bootstrap instead. This is an exact-artifact binding, not an
unchecked acceptance of the mismatching installed source. Test corrupt wheel
and installed-file drift before rebuilding and one further startup rehearsal.

**Repair verification.** **284/284 pytest cases** pass, including two new
archive/installed-member tampering cases. Ruff and pyright pass, containment
**20/20**. Rebuilt notebook **75c1403830052f18** matches its sources. Archive
verification happens before writing support modules or patched loader code;
the exact installed method contract is saved as `diagnostics/loader_contract.json`.
No checkpoint, agent policy, serving setting or deadline changes. Separate
receipt: `research_lab/traces/artifact_bounded_staging_deployment.json`.

### Version 10 — artifact binding passes; iterator keyword mismatch

Version 10 / **75c1403830052f18**, source `2757785`, matches the pulled
notebook and dataset membership. The approved archive digest matches;
installed loader SHA256 is **d12e619c9a334f1af61d79d1958c59554e0931986da4942b1f570562d0752bb1**,
distinct from the upstream reference. The metadata transaction, installation,
CUDA and SGLang process launch succeed. SGLang reaches model initialization,
then its packaged iterator supplies **prefetch_lookahead**, absent from the
upstream fixture signature. The adapter raises TypeError before any weight
copy. Startup fails after **170.084 seconds**, zero loaded/copied shards;
readiness remains false. COMPLETE save observed **2026-10-22 20:40:15.117
Asia/Calcutta**. Owned working metadata is cleaned. Full live evidence is in
`research_lab/traces/artifact_bounded_staging_runtime_evidence.json`.

**[EDIT] iterator-call repair.** AST-validate the single native safetensors
call, its weight-list argument and the reviewed optional argument names,
including packaged `prefetch_lookahead`. Replace the entire call with the
bounded iterator's weight list only: this adapter owns copy/mmap/prefetch
settings instead of inheriting upstream optional knobs. Unknown arguments
still reject before patching. Test actual tensor loading for both reference
and lookahead-call variants and rejection of an unreviewed keyword. The wheel
digest/installed-member checks and all bounds remain unchanged. The initial
test-only synthetic call syntax was corrected before final verification.

**Iterator repair verification.** **286/286 pytest cases**, required ruff and
pyright pass; containment **20/20**. The rebuilt notebook is
**79f942481dc996e6** and matches source. The two added cases exercise the
packaged lookahead argument and reject an unknown argument; no permissive
`**kwargs` fallback is introduced. Receipt:
`research_lab/traces/iterator_bounded_staging_deployment.json`.

### [EDIT] Bounded concurrent range reads — performance ticket

Version 11 / **79f942481dc996e6**, source `76ff46a`, is still running.
Its first 2,899,311,784-byte shard is copied/synced/digest-verified in
**97.714 seconds** and consumed successfully. The observed sequential-source
rate is about 29.7 MB/s; applying that rate to all 185.4 GB would exceed the
2,400-second deadline. This projection motivates a concurrency experiment;
it is not a measured final startup result.

Continue from engineering parent `76ff46a`, preserving agent policy and
archived parent. **[EDIT]** default-on `ENABLE_PARALLEL_SHARD_READS=1` adds four
bounded concurrent positional reads inside the single active shard copier.
Write and hash chunks in original order, retain the same two-file disk bound,
and cap read buffers to a small fixed window. Toggle 0 keeps the measured
serial-copy control. Test forced out-of-order completion, partial reads,
truncation, fsync failure, ordered digest equality and in-flight buffer bounds
before another deployment. The active version-11 control continues while
this independent repair is developed; do not overwrite its outcome or claim
that a projection establishes measured deadline failure.

**Control progress.** Three target shards were consumed, including 5.37 GB
copies taking **184.348** and **186.609 seconds**. Maximum logical cache
accounting is **10,741,458,608 bytes**, within the two-slot limit. The serial
control remains RUNNING; its final startup duration/readiness is unknown.
Partial build-bound evidence is preserved in
`research_lab/traces/iterator_bounded_staging_runtime_evidence.json` before
superseding the latest notebook with the concurrent-read trial.

**Range candidate verification.** **292/292 pytest cases** pass, including
six new parallel-read cases: reordered completion with 2..4 concurrent reads,
partial-read retry, truncation/growth/fsync failure, and the serial ablation.
Ruff, pyright and containment **20/20** pass. Four positional reads fill fixed
8 MiB buffers; a four-item window hashes/writes original byte order without
reading a whole shard into RAM. At most 40 MiB of shard buffers are live;
the two-file disk bound and 4 GiB headroom are unchanged. Notebook
**c5e21ca4933c5e51** matches source. Receipt:
`research_lab/traces/range_bounded_staging_deployment.json`. The active control
and the new rehearsal each retain the 2,400-second startup deadline; neither
is a Phase B evaluation or a measured RHAE result.

### [EDIT] Three-shard sequential-copy window — systems ticket

Engineering parent `09a84e3`; retain all checkpoint and policy identities.
Version 12 remains RUNNING at 390 startup seconds with no completed first
copy, whereas the serial control completed it near 267 startup seconds.
Within-file positional reads have not shown a throughput benefit on this
FUSE mount; no final failure duration is claimed. Preserve that candidate
behind `ENABLE_PARALLEL_SHARD_READS` with default 0.

Default-on `ENABLE_THREE_SHARD_WINDOW=1` instead overlaps three independent
sequential shard copies. Inventory checks require **3 * 5,370,927,288 +
64,194,294 metadata + 4,294,967,296 headroom = 20,471,943,454 bytes**, below
the observed 20,937,408,512-byte free working volume. Reject before launch if
the next run cannot meet this bound. Replenish a slot only after consumption,
file close and unlink; keep target/draft serialization and CPU tensor clones.
Toggle 0 keeps the measured two-slot pipeline. Test 2- and 3-slot pipelines,
simultaneous independent copies, cancellation and physical high-water bounds;
run full gates and one final bounded startup rehearsal. No timeout increase
or smaller/different checkpoint is authorized.

**Three-window verification.** **295/295 pytest cases** pass, including
both 2/3-slot full 38+2-shard tensor fixtures, both cancellation paths, and a
barrier proving three distinct source copies overlap. Physical cache block
bounds and unlink-before-replenish are verified. Ruff, pyright and containment
**20/20** pass. Notebook **fa9152ca54532566** matches source. Version 12 still
has no completed shard after 570 startup seconds; record its final outcome
as unknown when superseded, rather than calling this an observed timeout.
Receipt: `research_lab/traces/three_shard_staging_deployment.json`.

## [EDIT] Working-volume staging and cleanup-error isolation — 2026-10-21

Requested skill: `phd-systems`; continue PR #14 / `feat/local-model-staging`.
Engineering parent `31ffd70`, merged with the user's main commit `8908646`
containing `output_Version_7/`. Retain the archived parent and unchanged agent
policy. The user supplied this directory to finish the authorized startup
verification; the original I/O deployment remains unvalidated.

**Version 7 diagnosis.** All 39 events match the timestamped notebook log and
build **a56668207f023cce** / launcher hash. Staging failed after **860.627
seconds** with ENOSPC at `_stage_file` fsync, despite the 202.58 GB extent
allocation. Cleanup then raised EROFS from `shutil.rmtree`, replacing the main
exception and aborting the candidate loop. Only `/tmp/models/arc3` was tried;
no checkpoint was published and SGLang never launched. Startup ended after
**935.044 seconds** (time to failure), with `ready=false`, `ever_ready=false`.
`NOT_EVALUATED` at 956.981 seconds and containment/framework 20/20 remain valid.
The actual writable limit and successful copied-byte count were not recorded.
Allocation counters do not prove Kaggle will accept subsequent writes.

**[EDIT] deterministic repair.** Keep `ENABLE_STAGING_CAPACITY_RESERVATION=1`
and `ENABLE_LOCAL_MODEL_STAGING=1` controls. Prefer the previously untried
`/kaggle/working/.arc3-models` candidate, then the prior `/tmp` and temp
candidates. Capture deletion failures separately from ENOSPC, retain owned
orphan paths for guarded post-stop cleanup, and continue candidate selection.
Record read-only mount flags and successful copied-file/byte progress while
still withholding MODEL_DIR binding until both complete checkpoints validate.
The owned output cleanup, startup limit 2,400 seconds, global budgets and
strict Gen-3 policy remain unchanged. No claim of sufficient writable working
storage is made before the next full-copy rehearsal.

**Acceptance.** Inject ENOSPC at copy plus EROFS at cleanup and require actual
copy success on an independent fallback, preserving both errors and unrelated
files. Verify orphan cleanup ownership and continuation, read-only rejection
and progress before publication. Run full regression, ruff/pyright, containment,
rebuild and one bounded Phase A rehearsal. If full copying still fails, record
that storage failure rather than enlarging the timeout or claiming readiness.

### Working-volume candidate verification

**261/261 pytest cases** pass, including five new cleanup/progress/read-only
regressions, with required ruff, pyright and containment **20/20**. The notebook
matches its sources and the Gen-3 entry-module hashes remain unchanged. Build
**122eab780b108df8** keeps the shared 2,400-second startup limit. The working
path accepted the final version-7 diagnostic writes after the `/tmp` error;
this supports trying it first but does not establish model-sized capacity.
Receipt: `research_lab/traces/working_staging_deployment.json`.

### Version 8 rehearsal — COMPLETE save, working volume too small

`make submit` saved **version 8**, build **122eab780b108df8**, from `a9b0578`.
Initial and final Kaggle pulls match every cell source and all three datasets;
RTX PRO 6000, GPU-on/internet-off, containment/framework preflight pass.
`make status` first reports COMPLETE at **2026-10-21 19:55:57.526
Asia/Calcutta**. The notebook-start-to-observed-completion interval is
**1334.022 seconds**, including save/platform overhead;
it is not model readiness latency.

The working candidate is a distinct writable **ext4** mount, device 1794,
but reports only **20,937,515,008 bytes (20.94 GB)** free. It correctly rejects
**185,400,295,230 bytes** of checkpoint plus **17,179,869,184 bytes** of
headroom before copying. Fallback `/tmp/models/arc3` is the root overlay,
device 64; it preallocates 202,580,451,328 bytes and the last observed progress
is **20 synced files / 73,581,282,253 bytes**, still within the target copy.
No checkpoint-complete, server-launch, readiness or final failure event was
captured. The stream ends there and the completed-log API again returns `[]`.
Version-8 diagnostics are not yet present on main; keep its final copy/startup
outcome **unconfirmed**, separate from the established version-7 failure.

Evidence: `research_lab/traces/working_staging_runtime_evidence.json` and the
hash-bound deployment receipt. The repaired loop/progress tests pass locally;
the available console establishes correct rejection of the undersized working
volume, not a ready model. PR #14 remains draft. No Phase B evaluation was
requested or submitted.

**Constraint decision resolved.** The user subsequently answered **Develop
bounded shard staging**. The bounded-loading ticket above now governs that
implementation. Preserve version 8's unknown final outcome and all checkpoint
identities; no timeout increase is authorized.

## [EDIT] Capacity-backed local checkpoint boot — 2026-10-21

Requested skill: `phd-systems`; continue `feat/local-model-staging` / PR #14.
Engineering parent is `9084757` (the version-6 diagnostic closeout), merged with
main `fb1fbb9` (PR #13). Archived parents and the Gen-3 agent/security sources
remain unchanged. The user explicitly authorizes another build, Phase A save,
startup measurement and PR update. Version 6's ENOSPC supersedes the earlier
assumption that a filesystem free-space report establishes checkpoint capacity.

**Type C hypothesis / [EDIT].** Prefer `/tmp/models/arc3`, with configured
alternatives `/kaggle/working/.arc3-models` and `/kaggle/temp/arc3-models`.
Default-on `ENABLE_STAGING_CAPACITY_RESERVATION=1` uses `posix_fallocate`, fsync
and allocated-block verification on every destination checkpoint file plus
16 GiB of serving headroom **before reading any weight shard**. Record mount
type, device and free inodes. Reject memory/network filesystems; on capacity
failure clean only the owned transaction and try the next local candidate.
Copy into reserved files with `r+b`, preserving allocations. Release the owned
headroom file only after both copies complete, for the subsequent installation.
After the owned server process group stops, remove working-directory model
copies only when their transaction and marker match this supervisor's owner
token, keeping large checkpoints out of saved notebook outputs. Other paths,
symlinks and another run's files are never eligible for this cleanup.
Argument binding still follows complete close/fsync and metadata verification.
No enabled-path fallback to FUSE input or timeout increase is allowed.

**Ablation / acceptance.** Toggle 0 retains the prior free-space-only algorithm
at the preferred root. Test actual small-file allocations and copies, exhausted
or unsupported allocation despite high reported free space, fsync ENOSPC,
fallback/owned cleanup, late copy failure, a blocking final allocation,
sparse-allocation rejection, mount/inode rejection and generated handoff.
Run pytest, ruff, pyright, containment and exact notebook checks, then deploy
and record the observed capacity/copy/startup outcome. Allocation checks improve
capacity evidence; no candidate volume is assumed large enough before the
Kaggle run, and successful local fixtures are not GPU readiness or RHAE proof.

### Capacity candidate verification and deployment preparation

Final verification: **256/256 pytest cases**, including 19 additional capacity,
fallback and cleanup cases; required ruff and pyright pass; containment
**20/20**, `containment_passed=true`. One import-order lint repair passed on the
next check. The notebook exactly matches its sources; both agent entry-module
hashes match version 6. Build **a56668207f023cce** preserves the 2,400-second
startup and global deadlines. Receipt:
`research_lab/traces/capacity_staging_deployment.json`.

`make submit` successfully pushed saved **version 7** from `5699882`.
The pulled notebook matches every source cell and all three datasets, with GPU
on, internet off and `NvidiaRtxPro6000`. Kaggle containment/framework preflight
passes. At 2026-10-21 18:19:52 Asia/Calcutta, the local `/tmp/models/arc3` overlay
reports **202,580,451,328 bytes allocated**, **62 files reserved**, and **0.413
seconds** reservation time. Copying began into retained extents; completion and
model readiness are still pending. This establishes allocation evidence on
that candidate, not successful checkpoint contents or model loading.

The user will upload the new diagnostic folder again if the Kaggle log API
remains empty. Monitor Phase A to a terminal status; do not claim model
readiness from COMPLETE alone or reuse version-6 failure timings for this build.

### Version 7 Phase A closeout — COMPLETE save, staging failure confirmed

`make status` first reports **COMPLETE at 2026-10-21 18:41:08.815
Asia/Calcutta** (preceding RUNNING poll 18:40:22.791). A final source/metadata
pull still matches **a56668207f023cce**, all three datasets, GPU enabled,
internet disabled and NvidiaRtxPro6000. The interval from notebook start to
observed completion is **1373.083 seconds**;
this includes save/platform overhead and is not a measured model-startup time.

The initial live log ended during copying and the completed log API returned
`[]`. The user's main commit `8908646` supplies the complete diagnostics in
`output_Version_7/diagnostics`. All 39 events agree with the timestamped console
and deployed build/launcher. The staging attempt failed after **860.627 seconds**:
ENOSPC at copy fsync, followed by EROFS during deletion. Cleanup aborted the
fallback loop after only one candidate. The startup attempt failed after
**935.044 seconds**, ending at **2026-10-21 18:34:12.504 Asia/Calcutta**.
`ready=false`, `ever_ready=false`; no checkpoint was published and SGLang
never launched. NOT_EVALUATED follows at 956.981 seconds. Containment and
framework preflight passed. These are measured failure durations, not model
readiness latency. No Phase B evaluation or RHAE/cache claim is made; PR #14
remains draft while the working-volume repair is evaluated.

Evidence: `research_lab/traces/capacity_staging_runtime_evidence.json` contains
this run's complete events, final staging/launcher/status reports, diagnostic
file hashes and status polls; the deployment receipt binds them to source
commit `5699882`. That candidate passed 256 local tests, lint/type and
containment; the GPU rehearsal falsified its storage/candidate-loop acceptance.
The separate repair ticket above is not yet measured on Kaggle.

## [EDIT] Blocking local checkpoint staging — systems ticket, 2026-10-21

Requested skill: `phd-systems`; branch `feat/local-model-staging`. Engineering
parent is `d509fa2` (the verified Gen-3 + 2,400-second deployment from PR #13),
with main at `87fbb24`. Preserve the archived
`research_lab/top_programs/v1_retrodict_world_model_harness.py` and all agent
policy/security sources. No statistically promoted integrated champion exists.
The user's deployment instruction authorizes this I/O change and Phase A rerun.

**History / failure triage (Type C).** Gen-3 fixes replay, expectation checking
and memory retention, but cannot acquire skill without a ready local model.
Version 5 completed its save after a 2,400-second model-startup timeout: 25/38
shards loaded, no ready endpoint, no gameplay or measured RHAE. Inspection now
confirms that the retained `precache()` only reads files into page cache; it
never copies models. Its thread runs concurrently with SGLang, whose target and
draft paths still resolve to the FUSE dataset mounts. Earlier timing alone did
not separate FUSE reads from CPU transformations. The deterministic mutation
is a real local-copy barrier, not another timeout increase or a policy change.

**[EDIT], `ENABLE_LOCAL_MODEL_STAGING=1` (default).** The builder explicitly
configures `/kaggle/temp/arc3-models`. Copy all checkpoint/config/tokenizer files
for both target and selected draft with four bounded-memory I/O workers; wait
for every copy to finish, close and fsync, then verify local index/config/shard
metadata before assembling server arguments. Target `--model-path`, draft
links and tokenizer/template reads must all use the completed local trees.
Insufficient space, bad paths or copy failures must prevent server launch,
never silently substitute FUSE input. Staging-off is an explicit reference
control; the previous page-cache thread is active only in that control. Wheel
prewarm, startup/global deadlines, serving flags and Gen-3 gates are retained.

**Acceptance.** Real copy fixtures with 38 shards and draft/metadata, blocking
worker completion, copied symlink materialization, failure cleanup, insufficient
space, unsafe source paths, actual generated --model-path/draft-view binding,
and staging-off reference. Run full pytest, required ruff/pyright, containment
and exact notebook/source checks. Deploy using `make submit`, monitor Phase A,
and record staging/loader/readiness times separately. A ready endpoint is not
an inference/cache/RHAE measurement. No Phase B submission is requested.

### Local staging deployment — version 6 COMPLETE save, local-copy ENOSPC

Local verification passes **237/237 pytest cases**, including 13 new staging
cases, required ruff/pyright and containment 20/20. The agent sources are
unchanged. `python scripts/build_notebook.py` and `make submit` packaged
**8e291c0a90d1533a** and successfully pushed saved **version 6**. A CLI pull
matches every generated cell source. Kaggle-side containment/framework
preflight passes and the local-copy event reports **185,400,295,230 bytes**
required with **1,145,972,953,088 bytes** free under
`/kaggle/temp/arc3-models`; the copy-start observation uses four workers.

`make status` first reports **COMPLETE at 2026-10-21 17:38:20 Asia/Calcutta**;
the preceding poll was RUNNING at 17:37:34. The notebook began at 17:26:59.671,
so the interval to observed completion is **680.412 seconds**, including all
preflight/save work. This is **not a measured model-startup time**. The initial
live console fragment ended at copy start; the uploaded diagnostics below
resolve that uncertainty. Every final pulled cell still matches build
**8e291c0a90d1533a**.

**Uploaded diagnostics resolve the outcome: readiness FAIL.** The initial
completed-log APIs returned `[]` and `kernels output` was blocked at
`www.kaggleusercontent.com`. The user supplied all 11 version-6 diagnostic
files. Their build ID and launcher hash match this deployment, and every
structured event matches the timestamped notebook log. Retain SHA256/size
receipts for all uploaded files and the complete console events.

Local staging failed after **270.003 seconds** with
`OSError: [Errno 28] No space left on device`, raised by
`os.fsync(writer.fileno())` in a copy worker. `models={}`: no checkpoint was
published. Because the blocking barrier raised before argument assembly,
**SGLang was never launched**. `llm_status.json` records `ready=false`,
`ever_ready=false`, `RuntimeError: launcher_exited:1`. The measured
`llm_startup` attempt lasted **278.679 seconds**, ending at
**2026-10-21 17:31:59.387 Asia/Calcutta**. This is time to failure, not model
readiness latency. The notebook subsequently emits `NOT_EVALUATED` at 299.865
seconds and writes zero gameplay rows. Its later COMPLETE save status does
not establish a working model.

The target's **38 indexed shards total 181,196,006,784 bytes**; the draft's
**2 total 4,140,094,152 bytes**. The complete staging requirement including
ancillary files is **185,400,295,230 bytes (172.67 GiB)**. Only 64.2 MB lies
outside the indexed shards; required weights dominate the footprint. The
**1.15 TB reported free** did
not establish effective writable capacity; the filesystem free-space precheck
was insufficient. These logs do not identify the actual mount quota, backing
store limit or inode capacity, so do not assert a specific Kaggle quota or
assume `/tmp` has more space. Wheel prewarm completed (211 files, 6.70 GB,
4.16 seconds); dependency installation, containment **20/20**, and framework
preflight passed. No 2,400-second timeout or CUDA/weight-loading failure was
reached in this run.

Receipts: `research_lab/traces/local_staging_deployment.json` and the captured
events/status polls in `research_lab/traces/local_staging_runtime_evidence.json`.
The 2,400-second deadline and Gen-3 policy remain unchanged. No Phase B
competition evaluation was submitted; no cache, inference or RHAE acceptance
is claimed. PR #14 remains a draft pending capacity correction, with this
failure in its description. This closeout changes evidence only, not the tested
runtime sources.

**Next systems ticket (not yet deployed).** Measure the mount/device, writable
capacity, inode availability and any exposed quotas for candidate local storage
locations, then verify allocation for the full checkpoint footprint plus
serving-environment headroom before committing to another full copy. Preserve
the blocking barrier and copy-failure cleanup. Changing only the path or
removing `fsync` would not establish capacity. If no volume can hold the model,
full-checkpoint local staging needs a different storage/loading design; another
timeout increase cannot repair this failure.

## [EDIT] Gen-3 Kaggle deployment — 2,400-second startup, 2026-10-21

The user confirms PR #12 merged into main `87fbb24` and explicitly authorizes
this Phase A deployment and a revised **2,400-second (40-minute)** startup
allowance. This supersedes the earlier instruction to preserve 900 seconds.
The Gen-3 agent sources and security gates remain unchanged. Branch:
`feat/gen3-kaggle-2400s`; retained architectural parent remains
`research_lab/top_programs/v1_retrodict_world_model_harness.py`, followed by the
merged Gen-3 authorization refactor. No new score promotion is inferred.

Pre-deployment audit found the builder, runtime default, supervisor default and
maximum, and manifest all still using 900 seconds. Update the complete path to
2,400, preserving the shared monotonic deadline, remaining global run budget,
model readiness gate, offline dataset mounts and owned-process cleanup. Required
metadata datasets are present: `dfranzen/pennyroyal-v253`,
`mirzamilanfarabi/intel-qwen3-8-flash-next-w4a16-autoround`, and
`mirzamilanfarabi/albucino-qwen3-8-flash-next-drafter`. RTX PRO 6000 is selected;
GPU is enabled and internet disabled.

The save preflight also expected a synthetic non-reset action with no backend,
which Gen-3 correctly refuses. Its toy environment now begins at NOT_PLAYED and
asserts a single lifecycle RESET, preserving strict authorization while testing
the actual framework dispatch path. No real game or gateway is contacted by
this fixture. Run deadline/readiness fixtures, exact notebook synchronization,
full regressions, lint/type checks and containment before `make submit`.

Phase A is a saved notebook/startup validation. Hidden RHAE is measured by a
separate competition rerun (Phase B); do not report a save as a hidden score.
Deployment preparation passed **224/224 pytest cases**, required ruff and
pyright, exact source/build synchronization, and containment **20/20** with
`containment_passed=true`. Readiness at 1,800 simulated seconds succeeds; the
failed-start fixture exhausts exactly 2,400 seconds. The embedded save-preflight
fixture executes the actual strict Gen-3/framework lifecycle action.

`python scripts/build_notebook.py` packaged build **4086c46befdb2235**;
`make submit` rebuilt it identically and successfully pushed **saved version 5**
to `aanchal009/arc-agi-3-runtime-check`. `make status` reports **RUNNING**.
A normal CLI source/metadata pull matches every local cell source and confirms
all three accepted datasets, GPU enabled, internet disabled. Version-suffixed
CLI pull/log requests returned API errors; the supported latest-version paths
work and are bound to this build by the source comparison. Phase A monitoring is complete; see the final outcome below. Kaggle-side GPU,
containment and strict framework preflight pass; LLM startup began at
2026-10-21 16:07:16 Asia/Calcutta. Receipt:
`research_lab/traces/gen3_deployment_2400.json`. No Phase B score is claimed.

A fresh official submission-list read finds **7 entries**, including an existing
COMPLETE result **57474828**, dated 2026-10-21 09:05:10.793 UTC, with
**publicScore=0.14** and no released privateScore. It predates this Gen-3
version-5 deployment and cannot measure its lift. This updates the account's
latest-result evidence while retaining the prior 0.12 receipt historically;
no model-readiness or statistically significant improvement is inferred.

### Version 5 Phase A closeout — COMPLETE save, model readiness FAIL

`make status` now reports `KernelWorkerStatus.COMPLETE`, observed at
**2026-10-21 16:47:45 Asia/Calcutta**. Final source/metadata pulled from Kaggle
still matches build **4086c46befdb2235**, with every generated cell source
identical to the local artifact. The supervisor used the requested 2,400-second
allowance and cleanup: `llm_startup` lasted **2400.129 seconds** and emitted
`TimeoutError: startup_deadline_2400s` at 16:47:16 Asia/Calcutta. It never emitted
`llm_ready`. The owned launcher was stopped with returncode -15.

The RTX PRO 6000 Blackwell (97,887 MiB) and offline dependencies are usable.
Wheel prewarm read 211 files (6.70 GB) in 67.00 seconds. Server logs identify
SGLang 0.5.19+gd00d88efc8d6 and Torch 2.13.0+cu130; weight loading began at
10:39:40 UTC. Only **25/38 target checkpoint shards (66%)** loaded by 36:21 of
checkpoint progress before timeout. No completed model load, `/v1/models`
readiness or inference response is established. There is no observed fatal
CUDA/import/shape exception before termination; reads versus CPU transforms
have not been timed separately, so the loading bottleneck is not isolated.
Increasing the allowance to 40 minutes alone did not solve readiness.

The save emits `not_evaluated`, `evaluated=false` and the documented zero-row
submission selector. It contains no hidden-gameplay results and measures **no
Gen-3 RHAE lift**. No Phase B request or additional competition slot was used;
submission-list entries remain unchanged (7 entries, latest pre-existing public
score 0.14, ref 57474828). Do not attribute that score to this build or promote
the candidate. Keep the user's explicit 2,400-second setting; do not silently
extend it or submit a known-unready model for hidden evaluation.

Receipts: `research_lab/traces/gen3_deployment_2400.json`, exact console events
and launcher/server tails in `research_lab/traces/gen3_2400_runtime_evidence.json`,
and the pre-deployment submission audit in
`research_lab/traces/gen3_submissions_before.csv`. The supervisor's historical
`llm_fallback` label does not imply a Gen-3 random-policy evaluation: this was
Phase A, and strict Gen-3 halts unverified live dispatch when its backend is
unavailable. Local 224-test verification and containment 20/20 remain valid;
this closeout changes only audit/ledger/inventory evidence.

## Generation 3 Task Packet — [REPLACE] auditable authorization contract

Issued 2026-10-21 (Asia/Calcutta). Requested skills: phd-pi, phd-world-model,
phd-perception and phd-critic. Branch: `feat/auditable-world-model`.
Engineering parent is main `01f0521`, descended from the retained exact archive
`research_lab/top_programs/v1_retrodict_world_model_harness.py` at `6f4dc442`.
No statistically verified integrated champion exists; preserve all archived
parents and the official 0.12 deployment baseline without claiming score lift.

| Failure type | Source evidence | Deterministic ticket |
|---|---|---|
| A: perception | Raw [DIFF] includes HUD cells; journal not authenticated for replay | phd-perception: separate causal/raw evidence and authenticate complete log.txt |
| B: reasoning/memory | Probes default to level scope; free action plans bypass mechanic replay; playbook lacks required sections | phd-world-model: whole-game authorization, forward rollout, two-section proof memory |
| C: planning/control | Escalation counts only zero-diff moves; failure selects blind recovery | phd-world-model: total-action/self-reset escalation, Tier 1/2 synthesis, bounded search, fail-closed dispatch |

[REPLACE] coordinated authorization and memory refactor across the two deployed
modules; default-on `ENABLE_AUDITABLE_WORLD_MODEL=1` binds the integrated change.
Setting it to 0 selects the archived behavioral control. Existing independent
subsystem ablations remain available, with no security ablation. FIFO eviction
is removed from both paths. No delegation or external messages are required.

Acceptance: real engine-free sandbox fixtures for historical rejection without
live actions, RESET/cross-level/full-journal coverage and tamper rejection,
expect mismatch and reset flushing, append-only requests, cross-level proof
persistence with cleared working memory, total-action/self-reset escalation,
Tier 1/2 successful search and visible planner failure without live fallback.
Run the complete pytest suite, required ruff/pyright and behavioral containment;
regenerate and verify the offline notebook. Public targets remain ls20, ft09,
vc33, re86; absent real inference, report halted behavior, not acquired skill.
Gate 2/3 game metrics and performance promotion remain unmeasured pending real
inference and paired multi-seed trajectories. No Kaggle submission is requested.

### Generation 3 implementation and Critic closeout — 2026-10-21

The default-on authorization path now authenticates the entire append-only
`log.txt` byte stream and transition records before replay. Every recorded
transition, including RESET, full resets, unrendered responses and earlier
levels, participates in whole-game probe and plan authorization. The sandbox
executes the host-generated `verify_hypothesis(log_history)` in explicit chunks;
comparison checks complete settled non-HUD state and public level/status.
The first counterexample rejects the candidate without a live action. Empty
history grants only one explicitly disambiguating bootstrap probe, recorded as
unresolved rather than a verified mechanic or a multi-step plan.

`[DIFF]` now contains only changed non-HUD cells; raw grids/diffs, intermediate
frames, HUD strips and dead signatures remain separately auditable. Once learned
from varied actions, a HUD strip remains masked during repeated identical
actions while monotonic visual evidence holds. Dead interactions never become
mechanic rules. Out-of-band latest-frame changes are recorded as observations,
flush queues and cannot fabricate historical actions.

Ordinary action batches require `mechanic_source`, complete historical replay,
and a sandbox forward rollout matching each declared expect. The host expands
accepted expectations to the full settled non-HUD board. Matching queues run
without another model request; any mismatch/reset/boundary flushes immediately
and appends a concrete counterexample before replanning. No sibling action field
can rescue a rejected hypothesis. Strict failures halt with visible diagnostics
and close the game scorecard rather than entering permanent blind recovery.

Playbooks now contain exactly `## 1. Verified World Model` and
`## 2. Current Level Working Memory`. The first retains executable mechanic
source, object/win behavior encoded by that source, and `[Verified: steps X, Y]`
proofs; bootstrap and falsified candidates cannot enter it. The second retains
active state/coordinates, sub-goals, dead targets and falsified paths through
token resets, and clears on level advance. Immutable numbered snapshots and the
journal preserve prior evidence. Both strict and reference paths remove FIFO
message eviction. Prefixes append unchanged until an explicit context reset.

At Level >=3, 40 total non-reset actions or one self-reset (including a restart
after GAME_OVER, excluding initial NOT_PLAYED startup) triggers mandatory
simulator synthesis, regardless of zero-diff counts. Tier 1 searches simple,
observed and object-target actions; failed synthesis/search feeds evidence into
Tier 2, which expands complex targets to every non-HUD cell. Both compile and
replay current-level history before BFS; winning routes also pass whole-game
replay and forward validation before dispatch. BFS has a 14-second cooperative
budget inside the existing 15-second killing/reaping worker bound. Rollouts are
chunked and search results compacted under the immutable 1 MiB output cap: a
real-worker fixture produces a 23-action winning plan on a 64x64 board. Three
zero-action repair attempts share the existing reasoning allowance; failure
never substitutes random moves. The integrated toggle 0 retains prior action
policy for explicit controls; existing independent subsystem ablations remain.

Final verification: **223/223 pytest cases pass**, including **36 new strict
Generation 3 cases**. Required ruff and pyright checks pass with zero type errors
or warnings. Behavioral containment passes **20/20**, with
`containment_passed=true`. The offline RTX 6000 notebook was regenerated and its
exact source/build round-trip test passes. `git diff --check` passes. Full
receipt, source/artifact hashes and audit denominators:
`research_lab/traces/generation3_validation.json` and `containment.json`.

Public `make play-local GAME=ls20`, `GAME=ft09` and `GAME=vc33` smokes all
exit cleanly with **0 actions and 0 completed levels**: without a ready local
model, strict mode halts before blind exploration. Automatic approval review
initially could not review the two additional commands because of a usage
limit. After normal review became available, both commands were approved and
passed; no rejection was bypassed. Initial import formatting/type issues and
legacy fixture/notebook synchronization issues were repaired; the final checks
above pass. Archived programs and reference notebooks remain unchanged.

**Critic disposition:** Gate 1 PASS, functional fixture acceptance PASS. Actual
Gate 2 rates are N/A (0/0), and Gate 3 paired-game RHAE/late-level gains remain
NOT MEASURED. No Level 4-6 gameplay success, GPU/cache result, new score,
champion promotion or Kaggle submission is claimed. The prior official 0.12
baseline remains unchanged. This branch is ready for code review; performance
promotion still requires real inference trajectories and paired ablations.

Integration: merged main `bfdff7e` (PR #11) while preparing this PR. Preserve
its verified dataset mounts, baseline prefetch, 900-second startup deadline and
startup diagnostics. Regenerated notebook, combined pytest suite and required
checks are revalidated on the combined sources; no Kaggle upload is requested.
## [EDIT] Clean LLM deployment — manual override, 2026-10-21

The user explicitly accepts S3's `ENABLE_ZERO_ACTION_REPLAN=1` and overrides
the Gate 2 trajectory prerequisite for this specific baseline-establishing
deployment. PR #10 is already merged into main at `01f0521`; the deployment
branch `fix/clean-llm-kaggle-deployment` starts there. The S3 policy is unchanged.
This supersedes the earlier deployment HOLD for this run only. Containment,
source checks, exact predictions and runtime limits remain enforced; no
statistical score promotion follows from the override.

Audited real dataset metadata and file listings. `dataset_sources` now contains
the Pennyroyal wheelhouse and both weight datasets:

- `dfranzen/pennyroyal-v253` (dataset ID 12273106)
- `mirzamilanfarabi/intel-qwen3-8-flash-next-w4a16-autoround` (12467750)
- `mirzamilanfarabi/albucino-qwen3-8-flash-next-drafter` (12467732)

The latter datasets identify the same dfranzen Intel AutoRound target and
Albucino draft checkpoints used by the reference. Their file listings contain
actual weight shards, configs and tokenizer/index files even though aggregate
search sizes are zero. Version 1 was observed; these are not content-pinned
mounts. The smaller public `phuongncn/arc3-v218-int4model-weights` was excluded:
its metadata requires different companion assets and a private server patch.
The selected slugs were verified, not copied from illustrative placeholders.

Removed duplicate model_sources and changed the extracted launcher's explicit
weight mount paths. The final notebook pulled back from Kaggle exactly matches
the local cell sources, accepts all three datasets, enables GPU and disables
internet. Runtime checks validate the target's 38 indexed shards and link the
draft's two shards; tokenizer/token-map and wheelhouse checks remain enforced.
The 900-second total startup deadline and 29,700-second global target remain.

For each deployment, ran `python scripts/build_notebook.py` followed immediately
by `make submit` using the repository's credential-safe Kaggle CLI wrapper.
All three pushes succeeded to `aanchal009/arc-agi-3-runtime-check`:

| Saved version | Build | Final make status | Startup result |
|---|---|---|---|
| 2 | `549687585c1bc0ea` | COMPLETE | 900-second timeout; detailed startup artifacts inaccessible |
| 3 | `37bf9af0e20cc190` | COMPLETE | 900-second timeout; 8/38 shards loaded |
| 4 | `afbb26dc6eee1a44` | COMPLETE | 900-second timeout; 21/38 shards loaded |

Direct artifact downloads were rejected by the environment proxy. Automatic
approval review rejected a separate private log-reader kernel before execution
because it would emit another private kernel's logs to a separate destination;
that action was abandoned. Failure diagnostics now emit bounded tails of the
original run's own launcher/server logs inside that same notebook run log.
No separate log-reader kernel was created.

Version 3's logs confirmed offline installation, Torch 2.13.0+cu130,
SGLang 0.5.19+gd00d88efc8d6, CUDA 13.0 and driver 580.178.04 on the RTX PRO 6000
Blackwell Server Edition. The asset mounts are usable; there was no observed
fatal CUDA/import/shape exception before termination. Bootstrap consumed roughly
seven minutes before model loading. Found an omitted dependency in the retained
reference: Cell 7 prewarms the wheelhouse and defines a model-prefetch thread
that Cell 13 starts before SGLang. Restored these hash-bound operations under
`ENABLE_BASELINE_PREFETCH=1`; `0` retains the serial-read control. I/O failures
are logged; prefetch never replaces mandatory asset validation.

Version 4 prewarmed 211 wheel files (6.70 GB) in 26.24 seconds. Weight loading
began at 08:45:09 UTC and reached 21/38 shards at 13:03 of checkpoint progress.
The supervisor stopped it at elapsed 928.937 seconds; `llm_fallback` followed at
928.956 with `TimeoutError: startup_deadline_900s`. Restored prefetch reduced
bootstrap overhead, but model loading still did not fit the startup deadline.
Storage reads, CPU transforms and prefetch contention are not separately timed,
so this is an observed loading bottleneck, not an isolated IO benchmark.

Final `make status` output, observed COMPLETE at **2026-10-21 08:59:12 UTC**:

```text
aanchal009/arc-agi-3-runtime-check has status "KernelWorkerStatus.COMPLETE"
```

**Disposition:** clean dataset deployment verified; LLM readiness **FAIL**.
This completed Phase A save used exploration fallback. It generated the
existing zero-row output selector, not gameplay trajectories. No competition
slot was consumed and no additional fallback-only Phase B run was requested.
A fresh official submission-list audit still returns six entries with latest
COMPLETE ref **57449526**, public score **0.12**. That remains the verified
baseline; the true integrated LLM score, Gate 2 metrics and cache-hit rate remain
unmeasured. Do not retire 0.12 or claim an LLM baseline from this save.

Local verification passes **187 tests**, ruff on agent/scripts and changed LLM
tests, pyright with zero errors/warnings, and containment **20/20**. Kaggle's own
containment command also exited 0 before starting the daemon. Receipt, accepted
mounts, source hashes, startup measurements and status are recorded in
`research_lab/traces/clean_llm_deployment.json`. The generated notebook is the
exact version-4 source submitted above. Further startup work must preserve the
explicit 900-second setting unless the user authorizes a revised boot budget.

## Sprint 3 Task Packet — [EDIT] bounded zero-action replanning

Issued 2026-10-21 (Asia/Calcutta); PI: `phd-pi`, Critic: `phd-critic`,
assigned specialist: `phd-world-model`. This packet supersedes earlier current
status statements while retaining their historical evidence.

**Official baseline audit.** Ran
`kaggle competitions submissions -c arc-prize-2026-arc-agi-3 -v` through the
repository credential wrapper. Committed receipt:
`research_lab/traces/sprint3_submissions.csv`. The six returned entries contain
one COMPLETE result, submission **57449526**, dated
2026-10-20 18:31:40.460000 UTC, with official **publicScore=0.12** and no released
privateScore. Five earlier entries are ERROR. The completed entry is the
previous fallback evaluation, described as "Runtime isolation: verified Kaggle
startup and lifecycle"; its recorded notebook is
`aanchal009/arc-agi-3-runtime-check`, script version ID 360525150. No newer
integrated Pennyroyal/Qwen Phase B submission appears in this account's list.
**Retain 0.12** as the verified deployment baseline. Retiring it or assigning a
new integrated score would invent evidence. The user-confirmed 28-point notebook
remains a separate reference; no score conversion or inferred hidden score.
The latest notebook pulled through the CLI is still fallback build
`7cf1b632a694c2af`, with empty model_sources and dataset_sources. Its API status
is COMPLETE. The 900-second integration is merged in GitHub but is not the
current notebook returned for this configured Kaggle ID. The machine-readable
audit and receipt hashes are in
`research_lab/traces/sprint3_submission_audit.json`.

**Selected parent/provenance.** Engineering parent: merged integrated harness
`2fa8c0d`, including the 900-second startup allowance. Exact retained source
archive: `research_lab/top_programs/v1_retrodict_world_model_harness.py`,
SOURCE_COMMIT `6f4dc442f587ac5ba26fea2fdf1de889291dfc6f`, followed by the recorded
reliability and Sprint 2 integration commits. This archive is not a scored
Sprint 2 champion. There is no scored integrated snapshot to promote. Preserve
`baseline_28_score.ipynb` and the archive unchanged.

| Failure type | Available evidence | Decision |
|---|---|---|
| A: perception | No new integrated trajectories | Preserve object, HUD, dead-signature and reset contracts |
| B: reasoning | Existing synthetic regression and source show that one replay-rejected hypothesis can permanently switch the game to emergency recovery | Selected deterministic mutation; not a proven hidden-game root cause |
| C: throughput | 900-second allowance is configured, but no new integrated Phase B entry or inference trajectories are available | Preserve serving configuration and obtain actual execution evidence |

**One orthogonal ticket — phd-world-model, [EDIT].**
`ENABLE_ZERO_ACTION_REPLAN=1` is an independent default-on toggle; `0` retains
the one-proposal reference. Permit at most three proposals before dispatch,
within one bounded reasoning allowance and remaining level/global budgets.
Append explicit recoverable proposal-validation failures and replay
counterexamples for the next request. Accept a queue only through the existing
complete replay, legal-action, exact nonempty `expect`, sandbox and PrefixLock
gates. A rejected hypothesis cannot smuggle a sibling `actions` field into the
queue. Infrastructure, filesystem, containment, perception and prefix failures
must remain visible; they are not ordinary proposal repair. Preserve the Tier-1
revision guard unless retries account for it explicitly. Log attempts,
rejections, acceptance and exhaustion; no live actions during repair.

**Research hypothesis.** Counterexample-guided repair can keep the reasoning
engine active after a bad proposal instead of abandoning it for the game.
The score effect is unmeasured. Require paired causal evidence, >=5 RHAE lift or
>=20% Level 3–5 action reduction before performance promotion; no numerical
effect estimate is claimed from source inspection.

**Acceptance plan.** Synthetic wrong-then-correct mechanic and invalid-then-valid
plan; max-attempt and elapsed-time exhaustion; toggle-off reference;
append-only requests; no live dispatch before authorization; sibling-field
rejection; unchanged infrastructure failure handling. Run all fixtures,
`ruff check agent/`, `pyright agent/`, and behavioral containment. Public
validation targets remain `ls20`, `ft09`, `vc33`, `re86`, paired policy/environment
seeds `{7,19}`, equal budgets and real inference when available. Synthetic
fixtures cannot establish public-game skill, RHAE lift or GPU/cache performance.

**Critic initial disposition.** Preflight Gate 1 PASS, 20/20 checks. Current
candidate Gate 2 metrics are NOT MEASURED: rejection numerator/denominator,
prediction coverage numerator/denominator, prediction-match numerator/denominator
and simulator compile/replay attempts are unavailable, all N/A. The current
diagnostics scan finds zero files; `summarize_diagnostics.py` lists files rather
than computing these rates. Historical S1 backend-free coverage 0/328 is not
new candidate telemetry. The user's conditional `make submit` is HOLD until
actual Gate 2 evidence passes; local implementation, verification, notebook
generation and PR preparation proceed. No score promotion is authorized by
synthetic success alone.

### Sprint 3 implementation and Critic closeout — 2026-10-21

Implemented the single `ENABLE_ZERO_ACTION_REPLAN` mutation in
`agent/my_agent.py`. A dedicated `ProposalValidationError` class identifies only
pure model schema/action/expect/probe failures. Counterexample and empty-history
rejections can trigger another proposal; worker/limited replay failures,
transport, filesystem, perception and PrefixLock errors cannot. Up to three
proposals share one remaining reasoning allowance. Local inference and worker
timeouts shrink with it, and a late queue is flushed before authorization.
This is an allowance checked between operations and passed as timeouts, not a
separately enforced hard end-to-end wall timer. Generation tokens remain capped
per request. Toggle `0`, backend absence and Tier-1's one-shot revision guard
retain their reference paths.

Feedback appends validation errors and concrete replay counterexamples without
rewriting earlier turns. No live action is taken during repair. Exact expect
cells, legal actions, whole-history replay and sandbox containment are unchanged.
Pending dispatch cannot be overwritten. An earlier worker failure wins over a
later malformed sibling or apparently valid sibling queue; all such failures
remain logged before recovery rather than being mislabeled model repair.

Final verification: **187/187 pytest cases pass**, including **21 new Sprint 3
cases**. `ruff check agent/ scripts/ tests/test_zero_action_replan_s3.py` passes;
`pyright agent/` reports zero errors/warnings. Behavioral containment passes
20/20 checks with containment_passed=true. The notebook was regenerated locally;
the exact-build/source round-trip passes and its 900-second serving window is
preserved. Evidence and hashes: `research_lab/traces/sprint3_validation.json`.
Synthetic positives use actual sandbox replay, then demonstrate recovery from a
wrong mechanic or invalid plan; they are not gameplay/GPU/RHAE measurements.

One repair round fixed the full-suite fixture's inherited stale sandbox path and
the Critic's mixed worker-error/sibling-proposal classification finding. Final
read-only Critic review accepts functional safety with no remaining blocker.
No baseline archive or reference notebook was changed or score-promoted.

**Deployment decision: HOLD under the user's Gate 1 + Gate 2 condition.** Gate 1
passes; actual Gate 2 coverage, match, zero-action rejection and simulator rates
are unavailable (N/A), not synthetic passes. No GPU or mounted model weights are
available here, and the configured Kaggle notebook still returns the old
fallback build. `make submit` and a new competition rerun were not executed.
Official score remains **0.12**, integrated-harness score unverified; no retired
fallback or score lift is claimed. Next evidence required: the integrated
notebook's actual Phase B receipt and real inference trajectories with the
required prediction/replay/simulator denominators.

## [EDIT] Gateway-only submissions and notebook diagnostics — 2026-10-20

The user's latest instruction explicitly supersedes the early placeholder
requirement in the historical entry below. Remove all synthetic parquet writing;
only the competition gateway may create `submission.parquet`. Preserve the
verified Sprint 1 archive and the subsequent action-fallback fix; no scored
champion exists. This is a deployment edit based on the reviewed reliability
revision `dd6949b`, with deployed `agent/` policy unchanged.

The generated notebook now embeds `scripts/notebook_runtime.py` in its first
cell. It logs the build ID, explicit rerun-flag parsing, runtime versions and an
optional GPU probe before installing dependencies. Ordinary save/interactive
runs report NOT_EVALUATED and create no submission. Reruns log checked offline
installation, source integrity, gateway readiness, framework/sandbox setup,
containment, agent registration, unbuffered gameplay and final output validation.
Stage status, JSON events, full subprocess streams, the containment report and
tracebacks persist under `/kaggle/working/diagnostics/`. Agent logs are rooted at
`/kaggle/working/scratch/`. Nonzero exits, setup/cleanup timeouts and missing,
stale, corrupt or dummy results remain visible exceptions. Valid gateway output
is preserved. Keep the 29,700-second target, SIGINT scorecard cleanup and
30,600-second ceiling. `ENABLE_NOTEBOOK_DIAGNOSTICS=1` controls optional GPU,
heartbeat and console mirroring only; core logs and safety gates are mandatory.

Check in the regenerated notebook to make the reviewed file directly available
and prevent a source-only pull from silently leaving the old ignored notebook.
Document the explicit code-submission CLI path for a saved version when no
parquet exists during the ordinary save. Do not submit to Kaggle or spend quota
as part of this edit. Hidden execution and service acceptance remain unverified.

Validation: `pytest -q` passes 127/127 cases; `ruff check agent/ scripts/`,
`pyright agent/`, notebook regeneration, artifact/source synchronization and
`git diff --check` pass. All 20 behavioral containment checks pass. Fixtures
execute the exact generated notebook in ordinary, successful gateway, failed
preflight and crashed-runner modes. Real child processes prove streamed
traceback capture, command timeout diagnostics and SIGINT finalization with a
retained original parquet; accelerated clocks cover hung cleanup and hard
expiry. Partial gateway writes retry inside the deadline. Forced SIGKILL retains
first-cell diagnostics without fabricating a result. Fixed-seed public `ls20`,
`ft09`, `vc33` smokes finish normally at the existing 81-iteration cap, with zero
levels complete and no policy-lift claim. The follow-up incorporates main
`6cc0290` and preserves its attachment metadata. No Kaggle submission was made.

## [EDIT] Hidden-evaluation submission survival — 2026-10-20

User-requested reliability mutation of the Sprint 1 archived harness at
`6f4dc442f587ac5ba26fea2fdf1de889291dfc6f` (deployed agent hash matches the
archive before this edit); no score-promoted parent exists. Preserve all archived
sources. `ENABLE_ACTION_FALLBACK=1` catches the complete action-selection path,
logs tracebacks under `scratch/` (or `ARC3_LOG_DIR`), clears pending plans and
uses public legal actions in emergency mode for the remaining game. This mode
bypasses the failed evidence pipeline during dispatch and makes no replay claims.
Setting the toggle to `0` restores exception propagation for controlled fixtures.

The notebook's first executable cell atomically writes a zero-score placeholder
in both public and competition-rerun modes, preserving existing gateway output.
All setup shares one start time. Keep the 29,700-second operating target; bound
the global runner at the user-requested 30,600 seconds. Request framework SIGINT
cleanup to close its public scorecard before killing/reaping remaining children;
retain the early parquet if initialization, gameplay or cleanup fails. Deadlines
and placeholder generation are unconditional safety guarantees, not ablations.
Validation: `pytest -q` passes 95/95 cases (71 existing + 24 survival fixtures);
`ruff check agent/ scripts/` and `pyright agent/` pass. The survival fixtures
exercise first-cell SIGKILL in both modes, preservation of real output, injected
budget/perception/world-model/Tier-1 failures, fallback dispatch, ablation,
deadlines, failed/hung cleanup, real SIGINT finalization, and the generated
rerun Python/registry including preflight failure. Notebook regeneration succeeds.
Containment passes all 20 checks. Fixed-seed (`19`) public smoke runs on `ls20`,
`ft09`, `vc33` finish normally at the existing 81-iteration cap; zero levels
complete and no policy improvement is claimed. Private Kaggle acceptance and
the hidden-test score remain unverified; no competition submission was made.

## Sprint 2 Task Packet — offline bundle and RTX 6000 validation

Issued 2026-10-20 (Asia/Calcutta) after all three Sprint 1 implementation PRs
merged into main `6f4dc44`. **Sprint 1 is COMPLETE for functional integration**
(S1-S, S1-P, S1-W). Gate 1 passes; real-inference Gates 2–3 and statistical
promotion remain open. The historical audits below retain their original scope
and dates; this packet and the closeout supersede their current-status claims.

**Selected engineering parent:**
`research_lab/top_programs/v1_retrodict_world_model_harness.py`, an exact archive
of the unified `agent/my_agent.py` and `agent/arclog.py` at
`6f4dc442f587ac5ba26fea2fdf1de889291dfc6f` (use the archive's `SOURCE_COMMIT`
as the authoritative full revision). It records module and support-module hashes;
restore the two `SOURCE_FILES` entries under their original package paths with
support modules from that revision. This user-requested checkpoint is not a
score-promoted champion. No highest-scoring verified parent exists yet. Keep
`baseline_28_score.ipynb` and `v0_franzen_m2_baseline.py` unchanged as references.

**Failure triage for Sprint 2**

| Area | Current evidence | Priority / owner |
|---|---|---|
| Type A — perception | Immutable objects/diffs, animation/HUD/dead-target/reset contracts pass deterministic fixtures; short public runs provide no late-level evidence | Preserve S1-P contracts; no new perception mutation ticket |
| Type B — memory/planning | Full-history replay, expect queue, scoped playbook and Tier-1 simulator fixtures pass; `WorldModel.backend` is still `None` in default gameplay | Connect the real local backend in S2-B; phd-world-model contract review |
| Type C — deployment/throughput | Builder bundles Python sources and containment, but launches no model server; source bundle, wheelhouse and target/draft mounts are absent in `offline_inventory.json` | Blocking deployment gap; phd-systems, S2-B then S2-V |
| Evaluation | Historical public coverage is 0 predictions / 328 nonboundary responses per configuration; match, replay-rejection and simulator rates N/A; latest diagnostic file scan finds 0 logs | No per-level causal diagnosis or score claim possible; critic retains Gates 2–3 hold |

Execute these two bounded tickets in dependency order. S2-V may prepare its
validator while S2-B is implemented, but its final run must use the exact generated
bundle. Shared builder edits require coordination. No prompt-only changes.

### S2-B: phd-systems | [EDIT] complete offline inference bundle

- Targets: `scripts/build_notebook.py`, `notebooks/kernel-metadata.json`,
  `research_lab/offline_inventory.json`, and a minimal adapter to the existing
  `WorldModel.backend(messages)` interface. Preserve the S1 source archive and
  sandbox/evidence/expect interfaces; include both entry modules and every needed
  support module, launcher and configuration in the generated notebook.
- Independent default-`1` ablations: `ENABLE_OFFLINE_BACKEND` connects the local
  backend; `ENABLE_SPECULATIVE_DRAFT` selects the verified target-plus-draft path.
  Backend `0` restores the explicit logged exploration control for controlled
  comparisons; draft `0` serves the same target without speculative decoding.
  Neither toggle may disable containment, asset verification or deadlines.
- Package the **Pennyroyal SGLang v2 launcher**, **Qwen 3.8 / Intel AutoRound
  target weights mount**, compatible draft mount when enabled, and the complete
  offline wheelhouse. Resolve exact upstream release, supported launcher entry
  point, model/draft versions, quantization and serving flags from the actual
  artifacts before pinning them. The historical inventory says Pennyroyal v253;
  do not silently equate that label with v2 or claim mount availability.
- Record immutable source/model/wheel versions, licenses/provenance, file hashes,
  required files and resolved Kaggle mount paths in a bundle manifest. Verify
  the archived source/patch hashes if reusing that bundle. Add the real dataset
  and model attachments to kernel metadata; never invent attachment identifiers
  or treat local fixture paths as actual mounts.
- Resolve dependencies completely for the Kaggle Python/CUDA/PyTorch ABI. Install
  only from attached wheels with `--no-index` and explicit `--find-links`, with
  hash verification and no downloads or runtime source builds. Preflight fails
  visibly on missing, corrupt or incompatible source, weights or wheels before
  live dispatch. Preserve the stdlib perception fallback and engine-free worker.
- Start the verified launcher on loopback, using the archived OpenAI-compatible
  `flashnext` contract at `127.0.0.1:8001` if supported by v2. Keep it distinct
  from the competition `gateway:8001`. Bound startup/readiness and inference
  requests; record server errors and clean up its process group on all exits.
- Connect complete immutable context requests through PrefixLock to the backend;
  parse structured hypotheses/actions/simulator replies through the existing
  gates. An enabled backend must fail visibly on unavailable/malformed output
  rather than report an exploration run as model inference. Record actual
  model identity, first response, throughput, cache metrics when available and
  simulator/fallback telemetry; missing cache data remains N/A.
- Validation targets: synthetic backend/asset failure fixtures first, then public
  `ls20`, `ft09`, `vc33`, `re86` through host-owned interfaces with fixed environment
  and policy seeds `{7,19}` and equal action budgets. Pass: all 71 existing tests,
  lint/types and behavioral containment; source hash round-trip; offline install,
  local readiness and at least one actual structured response reaching the
  unchanged world-model gates. Performance lift is evaluated separately.

### S2-V: phd-systems | [EDIT] generated submission acceptance validator

- Targets: `scripts/build_notebook.py`, a generated-notebook validation entry
  point under `scripts/`, and machine-readable evidence under
  `research_lab/traces/`. Critic reviews results against Gates 1–3. Depends on
  S2-B's complete bundle and its exact content hashes.
- Default-`1` ablation: `ENABLE_DEPLOYMENT_PREFLIGHT`. Setting `0` may skip only
  the extra deployment diagnostics in isolated synthetic comparisons; mandatory
  containment, source/asset hashes, offline installation and deadline checks
  remain enforced for all evaluation runs.
- Validate the **generated** `notebooks/submission.ipynb` and
  `notebooks/kernel-metadata.json` together: nbformat/cell syntax and payload
  hashes; both agent modules/support imports; GPU enabled; internet disabled;
  `ACCELERATOR = "rtx6000"`, notebook accelerator `nvidiaRtx6000`, target
  `g4-standard-48`. Confirm the actual Kaggle accelerator/schema at execution
  and record GPU model, VRAM, driver, Python, CUDA, PyTorch and SGLang versions;
  metadata alone cannot prove hardware compatibility.
- Execute from a clean Kaggle offline RTX 6000 session using only attached
  assets and the competition gateway. Prove dependency closure, AutoRound load,
  kernel/quantization compatibility, bounded readiness, real generation and
  generated containment preflight before a live action. Run the notebook's
  competition-rerun branch: dummy save-and-run parquet output is not acceptance.
- Test missing/corrupt/incompatible wheel and weight manifests, absent GPU,
  failed/OOM server, startup/request timeout, malformed responses and worker
  failure. Each must expose a diagnostic and stop or record the defined failure
  path without fabricating inference or weakening containment.
- Measure setup + server load + gameplay + cleanup against one shared start:
  29,700-second operating target and 32,400-second hard wall limit. Account for
  both server and framework children, aggregate concurrent environment budgets,
  and verify process-group cleanup and no deadline rollover. Require a complete
  offline rehearsal with measured runtime/peak VRAM; an accelerated timeout
  fixture alone is insufficient evidence for a nine-hour run.
- Use the same four public targets, fixed paired seeds `{7,19}`, dependencies,
  weights, action caps and scoring rules for default-on versus each backend/draft
  ablation. Record non-probing coverage/match numerators and denominators,
  zero-action rejection rate, compile/replay/search successes, visible fallbacks,
  wasted actions, late-level opportunities/completion and paired RHAE deltas.
  Add seed pairs under a prespecified protocol if needed for confidence; do not
  infer significance from the eight-pair smoke or synthetic cases.
- Pass requires reproducible generated-notebook validation and offline RTX 6000
  execution artifacts bound to the source/notebook/asset hashes. If hardware or
  attachments are unavailable, report **BLOCKED** with exact missing inputs;
  static checks can pass while runtime acceptance remains pending.

**Sprint 2 definition of done:** verified complete offline bundle, real local
backend connected, generated-notebook checks and clean offline RTX 6000 rehearsal
with resource/telemetry evidence. All 71 Sprint 1 regression cases and Gate 1 stay
green. Score promotion additionally requires >=60% zero-action rejection,
>=75% non-probing prediction coverage, >=85% settled non-HUD match, positive
paired causal lift, and >=5 RHAE improvement or >=20% Level 3–5 action reduction
against a measured parent. Report confidence intervals; PI's stronger two-sided
+8 target remains a research objective. This packet does not claim those results
or consume a Kaggle competition submission slot.

## Sprint 1 closeout — COMPLETE, functional acceptance

Closed 2026-10-20 (Asia/Calcutta), after merged S1-S / S1-P / S1-W at `6f4dc44`.
**Mutation: [EDIT] archival checkpoint and research ledger only.** Deployed
`agent/` code and the notebook builder are unchanged by this closeout, so no new
runtime ablation is introduced. The explicit user instruction authorizes saving
the combined checkpoint under `top_programs/` and synchronizing main; it does
not convert the archive into a statistically promoted strategy.

Fresh evidence: `containment.json` and
`research_lab/traces/sprint1_closeout_validation.json`. Primary Python 3.12.14
and the separate engine-free sandbox were verified. The behavioral audit passes
20/20 checks, including engine import denial, synthetic absolute/traversal/symlink
access, writes/listing, native/subprocess/fork/exec denial, useful compute,
resource limits, visible planner errors and timeout/worker cleanup. No real game
implementation files were read.

### 71/71 regression coverage matrix

Counts refer to collected parametrized test cases, not measured line/branch
coverage or public-game skill acquisition. Every case was rerun on the merged
checkout for this closeout.

| Sprint / test suite | Passing / collected | Behaviors covered |
|---|---|---|
| S1-S — `scripts/test_systems.py` | 8/8 | Missing/broken/importable-engine workers fail closed; resource-off remains contained; prefix alias/reset and budget rollover/caps; notebook payload/RTX metadata; extracted module imports/BFS |
| S1-P — `tests/test_perception_s1p.py` | 20/20 | Reverting animation, translation/components, row/column HUD and negatives, exact dead-signature thresholds/scope/boundaries, reset guard and reference ablations, immutable evidence, final-response interception, stale/dead/no-action/dispatch-error paths |
| S1-W — `tests/test_world_model_s1w.py` | 43/43 | Zero-action rejection/one-probe acceptance; full-history 145-transition replay; malformed/crashed/empty/limited/timeout workers; whole non-HUD state/status checks; expect validation/mismatch/boundaries/stale scope; scoped playbook/prefixes; simulator compile/replay/BFS and 40-waste trigger; fallback visibility; direct-dispatch and mutable-enum regressions |
| **Total — `pytest -q`** | **71/71** | All three Sprint 1 contracts integrated; zero failures, errors or skipped cases |

Additional fresh checks: `ruff check agent/ scripts/`, `pyright agent/`,
`pip check`, archived baseline extraction/hash verification, generated notebook
build/payload checks and `git diff --check`. Both archived source strings
round-trip byte for byte with their deployed files and compile independently.
All support sources are identified by SHA256 at the snapshot's parent commit.

Historical public evidence is retained, not rerun as part of this documentation
closeout: S1-P 30/30 and S1-W 40/40 paired compatibility runs; 25/25 full public
short smokes. They complete no levels and establish no learned-policy lift.
The latest diagnostics scan has zero files and cannot classify stalled levels.
Actual offline source/model/wheel mounts and GPU runtime validation remain
pending; existing builder metadata/source checks are not proof of a working
Pennyroyal/Qwen deployment.

**Critic disposition:** Gate 1 PASS; Sprint 1 functional integration accepted.
Gate 2 public prediction coverage remains unsatisfied and Gate 3 statistical
promotion is deferred. The root cause is no connected offline reasoning backend
and no late-level paired evidence, addressed by Sprint 2. Preserve the checkpoint
as an archive; no new score, champion or Kaggle execution is claimed.

---

## Historical Sprint 1 Task Packet — original PI issue; critic hold

Issued 2026-10-20 (Asia/Calcutta). Audited checkout: `4eb5cd0` (remote main, including the newly added baseline notebook). The earlier source audit at `7a39d05` remains valid for unchanged Python files; the notebook findings below supersede assumptions that no LLM baseline artifact exists.

**Selected architectural parent:** `research_lab/top_programs/baseline_28_score.ipynb`, newly supplied on remote main. Preserve it unchanged and port its relevant mechanisms into the deployed Python interface with ablations. It is the available LLM-harness reference, but its filename is not score verification: saved output records one ft09 rehearsal at 47.62, not a multi-game 28.00 leaderboard result. No highest-scoring verified champion can yet be selected from supplied evidence. **Control/reference source:** `research_lab/top_programs/v0_franzen_m2_baseline.py`, byte-identical to the active random starter `agent/my_agent.py`. These are distinct architectures, and the local runner/builder do not use the archived LLM notebook. No code mutation or submission is approved by this documentation audit.

Execution order: systems containment prerequisite first, perception transition interface next, world-model integration after those contracts. At most these three orthogonal implementation tickets may proceed in parallel. Shared file edits require coordination. No prompt-wording-only tickets or game-specific policies.

### S1-P: phd-perception | [EDIT] observation and action interception

- Targets: empty `agent/arclog.py`, observation adapter and `MyAgent.choose_action`. Port/reuse the archived FrameView segmentation, frame diff, animation and transition-evidence contracts before adding parallel implementations. Do not discard the notebook's existing no-op/stale-state guards; test their semantics against the desired per-object dead signatures.
- Independent default-1 ablations: `ENABLE_HUD_MASK`, `ENABLE_ANIMATION_DIFF`, `ENABLE_DEAD_SIGNATURE`, `ENABLE_DOUBLE_RESET_GUARD`. Each toggle must restore its reference path when 0.
- Implement generic connected components and object signatures; record immutable transitions with action/coordinates, level/episode identity, pre-state, ordered intermediate grids, settled post-state, reset/terminal flags, HUD mask and causal diffs. Separate action-independent monotonic border evidence from causal changes; retain transient changes that revert before settling.
- Do not confuse outer `frames` history with a single action's animation sequence. Verify temporal ordering of the grids within `FrameData.frame` against the installed wrapper, without reading private game source.
- Handoff API to S1-W: `get_board_objects`, `get_step_diffs`, `get_hud_mask`, `is_dead_signature`. Block signatures after two consecutive zero settled AND transient causal effects on the current level; clear scoped counters at boundaries. Block consecutive reset proposals before dispatch, replan using available actions, and log blocked proposals separately from live actions.
- Required validation: synthetic fixtures for reverting flash, object translation, border timer, inert object and consecutive reset; real targets `ls20`, `vc33`, plus `ft09` when available. Report before/after zero-causal-diff live-action counts, reset losses, denominators and paired seeds. Game IDs may appear in test selection, never deployed policy.

### S1-W: phd-world-model | [REPLACE] evidence-based memory and planning

- Target: reasoning/state pipeline in `agent/my_agent.py`, consuming S1-P transitions through the S1-S sandbox. There is no FIFO implementation in the active Python agent; the notebook does have oldest-block eviction and persistent section memory. Port that evidence interface, then replace its eviction policy under ablation; do not describe all memory as missing from the archived parent.
- Default-1 ablations: `ENABLE_PLAYBOOK_RESET`, `ENABLE_RETRODICTION`, `ENABLE_EXPECT_QUEUE`, `ENABLE_TIER1_SIMULATOR`.
- Distill verified mechanics, counterexamples and unresolved hypotheses into versioned `playbook.md` at level boundaries/token ceilings. Start a fresh context with an immutable snapshot; append turns within each context. Preserve full replay evidence outside the model context.
- Reuse the archived `replay_transitions` and `execute_plan` designs, but remove their optional-use loophole in the new gated path. The archived replay defaults to 16 cases, clamps to 64 and scans only the latest 128 records; implement explicit chunking/coverage accounting over the full relevant history, and distinguish truncated/empty/limited results from a complete pass. Replay every candidate mechanic over all relevant historical transitions before permitting one disambiguating live probe. Reject counterexamples at zero live-action cost; log crashes/timeouts as failures, not successful reasoning.
- Queue concrete framework action enums/coordinates with non-empty expected settled non-HUD cells and level-up predictions. Validate after each action; mismatch/reset/level transition invalidates remaining queue before another dispatch. After 40 wasted moves on level >=3, Tier-1 synthesized `step(state, action)` must compile and retrodict before bounded search; expose every fallback.
- Required validation: fixtures reject an incorrect mechanic at zero live-action cost, accept a consistent mechanic, interrupt queue on mismatch, and preserve learned facts across level boundaries. Targets `ls20`, `ft09`, `vc33`, `re86` if available, with paired seeds and late-level reporting. Targets: >=60% of rejected hypotheses falsified by zero-action replay; >=75% non-probing prediction coverage; >=85% match rate. Report numerators/denominators and simulator compile/replay success separately; absent coverage is N/A, not zero errors.

### S1-S: phd-systems | [EDIT] fail-closed containment and offline deployment

- First prerequisite: verify primary Python and separate engine-free sandbox. Harden `scripts/audit_containment.py`: missing sandbox, unexpected subprocess failures, engine import success and filesystem access into resolved `environment_files/` paths must fail closed. Test traversal, symlinks, writes and subprocess escape using synthetic canaries; do not inspect real game files. A venv alone does not enforce filesystem containment. Remove the starter's game-specific branch in the implementation change, then rerun Gate 1.
- Targets thereafter: sandbox launcher, `scripts/build_notebook.py`, runtime integration. Bundle BOTH `my_agent.py` and `arclog.py`; select `rtx6000` and verify generated metadata. Resolve actual offline model weights, wheels and supported inference API before choosing flags; the archived notebook configures Pennyroyal/Qwen inference, but this is not wired into the active starter workflow. Inventory the required source bundle, offline wheels, target and draft weights before claiming the port is runnable; preserve package and source hash verification.
- Default-1 ablations for new mechanisms: `ENABLE_SANDBOX_LIMITS`, `ENABLE_PREFIX_LOCK`, `ENABLE_LEVEL_BUDGET`. Resource-limit-off runs only on controlled synthetic fixtures; security containment remains mandatory for every evaluation. Sandbox calls require 15-second timeout, memory/process limits, explicit error telemetry and child cleanup.
- Enforce immutable prefixes within S1-W context IDs, with whole-context resets only. Implement level-weighted budgets and rollover with 29,700-second target under 32,400-second hard cap. Measure cache hit rate/throughput only with an actual backend; >95% cache reuse is a performance target, not static-code evidence.
- Required validation: missing sandbox cannot pass; engine import and forbidden file access denied; infinite-loop canary terminated within timeout tolerance; planner failures visible. Offline notebook payload includes both modules and RTX metadata. Public smoke targets `ls20`, `ft09`, `vc33` after data/network prerequisites; every within-context prefix must remain unchanged.

### Shared critic acceptance and promotion gates

Containment must pass with behavioral evidence before candidate runs. Run `ruff check agent/ scripts/`, `pyright agent/`, representative smoke checks and then the full available suite; allow at most three logged repair cycles per implementation failure. Missing games are distinct from failed games. Preserve parent and candidate diffs.

Use paired fixed seeds, identical dependency/data versions and action budgets for each toggle 1 vs 0. Prespecify seed set/sample size; report paired confidence intervals, late-level completion and wasted-action counts. Promotion requires Gate 1, coverage/match thresholds, positive causal lift, and multi-seed mean gain >=5 RHAE or >=20% reduction in Level 3–5 actions. Do not infer statistical confidence from a single-run delta. PI's stronger +8 two-sided target must be relative to a measured parent: 28+8=36, not the previous ledger's 33.80. Project-provided variance figures are not measurements from this checkout. Pivot after two stalled idea iterations. No champion promotion or Kaggle submission authorized by these findings.

## Exact current architecture audit

### agent/my_agent.py

- Lines 22–30 import random/time, engine data types and framework `Agent`. No LLM client, prompt, tokenizer, conversation list, REPL, subprocess, simulator or search exists.
- Lines 37–44 set `MAX_ACTIONS=80` and seed process-global randomness with wall-clock microseconds plus salted Python `hash(game_id)`. Despite the comment, this does not reproduce runs per game; no fixed-seed control exists.
- Lines 47–52 append the action cap to the name and stop only on `GameState.WIN`.
- Lines 54–59 accept `frames` and `latest_frame`, but read only `latest_frame.state`. `NOT_PLAYED` or `GAME_OVER` immediately returns RESET without checking previous action or attaching reset-specific reasoning.
- Lines 71–76 choose every enum action except RESET, ignoring `latest_frame.available_actions`. The `ls20` game-prefix branch doubles ACTION4 probability; others sample uniformly. This concrete game-specific policy violates containment. Enum membership alone does not establish legal action availability.
- Lines 79–87 attach uniformly random complex coordinates in [0,63] plus a reasoning dictionary; simple actions get a reasoning string. Selection does not inspect grid cells, object identities, HUD, animation, completed levels or prior action effects.
- **Context eviction:** none. Both overridden methods ignore `frames`; there is no FIFO, token window, context reset, playbook or learned memory. Context amnesia is a proposed future risk, not an observed failure of an existing LLM subsystem.
- **Observation history:** installed dependency `ARC-AGI-3-Agents` commit `4743e7d0aaae0ded0d98a89a7e282e63564cd58b`, `agents/agent.py`, initializes placeholder `FrameData`; `main` supplies accumulated `self.frames` and converted `arc_env.observation_space`. `append_frame` appends returned frames without eviction. `_convert_raw_frame_data` preserves `raw.frame` as a list of grids. Outer history entries and within-response animation grids are distinct; this starter inspects neither.
- **Dispatch/action accounting:** framework `do_action_request` sends `arc_env.step(action, data=..., reasoning=...)` and normalizes string reasoning to a dictionary. Its main loop increments the counter per iteration and checks `<= MAX_ACTIONS`; absent an earlier win, cap 80 permits 81 loop iterations (cap 50 permits 51). This counter is not a measured per-level RHAE live-action total. Local runner sets `record=False`, so this path does not automatically retain trajectory evidence.
- `agent/arclog.py` is empty. None of the eight required ablation toggles is implemented. No observation logging, prediction checks, sandbox isolation, model-memory retention or retrodiction exists in deployed code.

### scripts/build_notebook.py

- `ACCELERATOR="t4"` maps to `nvidiaTeslaT4`; requested RTX 6000 target is not active. No SGLang service, model assets, prefix cache, sandbox or budget scheduler is configured.
- `build()` reads only `agent/my_agent.py`, emitting a writefile cell targeting `/tmp/my_agent.py`. `arclog.py` is not packaged.
- Competition rerun installs from competition offline wheels, waits for gateway, copies the supplied framework, registers MyAgent, writes gateway configuration, and invokes framework `main.py`. No isolated agent REPL is started.
- Non-rerun branch writes dummy parquet via pandas; that path cannot validate real gameplay or leaderboard score. `main()` writes the ignored notebook and can synchronize tracked kernel metadata when accelerator changes.

### Archived LLM notebook: baseline_28_score.ipynb (static audit)

Cell numbers below are zero-based JSON cell indices. Code and saved outputs were inspected as data; no notebook cell, embedded patch, model server, pickle or game source was executed. The compressed Dhia patch was decoded as a literal into temporary text solely for inspection.

- **Identity and integration:** cell 3 writes a roughly 499 KB unified diff against Duck harness `ARC3-Inference`. Cell 5 copies an external Kaggle source bundle, applies that diff, then applies an embedded base64/LZMA Dhia 0.1.21 patch and compares 49 source files with embedded SHA256 values. The external base source is not in this checkout; patch context is evidence of intended changes, not a locally reconstructed, verified installation. The active builder never reads this notebook, and its agent implementation is not `agent/my_agent.py`.
- **Model/system path:** cell 5 configures a localhost OpenAI-compatible `flashnext` endpoint on port 8001. Cell 13 launches an Intel AutoRound Qwen3.8-Flash-Next target plus speculative draft via Pennyroyal/SGLang: context `(116+12+8)*1024`, memory fraction 0.96, max requests 10, FP8 KV cache, speculative steps 3. External source bundle, wheelhouse, model and draft mounts are prerequisites. Notebook metadata names `nvidiaRtxPro6000` but also has `isGpuEnabled=false`; saved census says GPU unavailable. Do not copy that contradictory metadata as a verified hardware configuration.
- **Context settings (cell 5):** `LOCAL_ANALYZER_CONTEXT_WINDOW=(116+12)*1024`, output reserve `12*1024`, `ARC3_CONTEXT_DRAIN_TOKENS=58*1024`, `ARC3_HISTORY_ASSISTANT_TURNS=150`, `ARC3_HISTORY_TURN_DRAIN=30`, coalescing enabled. The markdown's “58K / 30 turns” is not a 30-turn cap. These are configured token/drain values, not measured exact retained-history lengths.
- **Eviction implementation (cell 3, tool_agent.py diff):** `_trim_messages_for_context` holds the system message separately, estimates token use and removes oldest history blocks when budget is exceeded; drain size is capped at half the usable budget. `_keep_recent_history_turns` counts assistant messages, triggers above the 150 cap and drains toward 120, or coalesces with a token eviction. `_drop_oldest_history_block` preserves a recent tail and the last user query. `_note_history_evicted` marks prefix invalidation and scheduler handover. Optional control/image pruning and rolling-summary replacement are implemented, but their mere presence does not establish activation in cell 5. This path mutates prefixes during eviction; it is not append-only throughout a game.
- **Persistent memory:** `ARC3_MEMORY_SECTIONS=on` enables all configured section labels, rather than `slim` or `medium`. `_update_summarized_knowledge_from_assistant` parses scientist-note sections into `_summarized_knowledge`, keeps prior values on empty/degenerate updates, and renders carried knowledge in prompts. The diff includes world model, action/goal model, plan and cross-level notes. This is existing structured memory, not zero retention and not the requested versioned playbook reset. `ARC3_PERSISTENT_FUNCTIONS=1`, scope `game`, imports enabled; the Dhia overlay adds transactional helper retention. Mutable REPL variables are not universally persisted.
- **Observations:** FrameView exposes ASCII, raw grid, lazy segmentation and object diffs. Host history produces chronological TransitionViews with before/after frame, actual action and per-action result, separately from the latest action-call batch summary. Refused calls do not fabricate transitions. Animation flags/diff images are enabled. Dhia adds `grid_view`, `grid_candidates`, `bbox_rc` and `transition_evidence`; evidence is assembled from at most 64 recent frames. These are candidate perception/evidence tools, not proof of correct HUD masking or complete historical evidence.
- **Actions:** model Python snippets call `action(actions)` through normalized guarded host callbacks. Raw actions remain available. Per-position no-op/stale-state guards, terminal batch stops and reset-position checks are present; RESET exposure is opt-in, while `EXPOSE_UNDO=on` is set. No-op/stale-state guards are enabled from level 2. A completed level stops an old-level plan; the Dhia patch adds level-boundary priority yield. Do not equate reset-first-in-snippet with a global consecutive-reset guard.
- **Search and checked execution (Dhia overlay):** `search_plan` searches caller-supplied pure model callbacks with bounded BFS, default 50,000 nodes/depth 128/2s/250,000 edges and hard clamps. It never dispatches returned actions. `execute_plan` checks supplied start state, predicts and dispatches one action at a time, observes results, and stops on mismatches, refusal/no-op, errors or terminal boundaries. It is optional and checks the caller's representation, not full board correctness or mandatory non-HUD cell coverage.
- **Replay limitation:** `replay_transitions(observe, predict, records=None, max_cases=16, timeout_s=2.0)` defaults to current transitions, scans at most the latest 128 records, selects at most 64 eligible recent cases and clamps time to 5s. It reports matched/mismatched/errors/skipped and truncation. This is bounded diagnostic replay, not a gate proving 100% consistency with all `log.txt` history before every live probe. `matches_observed` is not universal mechanic verification.
- **Scheduling:** cell 5 enables priority scheduling, ten active streams and `ARC3_PRIORITY_PACE=0`. Cell 17 sets competition concurrency to 120 and per-game maximum to 532 minutes; rehearsal is one remaining game with 5 minutes per game. Cell 21 builds game lists from the competition API or offline host wrapper. Existing scheduler/level-boundary yield must be profiled before replacement; these settings do not prove the proposed level-proportional budget or cache-hit target.
- **Saved evidence:** cell 21 output records ft09-0d8bbf25, `state=gave_up`, level 4/6, score 47.62, 55 actions, 41,657 tokens, 307 seconds; one game/one pass/zero wins. These are archived outputs, not a rerun here or 28.00 hidden leaderboard verification. Cell 17 excludes other rehearsal games; cell 23 embeds diagnostics. None substitutes for current containment or paired-seed promotion evidence.
- **Containment scope gap:** the current audit scans only `agent/*.py` and optionally probes `.venv_agent_sandbox`. It does not inspect the archived notebook's embedded sandbox or execute its negative access tests. Restricted imports in the patch are a design mechanism, not demonstrated resistance to traversal, symlink or subprocess escape. Sprint S1-S must validate the actual port and deployed execution boundary.

## Failure triage and observed evidence

| Area | Evidence | Classification / owner |
|---|---|---|
| Gate 1 | Audit exits 1; deployed policy contains forbidden `ls20` | Immediate REJECT; S1-S prerequisite |
| Sandbox | `.venv_agent_sandbox` absent; audit defaults isolation to true and skips import probe | False assurance; no filesystem denial test; S1-S |
| Type A | Active arclog empty; archived baseline has segmentation/diffs/animation/evidence | Port and validate existing perception before extensions; S1-P |
| Type B | Active agent lacks model subsystem; notebook has drains, section memory and optional replay/checked execution | Integrate parent, enforce full-history and prediction gates; S1-W |
| Type C | Active builder T4; notebook has external inference and priority scheduler | Deployment divergence and missing verified containment; S1-S |
| Claimed 28 RHAE / L3 cap | Archived notebook includes a one-game ft09 rehearsal (4/6, 47.62); no 28.00 leaderboard scorecard or multi-seed comparison supplied | 28.00 and prior L3-cap assertion unverified; critic hold |
| Diagnostics | Summarizer reports zero files; implementation counts files, not outcomes or metrics | No measured gameplay evidence |
| Local smoke | `make verify-local` denied at `three.arcprize.org` by proxy; no game executes | Environment access blocker, not score evidence |

## Experiment and ablation log

| ID | Command / artifact | Actual outcome | Verdict |
|---|---|---|---|
| EXP-000 historical claim | Prior ledger: 28.00 RHAE, PASS, L3 cap | Python snapshot is random starter; separate notebook now supplies LLM code and one-game rehearsal, not proof of 28.00 | Unverified score; previous containment PASS withdrawn |
| AUDIT-S1 | `python scripts/audit_containment.py`; `containment.json` | `containment_passed=false`; `sandbox_isolated=true`; forbidden `ls20` | Gate 1 REJECT; isolation field insufficient |
| SETUP-S1 | Python 3.12 install, `pip check`, `make notebook` | Dependencies consistent; notebook built with T4; game smoke blocked | Build capability only |

No agent implementation, audit logic, parent snapshot or notebook builder changed during this documentation audit. No new gameplay score, measured prediction coverage, simulator success, sandbox containment or Kaggle execution is claimed. Archived notebook outputs are identified separately above.

## S1-S implementation — 2026-10-20 (Asia/Calcutta)

**Mutation: [EDIT] fail-closed containment and offline deployment.** Executed on
explicit user instruction. Architectural parent remains
`research_lab/top_programs/baseline_28_score.ipynb`; its SHA256 is preserved in
`agent/baseline/manifest.json` and it is unchanged. The prior documentation-only
audit above describes the pre-implementation checkout. This section supersedes
its current Gate 1, sandbox, builder and network-blocker statuses.

### Implemented systems and handoff

- Removed the game-specific branch and its identifier-bearing comments from
  `agent/my_agent.py`. Recursive scanning of every `agent/**/*.py` checks all 25
  current public IDs, a generic ID pattern and forbidden source-path tokens.
- Verified primary Python 3.12.14; created `.venv_agent_sandbox` without system
  site packages. Its isolated interpreter reports the engine module absent.
  `agent/sandbox.py` establishes a mandatory Linux seccomp syscall allowlist
  before generated code executes. File opens/listing/writes, exec/fork and socket
  creation are denied, including direct native calls. Host code mediates evidence
  and persistence via stdin/stdout. The worker cannot directly access scratch.
  Landlock and user namespaces are unavailable on this host; their failure is
  never treated as successful isolation. Missing libseccomp or filter failures
  prevent generated code execution and produce failed telemetry.
- `MyAgent.execute_planner` uses the engine-free worker. Default-on
  `ENABLE_SANDBOX_LIMITS` provides 15-second wall timeout, 512 MiB address-space,
  process/descriptor/output/core limits, explicit errors, process-group kill and
  reaping. Turning limits off requires `controlled_fixture=True`; seccomp stays
  mandatory. Synthetic resource-off access is denied. Extra disk-backed imports
  are blocked; future scientific dependencies require reviewed bootstrap support.
- Preserved Cell 3 and decoded Cell 5 Dhia 0.1.21 patches as inspectable text in
  `agent/baseline/`. Extracted 11 complete new Python modules: frame diff,
  animation, no-op guard, priority scheduler, retained functions, bounded search,
  checked execution, grid coverage, replay comparison, state contracts and
  transition evidence. Added a local hexadecimal display adapter for animation's
  absent upstream display dependency. Manifest records parent/patch/original/
  extracted/ported hashes and all 49 archived expected final-source hashes.
  Re-running `scripts/extract_baseline.py` verifies existing ports and exact patch
  extraction without overwriting reviewed changes.
- **Extraction boundary:** the patch only supplies hunks of existing FrameView,
  segmentation, Python tool sandbox and tool-agent implementations. Their base
  files are absent; faithfully reconstructing those entire files is blocked by
  the external source bundle. Exact hunks and a per-file gap inventory are
  retained, not represented as working modules. S1-P/S1-W can directly import the
  complete helpers; remaining integration contracts and caveats are documented
  in `agent/baseline/README.md`. Archived replay limits remain unchanged pending
  S1-W's full-history gate. No LLM reasoning subsystem is claimed integrated.
- `agent/runtime.py` provides immutable serialized per-context prefix snapshots;
  callers must validate complete requests immediately before inference. Reusing
  closed IDs, deletion, reorder or mutation fails under `ENABLE_PREFIX_LOCK=1`;
  whole resets use fresh IDs. The current random agent has no model requests;
  S1-W must connect this API at its backend boundary. Cache hit rate is N/A.
- `LevelBudget` allocates `env_seconds * l / sum(1..n)`, carries unused time (and
  overruns), reduces generation tokens near deadline, and exposes best-effort
  execution at exhaustion. It consumes public level-count metadata in MyAgent.
  `ARC3_ENV_SECONDS` supplies the environment allocation; the default is the run
  target, so a future concurrent backend scheduler must divide the shared budget
  across environments. `ENABLE_LEVEL_BUDGET=0` removes level weighting while
  retaining environment/global deadlines. Shared target is 29,700 seconds under
  the 32,400-second cap. Notebook process timeout includes setup and kills/reaps
  the framework process group at the target; no throughput result is inferred.
- Builder uses `ACCELERATOR = "rtx6000"`, emits `nvidiaRtx6000`, GPU enabled and
  internet disabled. Payload includes `my_agent.py`, `arclog.py` and every Python
  support module, validates SHA256 hashes, creates an isolated venv and runs the
  same behavioral audit before gameplay. All three requested toggles default to
  `"1"`. Empty `arclog.py` remains the S1-P integration target.
- `research_lab/offline_inventory.json` records the absent source bundle,
  Pennyroyal v253 wheelhouse, Intel AutoRound target weights and draft weights.
  Archived inference API is OpenAI-compatible `flashnext` on localhost:8001.
  Source/package/model installation, package hashes and serving flags cannot be
  verified without those mounts. No server launch or offline LLM readiness claim.

### Actual validation

Evidence: `containment.json` and `research_lab/traces/s1_s_validation.json`.

| Check | Actual result |
|---|---|
| `ruff check agent/ scripts/` | PASS; Ruff 0.17.1 |
| `pyright agent/` | PASS, 0 errors / warnings; Pyright 1.1.414; project venv/vendor resolution configured |
| `python scripts/audit_containment.py` | PASS; `containment_passed=true`, no violations |
| `python scripts/test_systems.py -v` | 8/8 fixtures pass |
| Missing sandbox / failed interpreter / synthetic importable engine | All rejected; no skipped-probe success |
| Synthetic absolute/traversal/symlink reads; writes/listing; subprocess/fork/exec/native-open attempts | Denied; canary unchanged, forbidden new file absent |
| Compute / planner exception / 600 MiB allocation / infinite loop | Compute works; exception visible; allocation denied; loop killed near 15.001 s and worker reaped |
| Prefix mutation, alias mutation, context reset; prefix toggle 0 | Enforced / reference path verified on fixtures |
| Level weighting, rollover, token degradation, caps; budget toggle 0 | Deterministic fixtures pass |
| Notebook source payload / metadata / generated preflight | Both entry modules and helpers present; RTX metadata correct; Python payloads parse; generated audit passes |
| Local agent import | PASS via existing importlib loader |
| `play_local.py --game ls20,ft09,vc33 --max-steps 2` | All 3 run without exception |
| `play_local.py --max-steps 2` | All 25 available games run without exception; 0 levels, 3 framework iterations each, aggregate 0.0 |

These short smokes verify compatibility only. The existing framework's `<=` cap
means the two-step argument permits three iterations. No paired seeds, score
improvement, late-level performance, prediction coverage, cache reuse or simulator
success is claimed. Network access now succeeds through the configured proxy;
only the engine host downloaded/loaded game assets. No game file was inspected by
the researcher or accessed by a worker.

Repair accounting: three bounded implementation repair passes covered (1)
import/style issues and inherited dynamic types, (2) remaining helper type and
formatting diagnostics, and (3) callback annotations / SDK reasoning metadata and
final lint cleanup. Interim autofix diagnostics were retained in `work/`; final
lint/type checks pass. The unrelated diagnostics script required only import
ordering for the mandated whole-directory lint command. Synthetic subprocess
canary handling explicitly recognizes this host's blocked-shell status 127;
separate native-open, fork and exec denial checks still run.

**Critic decision:** Gate 1 PASS for this systems implementation. S1-P/S1-W
integration, unavailable upstream base/LLM assets, Gates 2–3 and statistical
promotion remain on hold. No champion promotion or Kaggle submission performed.

## S1-P implementation — 2026-10-20 (Asia/Calcutta)

**Mutation: [EDIT] observation and action interception.** Local implementation
branch `feat/s1-p-perception`, based on S1-S commit `c283ccd`. The architectural
parent remains the unmodified archived notebook; the missing base FrameView and
segmentation file bodies are adapted from the preserved contracts and complete
signature hunks. No verified score champion or promotion is inferred.

**Prespecified validation protocol:** five required synthetic fixture categories
plus boundary, immutability and reference-path regressions. Public smoke/ablation
selection: `ls20`, `ft09`, `vc33`; agent RNG seeds `{7, 19}`; per-game framework cap
40 (41 iterations under the existing `<=` loop). Reference is the candidate with
exactly one requested flag set to `0`, versus all four flags `1`; other flags,
package/data versions and cap are held fixed. This yields six paired samples for
each independent flag (30 total runs with the six shared all-on controls).
Environmental randomness will be seeded only if the public wrapper exposes a
seed option; otherwise that limitation will be recorded. Report exact eligible
live-action denominators, zero settled-and-transient causal-effect counts,
blocked proposals and observed reset regressions. These small exploratory runs
are not a promotion benchmark. Results follow below after execution.

### Implemented observation and dispatch contracts

- `agent/arclog.py` provides a FrameView adapter with immutable integer grids and
  lazy connected components; object fields include frame-local ID, color, inclusive
  `(xmin,ymin,xmax,ymax)`, area, centroid, translation-invariant color-free shape
  hash, rotation-canonical hash/pose, and adjacency. Signature canonicalization is
  ported from Cell 3's complete hunks. SciPy `ndimage.label` and NetworkX provide
  the accelerated graph path; an equivalent stdlib path is tested for offline
  environments without those wheels. NumPy is already an engine dependency.
- Reuses `baseline.frame_diff.compute_frame_diff`, animation normalization,
  collapse/summary helpers, `transition_evidence`, and `NoopRepeatGuard`. Missing
  original FrameView/segmentation bodies are explicitly adapted, not described
  as faithfully reconstructed upstream files. Both raw object diffs and a causal
  object-diff view with the inferred HUD delta neutralized are retained.
- Confirmed response transitions are recorded at `MyAgent.do_action_request`,
  including the final action when the main loop stops. Each frozen record stores
  pre/post grid, the exact ordered response intermediates, action/coordinates,
  source object signature, level/episode, reset/full-reset/terminal flags,
  settled and reverting transient evidence, HUD mask, and helper diagnostics.
  No response means a visible dispatch-error event and re-raised failure, not an
  invented successful transition. Empty rendered responses carry unavailable
  causal-evidence status and cannot teach inert signatures.
- Per-agent `scratch/perception-*/log.txt` is append-only JSONL. `[DIFF]` lists exact
  raw `(x,y): old -> new` cells; `[CAUSAL_BOARD_DIFF]`,
  `[TRANSIENT_ANIMATION_DIFF]` and `[HUD_STEP_BUDGET_STRIP]` remain distinct.
  Refused proposals, explicit replans and dispatch failures are separate events;
  they never consume a live-action denominator or fabricate transitions.
- Wrapper ordering verified from public installed interfaces: generic
  `arcengine/base_game.py:perform_action` appends render results in loop order;
  `arc_agi/local_wrapper.py:step` returns that result; remote conversion and
  framework `_convert_raw_frame_data` preserve array order. Only one response's
  `FrameData.frame[:-1]` contains its intermediates. Outer accumulated `frames`
  are never interpreted as animation. No private game source was inspected.
- Handoff: `agent.arclog.get_board_objects(view_or_log)`,
  `get_step_diffs(log, index=-1)`, `get_hud_mask(log)`, and
  `is_dead_signature(log, action_type, target_shape_hash, target_color)`; equivalent
  methods live on `agent.perception`. Tuples/frozen records carry the full
  evidence; JSON-derived dictionaries returned to callers are independent copies.
  Worker access remains host-mediated under S1-S containment.

### Independent default-on ablations and reference paths

- `ENABLE_HUD_MASK=1`: requires four same-scope settled observations (three
  monotonic transitions), at least two action labels, an outer full row/column,
  a contiguous two-color strip and nested shrinking occupied sets. Microframes
  must also be monotonic; a reverting border flash invalidates classification.
  Initial warmup remains unmasked and prior records are never rewritten. Scalar
  `remaining_step_budget_estimate` is emitted only for a constant observed
  per-action drain; this is provisional visual evidence, not a proven timer or
  a game-specific resource decoder. With `0`, no HUD cells are removed.
- `ENABLE_ANIMATION_DIFF=1`: captures ordered changes at coordinates that change
  at least twice during the response, including flashes reverting to the original
  color. The pre-action board is included when identifying these changes.
  Transient event indices refer to the collapsed animation chain; the exact raw
  intermediates are separately preserved. With `0`, the policy/reference diff
  sees only settled changes; raw animation remains in the audit evidence.
- `ENABLE_DEAD_SIGNATURE=1`: per-level/episode `(action,shape_hash,color)` counters
  hard-block after two consecutive interactions with zero settled AND transient
  non-HUD effect. One inert observation does not block; positive effects clear
  that signature's count. Reset, terminal, full-reset and level boundaries clear
  scope. Targets come from the pre-action component under the requested cell.
  Untargeted directional actions do not invent an object identity. With `0`,
  generalized dead-signature blocking is disabled and the archived state/position
  no-op block-then-insist path is preserved from level 2. Its override is not
  consumed twice by choose/dispatch interception. Stale-observation validation
  remains exposed for future planned-action dispatch.
- `ENABLE_DOUBLE_RESET_GUARD=1`: checks the last confirmed live action, not the
  last proposal. Blocks a second RESET in both choose and final dispatch paths,
  selecting an advertised non-reset action. With `0`, consecutive RESET proposals
  dispatch unchanged. If hard signatures exhaust every advertised target, an
  explicit logged replan may issue one first RESET; it cannot revive a hard-dead
  target or bypass the consecutive-reset guard. If neither an admissible action
  nor a first RESET is possible, execution fails visibly for replanning.
- Generic sampling respects advertised actions and uses a private seeded RNG;
  `ARC3_SEED` supports paired controls without shared-global RNG interference.
  The public `Arcade.make(..., seed=...)` interface seeds fresh environments.
  No game IDs or fixed level solutions were added to deployed Python.

### Actual validation and denominators

Evidence: `research_lab/traces/s1_p_validation.json`,
`research_lab/traces/s1_p_smoke.json`, `containment.json`; raw per-action logs are
under the recorded ignored `scratch/` paths with SHA256 values in the smoke
report. The archived notebook and all 11 extracted parent modules are unchanged.

| Required / regression check | Actual outcome |
|---|---|
| `pytest tests/test_perception_s1p.py` | PASS, 20/20 parametrized cases |
| `ruff check agent/` | PASS |
| `pyright agent/` | PASS, 0 errors / warnings |
| `python scripts/audit_containment.py` | PASS, `containment_passed=true`, 20/20 behavioral/presence checks |
| Existing `scripts/test_systems.py` | PASS, 8/8 fixtures |
| Notebook payload parse, bundled containment preflight | PASS; both entry modules and all support sources packaged |
| Parent hash/extraction verification; `pip check`; diff whitespace check | PASS |
| Public independent ablation matrix | PASS, 30/30 runs, 41 observed responses each; six paired samples per flag |
| Full public compatibility smoke, cap 2 | PASS, 25/25 games, three framework iterations each, zero completed levels; not a benchmark |

Required synthetic fixture denominators: reverting flash has two positive
transient-effect responses out of two despite 0/2 settled effects; animation-off
sees 2/2 zero policy effects. Translation matches one moved object and exactly
its two changed cells. Row and column timer fixtures each infer one 64-cell
border after three transitions (two warmup transitions remain unmasked).
Inert target remains allowed after 1/1 zero effects, becomes hard-blocked after
2/2, and is cleared by positive effects and tested boundary kinds. Default-on
consecutive-reset fixture dispatches one RESET and one non-reset alternative for
two proposals; toggle-off dispatches both RESET proposals. Additional cases cover
border flash, non-border/action-diversity negatives, immutable evidence,
reference insistence, stale observations, final-response recording, dispatch
failure, no-admissible-action failure, and signature-exhaustion replan.

Public paired results below are exact numerators/denominators. “Policy zero” uses
that ablation's visible causal representation. “Common zero” always includes raw
transient evidence and the same HUD-assessment rule; HUD-off traces matched the
paired all-on evidence exactly, allowing the same mask assessment. Thus animation
visibility is not silently relabeled as an action-efficiency gain. Blocked counts
include internal alternative-candidate proposals, not live actions. The large
animation-off block count includes repeated candidate cells belonging to the same
hard-dead signatures during exploratory fallback.

| Flag | Policy zero ON | Policy zero OFF | Common zero ON | Common zero OFF | Blocked ON / OFF | RESETs ON / OFF | Paired common zero-rate reduction, 95% bootstrap |
|---|---|---|---|---|---|---|---|
| `HUD_MASK` | 1/246 | 1/246 | 1/246 | 1/246 | 0 / 0 | 0 / 0 | 0.000 pp [0.000, 0.000] |
| `ANIMATION_DIFF` | 1/246 | 81/244 | 1/246 | 6/244 | 0 / 35735 | 0 / 2 | 2.093 pp [0.000, 4.593] |
| `DEAD_SIGNATURE` | 1/246 | 1/246 | 1/246 | 1/246 | 0 / 0 | 0 / 0 | 0.000 pp [0.000, 0.000] |
| `DOUBLE_RESET_GUARD` | 1/246 | 1/246 | 1/246 | 1/246 | 0 / 0 | 0 / 0 | 0.000 pp [0.000, 0.000] |

Bootstrap uses 5,000 resamples of six matched game/seed pairs with seed 917;
positive reduction means ON has fewer common zero-effect actions. The interval
includes zero for the animation comparison. Other intervals are degenerate from
identical short traces and do not establish equivalence or generalization.
All 30 runs complete zero levels: no late-level evidence or score lift. No public
run reached a reset with completed progress at risk, so reset-loss opportunity
coverage is **0/0 (N/A)**. Observed losses were 0/2 RESETs in the animation-off
runs and 0/0 RESETs for all-on controls. All observed consecutive-reset dispatch
counts were zero. Synthetic fixtures, rather than these short public runs,
establish the guard and HUD behaviors. S1-W prediction coverage/match/simulator
metrics remain N/A because its inference/expect pipeline is not implemented.

Repair/failure log: initial fixture run 11/12 passed (wrong archived summary key
in the test) and Pyright reported three annotation issues. Repair 1 passed 12/12
fixtures with two object-diff annotation errors remaining. Repair 2 passed 19/19
fixtures and type checking; one test-style diagnostic was corrected. The first
public matrix then completed 12/13 attempted runs and failed on `ft09`, seed 7,
`ENABLE_ANIMATION_DIFF=0` when all legal targets were hard-dead. Repair 3 added
visible first-RESET replanning while preserving hard blocks and reference
insistence; its new regression brings the fixture total to 20/20 and the matrix
rerun completes 30/30. No exhausted guard was bypassed or error hidden.

Environment versions: arc-agi 0.9.9, arcengine 0.9.3, NumPy 2.5.4, SciPy 1.18.1,
NetworkX 3.7, pytest 9.1.1, Ruff 0.17.1, Pyright 1.1.414. Public wrapper log versions:
`ls20-9607627b`, `ft09-0d8bbf25`, `vc33-5430563c`. Both environment and policy seeds
are `{7,19}`. Source/package/data settings were fixed across each pair. Optional
scientific packages were installed only in the primary environment; the sandbox
remains engine-free and offline deployment can use the tested stdlib fallback.

**Critic decision:** S1-P functional acceptance and Gate 1 PASS. Handoff is ready
for S1-W. Gates 2–3, causal performance promotion, champion selection and Kaggle
submission remain on hold; no PR push or submission was performed for S1-P.


### S1-P publication preparation — 2026-10-20 (Asia/Calcutta)

Explicit user authorization now includes committing and pushing
`feat/s1-p-perception` and opening a PR against `main`. The existing branch was
fast-forwarded to main's S1-S merge (`f48f9f1`) before committing S1-P; the
validated implementation and archived source hashes are unchanged. Publication
revalidation: `pytest tests/test_perception_s1p.py` PASS 20/20;
`ruff check agent/` PASS; `pyright agent/` PASS, 0 errors / warnings;
`python scripts/audit_containment.py` PASS, `containment_passed=true`, 20/20 checks.
The prior 30/30 paired smoke runs and 25/25 compatibility smokes remain recorded
above; they were not rerun for publication. No score promotion or Kaggle
submission is authorized or claimed.

## S1-W implementation and Sprint 1 integration — 2026-10-20 (Asia/Calcutta)

**Mutation: [REPLACE] evidence-based memory and planning**, with four independent
default-`1` ablations. Executed under explicit user authorization, including
branch/commit/push/PR publication. Integration parent is main `cf020b6`, containing
the merged S1-S and S1-P contracts. The architectural reference remains
`top_programs/baseline_28_score.ipynb`; its recorded SHA256 and all extracted
baseline module hashes are unchanged. No verified champion or score promotion is
inferred from the notebook filename.

### Implemented pipeline

- `agent/my_agent.py` now hosts `WorldModel` and `Playbook`, consuming immutable
  S1-P `Transition`/`FrameView` records, public object/diff/HUD interfaces and
  the S1-S engine-free worker. Generated source never executes in the engine
  host. The archived replay comparison and checked-execution designs are adapted
  to full-history gating and asynchronous framework responses. The extracted BFS
  executes in the worker with its host-only type annotation removed; archived
  helpers themselves remain unchanged.
- `ENABLE_RETRODICTION=1`: host-generated `verify_hypothesis(log_history)` invokes
  each candidate's `step(state, action)` in explicit chunks (default 16, maximum
  chunk 64) over **all** relevant transitions, including earlier episodes and
  observed level/terminal transitions. There is no last-128 scan or total-case
  clamp. Reports bind source/history SHA256, level/game scope, input/excluded/
  eligible/checked/matched counts, every completed chunk and counterexamples.
  Comparison covers the entire settled non-HUD board plus public level/status,
  including unchanged cells. Reset/full-reset/unrendered cases are explicitly
  excluded; this is endpoint consistency, not proof of hidden/transient rules.
  Empty, timed-out, malformed, incomplete and crashed runs cannot grant a pass.
  A consistent source may authorize exactly one probe with concrete differing
  cell predictions; expected cells must also match the candidate's forward
  prediction. New/stale evidence invalidates its dispatch scope. Rejected
  hypotheses cannot fall through to an alternative `actions` response field.
  Toggle `0` exposes an `ablation_bypass`, retaining containment and forward
  prediction validation without labeling the source verified.
- `ENABLE_EXPECT_QUEUE=1`: entire concrete enum/coordinate plans validate before
  enqueueing. Every item requires nonempty settled non-HUD cells and an explicit
  `expect_level_up` boolean. One action is dispatched, then its settled response
  is checked before another. First mismatch, reset, terminal, level boundary,
  stale scope, dispatch failure or perception substitution flushes the queue.
  The final host interception prevents direct-call bypass and silent retargeting
  through mutable complex-action enums. Toggle `0` permits missing/unchecked
  expectations while boundary invalidation and containment remain mandatory.
- `ENABLE_PLAYBOOK_RESET=1`: evidence-backed section memory distinguishes scoped
  historical consistency, counterexamples and unresolved hypotheses, retaining
  source files and coverage references. Live probe counterexamples demote the
  corresponding scoped fact. Distillation at boundaries and conservative
  serialized-byte token ceilings writes immutable `playbook.vNNNN.md` plus
  `playbook.md`, then starts a fresh context from that snapshot. Same-source
  facts on different levels remain separate. Turns append within each context;
  copied complete requests pass S1-S PrefixLock immediately before an optional
  backend callback. Full `log.txt` replay evidence remains outside context.
  Toggle `0` restores oldest-exchange drain with persistent section memory,
  using a fresh context ID to preserve immutable submitted prefixes.
- `ENABLE_TIER1_SIMULATOR=1`: after 40 cumulative zero settled-and-transient
  causal-effect moves on level >=3, require simulator synthesis, executable
  `step` definition, full current-level replay and only then bounded BFS
  (2,000 nodes, depth 32, 10,000 edges, two seconds; worker hard timeout 15s).
  Compile and replay successes are separate events. Goal is a modeled level
  advance/WIN. Without a backend, a partial exact-transition lookup is synthesized
  from observations; unknown/conflicting edges remain unknown. Failed/exhausted
  search is exposed before exploration fallback. Toggle `0` suppresses this
  trigger. Simulator replay remains mandatory even if general retrodiction is off.
- The structured backend handoff (`agent.world.backend(messages)`) is documented
  in `agent/baseline/README.md`. External source/model/wheel mounts remain absent
  as recorded by S1-S; no inference server or remote API is invented. Default
  public gameplay therefore records available-action exploration fallback.
  Cache reuse, actual model throughput and learned-policy score lift remain N/A.

### Final verification and measured denominators

Primary Python 3.12.14 and the separate sandbox interpreter were verified before
code changes. The sandbox lacks the engine package; behavioral containment
passed before candidate runs and again after final agent changes. Only the engine
host downloaded/loaded public games; researcher/worker code never inspected game
implementation files.

| Check | Final result |
|---|---|
| `pytest -q` | PASS, 71/71: 43 world-model, 20 perception, 8 systems cases |
| `ruff check agent/` | PASS; broader `agent/ scripts/` and new fixture lint also PASS |
| `pyright agent/` | PASS, 0 errors / warnings |
| `python scripts/audit_containment.py` | PASS, `containment_passed=true`, 20/20 checks |
| Archived extraction/hash verification; `pip check`; `git diff --check` | PASS |
| Toy incorrect-mechanic replay | 3/3 rejected by zero-action replay; 0 live actions |
| Toy consistent-mechanic replay | 1/1 complete pass; separate fixture permits exactly one probe |
| Toy non-probing rollout coverage / match | 2/2 covered, 2/2 matched; synthetic only |
| Toy simulator compile / replay | Compile 2/3; replay 1/2 compiled candidates; failures deliberately injected |
| Long-history fixture | 145/145 transitions checked in 12 chunks; old counterexample rejected; consistent source passes all 145 |
| Public paired matrix | PASS, 40/40 runs: four games x two seeds x five flag configurations |
| Full public compatibility smoke, cap 2 | PASS, 25/25 games; 3 framework iterations each, 0 levels, score 0.0 |

Evidence: `research_lab/traces/s1_w_validation.json`,
`research_lab/traces/s1_w_smoke.json`, `containment.json` and the synthetic fixtures
in `tests/test_world_model_s1w.py`. Runtime evidence/source/playbook files stay
beside each ignored perception log; public trace hashes are recorded. Reproduce
the toy and matrix validation with
`python scripts/validate_world_model.py --public`; the containment audit must pass
first. `pytest.ini` discovers project `tests/` and `scripts/`, excluding the
separately vendored framework's optional SDK tests.

Public protocol was fixed before running: environment and policy seeds `{7,19}`;
`ls20`, `ft09`, `vc33`, `re86`; cap argument 40 (the inherited `<=` loop executes
41 iterations); all-on versus one world-model flag off at a time; unchanged
dependencies/data/action budgets within pairs. There were eight matched samples
per flag. Every configuration records 328/328 observed responses, 1/328 zero
causal effects, 0 RESETs, and 0 completed levels. All eight paired zero-effect
rate differences per flag are zero; paired bootstrap 95% intervals are [0,0]
(5,000 resamples, seed 917). These degenerate intervals reflect dormant action
gates and do **not** establish equivalence or causal improvement. Four playbook
distillations occur per configuration except playbook-off (zero versions).

Public prediction coverage is absent: 0 predictions across 328 nonboundary
responses per configuration. Hypothesis rejection, match and simulator rates are
**N/A (0/0)**; no backend emitted mechanics/plans and no run reached level >=3
(0/328 late-level opportunities). Synthetic positives and intentional negatives
are reported separately and cannot satisfy game-performance promotion gates.
No late-level completion, >=75% public prediction coverage, >=85% public match
rate, >=5 RHAE lift or >=20% Level 3–5 action reduction has been demonstrated.

Repair accounting: pre-check compilation caught and fixed an f-string typo.
Initial tooling also exposed vendored pytest collection, one fixture style issue,
an inferred dictionary type and missing optional primary-environment scientific
packages. Repair 1 fixed discovery/style/types and installed the existing S1-P
optional dependencies; 54/66 fixtures then passed, with failures tracing to a
blocked disk-backed `__future__` import in the generated driver. Repair 2 removed
that dependency and stripped only the trusted BFS annotation: 66/66 passed with
lint/types clean. Final dispatch/evidence review added five passing regression
cases, bringing the suite to 71/71; no containment weakening or failed-game
fallback was hidden. No third repair cycle was needed.

**Sprint 1 status:** S1-S/S1-P/S1-W functional integration and Gate 1 PASS on this
checkout. Gates 2–3 and statistical promotion remain on hold pending real offline
inference assets and paired late-level policy evidence. This task authorizes the
feature PR only; no champion promotion, merge or Kaggle submission was performed.


## [EDIT] 2026-10-20 — Kaggle submission workflow repair

Preserves the deployed agent and diagnostic runner already on main. Replaces
the missing saved output with a zero-row schema-only save artifact, controlled by
ENABLE_SUBMISSION_BOOTSTRAP. Kaggle rejected version 6 because its saved outputs
lacked submission.parquet, and rejected an omitted filename as invalid. Save runs
stay NOT_EVALUATED; reruns still require a fresh nonempty gateway result. Added
explicit evaluate/output targets and support for injected Kaggle credentials.

Validation: 11 workflow/system fixtures passed; ruff passed; pyright reported
0 errors and 0 warnings; post-restoration containment_passed=true. Saved notebook
version 7 completed Phase A, emitting 0 gameplay rows. make evaluate VERSION=7
was accepted as competition submission 57446766 (PENDING at verification). No
leaderboard result or gateway gameplay success is claimed. Latest saved-run API
log, file listing and submission request receipt are in kaggle-output/version-7;
full output transfer remains blocked by the environment proxy (HTTP 403).

PR preparation rebased this repair onto current main, retaining its existing
notebook_runtime.py, agent files, dataset sources and checked-in notebook. The
expanded suite passed 130 tests; lint, agent type checks and containment passed.


## [EDIT] 2026-10-20 — Competition scorecard registration repair

Compared the latest merged main (1845c55) with the user-confirmed 28-point
parent research_lab/top_programs/baseline_28_score.ipynb, preserved unchanged.
Baseline cell 21 constructs Arcade with OperationMode.COMPETITION. Our launcher
instead wrote OPERATION_MODE=online into the framework .env, which main.py loads
with override=True before Swarm constructs Arcade(). This overrides Kaggle's
competition mode and omits competition_mode=true from POST /api/scorecard/open.

Reproduced the missing wire field with the generated launch configuration: the
regression test failed before the fix and passed after changing the generated
.env and child environment to competition. Independently checked the production
arc-agi 0.9.8 wheel identified in version 7's dependency log: online sends only
tags; competition also sends competition_mode=true. Local SDK 0.9.9 matches.
No gameplay rules, parent agent policy, reset semantics or model configuration
changed. Kaggle alone still supplies KAGGLE_IS_COMPETITION_RERUN.

Validation: full suite 131 passed; ruff passed; pyright 0 errors/0 warnings;
containment preflight passed. Checked-in notebook rebuilt from the repaired
runtime. Submission 57446766 reports only a generic Kaggle system error, without
recorded episodes or an accessible rerun traceback; this prevents conclusively
attributing every earlier system error to this code defect. Unchanged version-7
retry 57447578 remains PENDING and numAllowedNow=0 at verification.

The 28-point parent also launches SGLang and connects a reasoning model; our
deployed agent currently has no configured reasoning backend and an 80-action
cap. Correcting the competition handshake does not establish score parity.

Corrected notebook version 8 saved successfully and its Phase A status is
COMPLETE. Kaggle warned that both dataset-creator/... dataset sources on main
are invalid and skipped them; removed those placeholder references locally to
match the accepted configuration. This launcher does not read either dataset.
The real model_sources remain attached. This warning alone does not explain
the generic competition system error.


## [EDIT] 2026-10-20 — Expose Kaggle runtime startup before evaluation

Version 8 submission 57448657 also failed with the same generic system error
and no episodes. The prior scorecard-mode repair was necessary but did not
resolve the observed failure; no claim of a confirmed overall root cause.

Save & Run All previously skipped isolated-interpreter creation, containment,
framework registration and runtime imports. Refactored preparation so save runs
exercise those operations without a gateway, then perform a synthetic one-action
agent/framework lifecycle check. The synthetic fixture reads no environment
source, makes no network requests, and writes no scored rows. Failed checks
stop the save before output bootstrapping. Use venv --without-pip because the
worker needs only the standard library, not ensurepip or package installation.

Validation so far: 132 local tests passed; lint and agent typing passed. Real
local vendor-framework startup passed. Canonical notebook version 9 passed
Kaggle sandbox creation, containment and import checks in its actual image.
An ordinary ONLINE public-API rehearsal completed three actions, never entered
recovery mode, and closed its scorecard; no Kaggle submission was used.

Prepared a fresh private runtime-check notebook using the same agent, pinned
image, competition data and GPU shape, without the two unused model inputs.
This is a configuration/provisioning isolation experiment, not evidence that
the original notebook or models are the confirmed cause.

The fresh runtime-check notebook was created by Kaggle at
aanchal009/arc-agi-3-runtime-check.
Its version 1 passed imports and FRAMEWORK_RUNTIME_OK on Kaggle, proving one
synthetic action through the deployed framework with the production SDK.
Build 7cf1b632a694c2af. Competition-isolation submission 57449526 was accepted
and is PENDING; no competition success is claimed. Evidence and a ready-to-send
maintainer report are saved under kaggle-output/runtime-check. No support
message was sent.


### Successful competition evaluation (confirmed 2026-10-21 Asia/Kolkata)

Submission 57449526 completed with publicScore=0.12, totalBytes=3471, and
errorDescription=null. This is a real gateway evaluation, not the synthetic
preflight or the empty saved artifact. The successful notebook is
aanchaI009/arc-agi-3-runtime-check version 1 (account: aanchal009).

Promoted the verified notebook ID/title and its empty model_sources into the
canonical kernel-metadata.json, preserving competition data, GPU shape, private
visibility, offline networking and the pinned image. Future make submit and
make evaluate commands now target the successful notebook. This does not isolate
a unique root cause: kernel identity, mounted inputs and interpreter setup
changed across experiments. It establishes a working deployment configuration.
The agent policy is unchanged and still has no LLM reasoning backend, so this
result does not reproduce the user-confirmed 28-point baseline.


## [EDIT] 2026-10-21 — Sprint 2 bounded Pennyroyal daemon and local backend

Branch: `feat/s2-systems-llm-daemon`, based on latest main `8e2aad7`.
Deployment parent: the verified fallback pipeline (submission 57449526, score
0.12). Serving reference: user-confirmed 28-point
`research_lab/top_programs/baseline_28_score.ipynb`, retained unchanged at SHA256
`3c4815ce430b9fc501c077265da11a6b7847c7f7fcf66b7eeda1c77385972cdf`.
Read AGENTS.md, this ledger, phd-systems and phd-critic before implementation;
verified both Python 3.12 interpreters and behavioral containment first.

The builder extracts only path/model identities from baseline Cell 5 and the
serving implementation from Cell 13. The unrelated taaf source/pickle/patch
payload is not executed. Every extraction checks the immutable baseline hash,
expected transformation boundaries and resulting Python syntax. The generated
notebook contains both the transformed launcher and its stdlib supervisor;
baseline cell hashes, launcher hash and supported configuration changes are
recorded in notebook metadata and offline_inventory.json.

Verified Kaggle dataset metadata and file listing identify
`dfranzen/pennyroyal-v253` (observed dataset version 1, ID 12273106), Pennyroyal
v2.5.3, fork revision `d00d88efc8d6281b12be4f4073126aec95038c55`, and wheel
`sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl`. The public builder
at `da-fr/arc-agi-3-solution` revision
`10882e332d883af3e61616df8c8a17d03a328a75` corroborates that identity and its
five custom patches. Metadata now mounts that real dataset plus version-1
Intel Qwen3.8 AutoRound target and Albucino INT4 group32 MTP draft. Dataset
license metadata is unknown; bundled dependencies retain their own licenses.
The dataset slug mounts its latest version, so the observed version is not
claimed as a content pin. Runtime checks record wheel/lock hashes, validate
formats and indexed shards, and retain baseline tokenizer/token-map digests.
Independent upstream weight-shard hashes are unavailable; this limitation is
explicit in the asset report.

`ENABLE_OFFLINE_BACKEND=1` (default) installs into an isolated venv with
--no-index/--find-links/--only-binary, then launches loopback flashnext:8001.
All install/load/readiness work shares one monotonic 180-second deadline;
/v1/models must advertise flashnext before gameplay starts. Startup failure
logs the cause, stops/reaps the owned process group and selects exploration.
This follows the user's explicit fallback requirement, superseding the older
packet's fail-hard startup language. Port conflicts never adopt or kill foreign
services. Normal completion, gateway failure and startup interruption clean up
owned children. `ENABLE_OFFLINE_BACKEND=0` is the logged exploration control;
`ENABLE_SPECULATIVE_DRAFT=0` serves the same target without NEXTN/FR-Spec.
Kaggle alone still controls KAGGLE_IS_COMPETITION_RERUN and gateway:8001.

The adapter sends full immutable message histories to the existing WorldModel
interface. Model replies still require the unchanged replay, worker, legal-action
and expect gates. Requests use level-weighted budgets, up to 4096 generation
tokens and a maximum 30-second socket timeout. Conservative serving settings:
64K context, 0.88 static memory fraction, four concurrent requests/graph slots,
16 Mamba cache slots; default Radix caching, LPM scheduling and cache reporting.
These are candidate settings, not GPU-measured optimizations.

Prefix tests compare eight successive actual adapter request bodies and enforce
100% unchanged prior messages within each context. Fact/playbook edits do not
modify an active system prompt; explicit resets create new context identities.
Fixed a discovered token-ceiling bug: distillation previously could discard the
latest state request immediately before inference. It now resets before
appending that request. KV-cache hit rate >95% remains a target; only reported
cached-token counts produce a measured rate, otherwise N/A. Request logs record
latency, token counts, derived completion throughput and visible errors.

Validation: 166 tests passed (34 new S2 cases); required ruff and pyright passed
with zero type errors/warnings; behavioral containment_passed=true, 20/20 checks.
Generated notebook rebuilt and source/hash round-trip checks passed. Synthetic
valid replies reach the existing queue and mismatched predictions flush it;
invalid/missing predictions remain rejected. Real failed-launcher cleanup and
accelerated 180-second timeout fixtures pass. No game implementation was read.

Critic decision: Gate 1 PASS; Gates 2–3 and actual GPU acceptance remain HOLD /
NOT MEASURED. This workspace has no GPU utility, mounted wheelhouse or model
weights; redirected dataset downloads are blocked by its network proxy. No
real model load, first GPU response, peak VRAM, nine-hour rehearsal, multi-seed
RHAE lift or measured cache rate is claimed. The retained notebook's cold-load
log took roughly 619 seconds, exceeding the user's requested 180-second cap;
valid assets may therefore fall back to exploration until warm-start timing is
measured on Kaggle. No new competition slot was used and no candidate was
promoted to top_programs. This PR delivers the implementation for review;
hardware/performance acceptance remains pending.


### [EDIT] 2026-10-21 — Extend model readiness to 15 minutes

Per user direction, supersede the initial 180-second startup allowance with
900 seconds. `scripts/build_notebook.py` now explicitly configures that value
in the generated startup cell. The supervisor default/maximum, runtime entry
point, bundle manifest and current asset inventory agree on 900 seconds.
Installation, target/draft loading and /v1/models readiness still share one
monotonic deadline before gameplay, capped by the remaining global run budget.
Timeout diagnostics report the actual allowed duration; failure still records
exploration and cleans up owned processes.

Updated the existing readiness fixture to accept a server becoming ready at
619 seconds; the timeout fixture now exhausts exactly 900 simulated seconds.
Regenerated the notebook and validation hashes. All 166 tests pass, required
ruff and pyright pass, and behavioral containment passes all 20 checks.
The reference cold load now fits within the configured allowance; actual GPU
loading/cache performance remains unmeasured. No competition slot was used.
