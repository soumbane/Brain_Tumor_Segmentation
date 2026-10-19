# RUNBOOK

Every command, in order, with its gate. Nothing here has been executed yet except
environment setup and archive extraction (see **Status** at the bottom).

All commands run from the repo root. The venv is `.venv` (Python 3.12.14).

---

## 0. Environment

Already done, but for a fresh machine:

```powershell
# uv, then a real Python (the only python on PATH is the Microsoft Store shim)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "C:\Users\$env:USERNAME\.local\bin;$env:Path"

# Corporate TLS inspection breaks uv's bundled cert bundle; use the OS store.
$env:UV_SYSTEM_CERTS = "1"

uv python install 3.12
uv venv --python 3.12
uv pip install -e .

# Local CPU torch + MONAI + wandb, for authoring and CPU smoke tests only.
# All real training happens on SPCS.
uv pip install -e ".[torch]" --index-strategy unsafe-best-match `
  --extra-index-url https://download.pytorch.org/whl/cpu

# W&B: log in once for online mode. Offline mode needs no login.
wandb login

# Official lesion-wise metric: git clone only, there is no pip package.
git clone https://github.com/rachitsaluja/BraTS-2023-Metrics.git external/brats_metrics
```

**Snowflake launch gates:** project stages/repository and suspended
`CARE_ML_RSRCH_GPU_M` were observed under `SFK_CARE_ML_RSRCH_ADM`. EAI inspection
is blocked by Restricted Session Scope; use `/guardrails`, not an admin-role
workaround. Secret/entity, telemetry, credits and paid-run approval remain
unresolved. See sections 4-6 for the current SPCS workflow.

---

## 1. Data: extract, validate, split

```powershell
# 1a. Extract to C:\data\brats2023 (NOT OneDrive -- the script refuses synced paths)
.venv\Scripts\python.exe -m brats.data.extract

# 1b. Manifest + integrity assertions. THE label-3-vs-4 gate.
#     Redirect to a file: logging goes to stderr, and piping it through Out-String
#     deadlocks on a full pipe buffer.
.venv\Scripts\python.exe -u -m brats.data.manifest --workers 6 `
  2> C:\data\brats2023\_manifest\run.err

# 1c. Splits: 80/10/10, stratified by cohort, grouped by patient, seed 42.
.venv\Scripts\python.exe -m brats.data.splits
git add splits/split_random_seed42.csv && git commit -m "Freeze data splits"
```

**Gate (1b).** Non-zero exit means a hard invariant failed. The one that matters:
`cases containing label 4 : 0`. A non-zero count means the data is not BraTS 2023 and
every ET result downstream would be silently empty.

**Gate (1c).** Splits are generated **once** and committed. `--force` is required to
overwrite, deliberately, because regenerating invalidates every result computed
against them. Verify anytime with `--check`.

---

## 2. Confound pre-screen — before any model is trained

```powershell
.venv\Scripts\python.exe -m brats.confound.probes --max-cases 300
```

**Gate.** A written note stating, with numbers, how separable the three cohorts are
from intensity histograms and brain-mask geometry alone, using **no tumor information
at all**. Output lands in `reports/confound_prescreen.txt`.

This result shapes the classification head. If a random forest on mask geometry alone
hits >0.90 balanced accuracy, then a near-perfect deep classification accuracy is
evidence of confounding, not of a good model.

---

## 3. Preprocess and cache

```powershell
# Verify the quantization round-trip on 4 cases first.
.venv\Scripts\python.exe -m brats.data.preprocess --limit 4 --verify

# Then the full cache (~52 GB, ~22 MB/case).
.venv\Scripts\python.exe -m brats.data.preprocess --workers 8
```

**Gate.** Round-trip error within half a quantization step (0.039σ) and labels
lossless. The projected full-cache size must fit the node's **93.13 GiB** local disk —
that constraint is why the cache is uint8 and not float16 (which would need ~110 GB).

---

## 3b. Run on Databricks (current target) or any Linux GPU host

The training code is platform-neutral: paths come from environment variables, there is no
Snowflake call unless `stage_uri` is set, and the same entry point serves one GPU or many.

```bash
pip install -r requirements.txt            # torch+CUDA first if the runtime lacks it (see the file header)
export BRATS_CACHE_ROOT=/Volumes/imaging_research/brain_tumor_segmentation/brats_dataset   # .npz cache: <root>/{GLI,MEN,PED}/<case>.npz
export BRATS_SPLITS_CSV=$PWD/splits/split_random_seed42.csv                                # or leave unset: repo copy is used
export BRATS_CKPT_ROOT=/Volumes/imaging_research/brain_tumor_segmentation/checkpoints     # per-epoch checkpoints; resume reads here

# one GPU
python -m brats.train --config configs/segresnet_base.yaml --epochs 150 --run-name base --no-wandb --resume

# N GPUs on one node (NCCL; SyncBatchNorm is enabled automatically)
torchrun --nproc_per_node=4 -m brats.train --config configs/segresnet_base.yaml --epochs 150 --run-name base --resume
```

Things that behave differently from what you might assume:

* **Batch norm vs instance norm is chosen at run time**: `batch_size_per_gpu × GPUs >= 4` gives batch norm,
  otherwise instance norm. Checkpoints record which (`model_config`), and `evaluate`/`qc` rebuild the model
  from it, so a one-GPU checkpoint evaluates correctly.
* **Cohort balance**: single process → a weighted sampler *and* an unweighted CE loss; DDP → the
  `cohort_balanced_sampling` flag is ignored and the class-weighted CE carries the balance (a warning says so).
* **Non-finite gradients are skipped**, not applied: the step is logged with the offending case ids and the epoch
  log reports `skipped_steps`. After `max_nonfinite_steps` (default 20) consecutive skips the run aborts and
  `last.pt` from the previous epoch is intact. If you ever see this, it is a bug to report, not noise.
* `data.yaml` still lists Windows paths for archives/extraction and the Databricks Volume for `cache_root`;
  the env vars above override it per run. `preprocess` needs the raw NIfTI and manifest, so run it where the
  data was extracted and copy the finished `.npz` cache to the Volume.
* If a cache was written before the z-axis crop fix, re-run `python -m brats.data.preprocess --overwrite --verify`
  (brains spanning more than 144 slices were silently clipped).

## 4. SPCS local image and launch prerequisites

The approved workflow is a finite Docker/torchrun job, not the legacy
`brats.snowflake.submit_job` ML Jobs examples. Keep Databricks notebooks and
`segresnet_base.yaml` unchanged; use `configs/segresnet_spcs.yaml` for SPCS.
The SPCS candidate is four A10Gs on `CARE_ML_RSRCH_GPU_M`, batch 2 per GPU,
global batch 8 (not Databricks' 16), 128-cubed, bf16, 100 epochs, LR warmup 8,
attention warmup 20. DDP retains class-weighted CE and rank-local batch Dice;
it does not use the single-process cohort-balanced sampler. Do not silently
change the learning rate or sampling algorithm.

Local preparation (no registry push, stage upload, W&B run or GPU job):

```bash
bash spcs/build.sh spcs-YYYYMMDD-unique-source-build
# The script refuses existing tags and latest, checks pip, imports and CPU model.
# tests are mounted, not baked into the runtime image:
docker run --rm --network none --hostname brats-local \
  --add-host brats-local:127.0.0.1 --platform linux/amd64 --shm-size 2g \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/repo/src \
  -v "$PWD:/repo:ro" -w /repo brats-train:spcs-YYYYMMDD-unique-source-build \
  python -m pytest /repo/tests -q -rs -p no:cacheprovider
```

The Dockerfile pins a runtime base by digest and constrains Torch so dependency
resolution cannot replace its CUDA wheel. The initial 2.8.0/1.6.0 candidate was
not accepted blindly: PyPI advisory metadata reports Torch checkpoint/unpickler
and other local API issues and MONAI bundle-eval issues. The replacement candidate
is Torch 2.13.0/CUDA 12.6 and MONAI 1.6.1. Metadata showing no listed advisories is
not a complete vulnerability scan. Evidence checked October 15, 2026:
[PyTorch 2.8 metadata](https://pypi.org/pypi/torch/2.8.0/json), including
[PYSEC-2026-2286](https://osv.dev/vulnerability/PYSEC-2026-2286), and
[MONAI 1.6.0 metadata](https://pypi.org/pypi/monai/1.6.0/json), including
[PYSEC-2026-4017](https://osv.dev/vulnerability/PYSEC-2026-4017).
The selected direct pins had no listed PyPI advisories; OS/transitive scanning
and account GPU driver checks remain separate gates. Pytest 9.0.3 avoids the
listed temporary-directory issue in <=9.0.2. `/app/build-provenance.json` contains actual
source/config hashes and versions; `/app/dependencies.lock.txt` captures resolved
transitives. OCI labels include source revision and dirty-working-tree digest.
Final rebuild must happen after all source edits, under another immutable tag.
Python 3.11/3.12, imports, model forward/backward, image platform and pinned SDK
settings must pass; CPU success does not establish GPU driver/NCCL compatibility.

Before any remote action, resolve these gates:

- `/guardrails`: current Restricted Session Scope blocks the EAI inspection.
  Do not bypass by role changes, removing restrictions, or alternate CLI SQL.
- Explicit W&B entity, project `brats2023`, and securely provisioned project
  SECRET injected into `WANDB_API_KEY` through `containers.secrets`. Values go
  through `/secrets`, never chat/spec/build args or source. Local secret storage
  is not Snowflake SECRET provisioning. First verify project-role CREATE SECRET
  and READ/ownership; do not assume ACCOUNTADMIN is necessary.
- Verify `CARE_ML_RSRCH_INTERNET_EAI` usage, job attachment, required W&B API and
  signed-upload hosts, and exact create/observe/cancel privileges using
  `SFK_CARE_ML_RSRCH_ADM`. No wildcard egress or TLS bypass.
- Verify full CPU cache integrity and frozen split digest. Old staged NPZ files
  may predate the z-crop fix; a rebuilt image does not repair them. Stop/report
  mismatches; do not regenerate splits or alter labels. The all-case mechanical
  audit may inspect test labels for integrity, never predictions or tuning.
- Confirm intended monitoring warehouse (observed `CORP_ENTITY_XS_WH` differs
  from supplied `CARE_ENTITY_XS_WH`), available credit history, and approved smoke
  runtime/startup deadline. Fill every `REQUIRED_*` template value before upload.

`spcs/push.sh TAG --approve-image-push` only pushes an existing reviewed image
and requires independently verified remote tag absence. Registry login, image
push, spec upload, secret/EAI changes and paid jobs each need concrete approval.
Do not overwrite the previous `latest` image. No new pools or public endpoints.

## 5. Bounded four-GPU smoke and recovery gate

`spcs/service_spec.yaml` is a non-launchable template until prerequisites above
are filled. It launches four local processes with `python -m torch.distributed.run
--standalone --nnodes=1 --nproc_per_node=4` (torchrun's module entry point), not four
GPUs pooled into one address space. The module entry point keeps elastic and rank
workers on the image virtual environment rather than a system-script shebang.
CPU40/RAM144Gi request, RAM160Gi limit and 16Gi shared-memory are candidates;
shared memory and loader copies count toward memory. Verify sidecar headroom,
schedulability and measured memory. DATASET_STAGE is read-only by application
contract, not enforced by an invented YAML readOnly flag or by stage ownership.
All mutable checkpoints/logs use `/work`; only closed immutable objects and
completion manifests publish under OUTPUT_STAGE run prefixes.

Use a distinct smoke run and explicit approved seconds. Require four CUDA GPUs,
bf16 and NCCL, at least two successful batch-2/rank full128-cubed updates,
validation followed by more training, and measured peak VRAM per rank. Stop on
OOM or configuration mismatch; do not shrink the recipe silently. Verify a
completed epoch and its hashes, kill mid-next-epoch, resume into fresh local
storage with the same run ID/new attempt ID, and perform another update.
Resume must reject missing/incompatible checkpoints and preserve one exact
collectively selected completed epoch. Partial-epoch work can be lost; CUDA
bit-identical replay is not promised. Never publish partial weights as complete.

`spcs/supervise.py` bounds container execution, sends TERM then bounded KILL,
and tracks elastic rank sessions, adopted orphans and loader descendants with
Linux subreaper/process identity checks. The runtime's collective stop handling
gets a finite grace (120s candidate). Forced termination can interrupt publication;
readers must fall back to the previous verified completion. No pool suspend.

The container clock does **not** cover queue, node resume or image pull. Before
submission record a local receipt with exact fully qualified unique `job`,
`owner`, `created_on`, timezone-aware `startup_deadline` and `hard_deadline`, and
`cancellation_approved: true`. Independently observe that exact job's same
identity and `state`; run `python spcs/monitor.py deadline --receipt receipt.json
--observation observation.json` between bounded observations. If startup or hard
deadline expires, use the separately verified own-job cancellation operation,
then keep observing until terminal and confirm no running containers. An API
error or missing log is not proof of termination. The helper intentionally does
not execute unverified SQL or cancel anything while scope/permissions are
unresolved. Operator presence is mandatory until a remote watcher is verified.

## 6. Tracking, observability and the full-run decision

SPCS requires online numeric-only W&B preflight; missing key/entity/SDK or failed
init stops before training. It never silently falls back offline. Generated UUID
run identity is stored with checkpoints; resume uses that same ID and SDK-managed
monotonic event steps, while optimizer step is a separate numeric field. Replayed
epochs remain visible; W&B does not rewind. Names/tags are generated, configuration,
summary and metric keys have closed allowlists, and artifact/watch methods are
no-ops in this profile. Console, code, git, metadata and automatic system capture
are disabled. MRI/PNG/NIfTI, identifiers, tables, source and model artifacts stay
in local/Snowflake storage, not W&B. Do not indiscriminately sync SDK directories.

After successful preflight, tracking errors are visible but nonfatal. Local
`tracking-health.json` and `tracking-metrics.jsonl` snapshots are authoritative.
`flush_ok` records SDK finish, **not** independently verified remote history or
file inventory; `remote_history_verified` stays false until separately checked.
The approved smoke must read back numeric history and inspect actual W&B run
files for console/source/case metadata before accepting privacy/egress. An init
success alone does not prove signed uploads or history flush.

The spec requests platform metric groups `system`, `system_limits`, `network`,
`storage` and event-table log export. Actual collection and privileges must be
verified on the named smoke job. Current container logs use
`SYSTEM$GET_SERVICE_LOGS`; historical service-scoped `SPCS_GET_LOGS` and
`SPCS_GET_METRICS` helpers avoid blanket event-table access where owner/MONITOR
and collection permit. These names are documentation guidance, not evidence
that this account is collecting them. Keep local/stage snapshots as fallback.
See [monitoring services](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/monitoring-services).

Cost reporting must separate:

- Live estimates: `python spcs/monitor.py estimate --node-hours OBSERVED_HOURS`
  reports unknown credits without an evidenced rate. Add
  `--credits-per-node-hour VERIFIED_RATE` only when supplied/verified; include
  startup/idle/tail and shared-pool assumptions. No invented dollar rate.
- Delayed actual **pool** credits: documented
  `SNOWFLAKE.ACCOUNT_USAGE.SNOWPARK_CONTAINER_SERVICES_HISTORY` is hourly and may
  lag up to 180 minutes. It is not exact per-run live billing. Access remains
  unverified; an earlier permission failure is not zero spend. Ask for a narrow
  project-pool report/view if needed, not blanket admin. Validate any alternative
  API before promising it. Monitoring SQL warehouse cost is separate.

Resource monitors do not hard-cap SPCS compute pools. The requested 1000-credit
monthly value is unverified; budgets/alerts are delayed, not instantaneous caps.
After smoke, report train/validation/checkpoint times and full-run time/credit
range, then obtain a full-run duration/spend decision. Never auto-launch full
training or automatically suspend a shared pool.

`spcs/inference_spec.yaml` targets one GPU on `CARE_ML_RSRCH_GPU_S` (pool remains
unverified). It streams validation first into a fresh local directory, publishing
original-grid NIfTI, compact results and bounded QC to OUTPUT_STAGE. No online
API and no model registry. Internal-test evaluation remains explicitly gated.

---

## 7. Post-processing — the highest-return stage

Tune `configs/postproc.yaml` on **validation only**. It is read by `evaluate` and `qc` (`--postproc-config` to use another file, `--tta` to override the YAML); the settings used are written to `postprocess_config.json` next to the results. Unknown keys fail loudly.

```powershell
.venv\Scripts\python.exe -m brats.evaluate --split val --checkpoint <ckpt> --tta none
.venv\Scripts\python.exe -m brats.evaluate --split val --checkpoint <ckpt> --tta eight_flip
```

**Gate.** ≥ **+0.05** average lesion-wise Dice over the un-post-processed baseline,
from post-processing alone. The ablation table is printed automatically.

Two traps, both encoded in the code and configs:
- Validation needs **larger** thresholds than training folds — never tune on
  training-fold predictions and ship.
- Tune to the **BraTS rank**, not to mean Dice (`brats/metrics/ranking.py`).

TTA is a sweep, not an assumption: the published evidence is genuinely split.

---

### Leaderboard-comparable numbers

`evaluate` uses the in-repo lesion-wise implementation, which reproduces the official one (verify it yourself:
`git clone https://github.com/rachitsaluja/BraTS-2023-Metrics.git external/brats_metrics`, then
`python -c "from brats.metrics.lesionwise import verify_against_official as v; r = v(); print({k: x for k, x in r.items() if k != 'rows'})"`). For a number you will
publish, score exported label maps with `brats.metrics.lesionwise.score_with_official(pred_dir, gt_dir, out_csv)`.

## 8. Confound diagnostics

```powershell
.venv\Scripts\python.exe -m brats.confound.binarize_control --epochs 30
# embedding + permutation probes: see brats/confound/embedding_probe.py
```

**Gate.** Quantified answers to *"does the classification head cost segmentation
Dice?"* (from §6's λ=0 ablation) and *"how much of the classification signal is
confound?"*

---

## 9. Final evaluation — the test split opens ONCE

```powershell
# Internal test split: full quantitative segmentation + classification.
.venv\Scripts\python.exe -m brats.evaluate --split test --checkpoint <ckpt> `
  --i-understand-this-opens-the-test-split

# Official ValidationData (405 cases):
#   - classification IS quantitative (cohort label comes from the source archive)
#   - segmentation is VISUAL ONLY (no seg.nii.gz exists)
.venv\Scripts\python.exe -m brats.evaluate --split official_val --checkpoint <ckpt>
.venv\Scripts\python.exe -m brats.qc --split official_val --checkpoint <ckpt> `
  --max-per-cohort 10
```

The flag on `--split test` is a deliberate speed bump. The test split may be opened
exactly once, after every hyperparameter and threshold is frozen. Every peek
invalidates it.

Report per cohort × per region × {mean, median, IQR, bootstrap CI}. **Never a bare PED
number** — n≈10 there, and differences under 0.05 Dice are noise.

---

## Status

Historical local data reports below have not been regenerated in this session.
The user reports a successful Databricks notebook run; its metrics/checkpoints
were not independently inspected. The current SPCS image is
`brats-train:spcs-20261015-local2` (linux/amd64), with Python 3.12.3,
PyTorch 2.13.0+cu126, MONAI 1.6.1 and W&B 0.23.0. Dependency checks and a CPU
model forward/backward pass succeeded. No registry push, remote W&B run or
paid GPU job was performed. Independent final-review workers could not start
(`ENOENT: uv_cwd`); the parent performed the final review instead.

| Step | State |
|---|---|
| 0. Environment | SPCS local image validated as above; host Python 3.14 is outside the supported project range |
| 0. `external/brats_metrics` clone | **not done** — needed before reporting any number |
| 0. Snowflake launch | **blocked** - scoped EAI inspection, secret/entity, telemetry/cost and explicit paid-job approval remain |
| 1a. Extract | **done** — 2755 cases: GLI 1251/219, MEN 1000/141, PED 99/45, all counts exact |
| 1b. Manifest | **done, PASSED** — 0 label-4s, labels exactly {0,1,2,3}, 0 integrity errors |
| 1c. Splits | **done, committed** — `splits/split_random_seed42.csv`, 0 patient leakage |
| 2. Confound pre-screen | **done** — mask geometry alone reaches 0.769 balanced acc (chance 0.333) |
| 3. Preprocess cache | **NOT RUN here** — re-run with `--overwrite --verify` if built before the z-crop fix |
| 6. Train / 7. Evaluate | Final baked SPCS image: 250 CPU tests passed, 3 skipped; distributed CUDA execution remains unverified |
| Lesion-wise metric | Internal approximate HD95; official-reference tests skipped because the reference repository is absent |
| SPCS rollout | No remote launch; follow sections 4-6, including cache audit and bounded GPU smoke |

The historical pre-screen numbers above used patient-leaky CV and should be rerun
before scientific interpretation. Do not regenerate the frozen split. Validate
the existing staged cache before deciding whether preprocessing repair is needed.
