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

**Snowflake prerequisites — blocking for anything GPU.** Your user holds only
`PUBLIC` and `SFK_CORP_JIRA`; `CREATE COMPUTE POOL` is an account-level privilege you
do not have, and no external access integration exists. Hand
`scripts/admin_grants.sql` to someone with `ACCOUNTADMIN`. Phases 1–4 below run
without it.

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

## 4. Stage to Snowflake *(needs the admin grants)*

```powershell
.venv\Scripts\python.exe -m brats.snowflake.stage_data --print-sql   # stage DDL
.venv\Scripts\python.exe -m brats.snowflake.stage_data --estimate    # size check
.venv\Scripts\python.exe -m brats.snowflake.stage_data --stage @BRATS_MRI.CORE.DATA
```

---

## 5. GPU smoke test — the compute gate

```powershell
.venv\Scripts\python.exe -m brats.snowflake.submit_job --smoke-test
.venv\Scripts\python.exe -m brats.snowflake.submit_job --status <job_id>
```

**Gate.** 4 visible A10G GPUs, `bf16 supported: True`, a successful 128³ forward pass
with reported peak VRAM, and one cached case loading with label values ⊆ {0,1,2,3}.

---

## 6. Train

```powershell
# Short smoke run first: confirms the loop, then confirms RESUMABILITY.
.venv\Scripts\python.exe -m brats.snowflake.submit_job --train --epochs 5 `
  --run-name smoke

# Kill it mid-run, then:
.venv\Scripts\python.exe -m brats.snowflake.submit_job --train --epochs 5 `
  --run-name smoke --resume

# Baseline, then the full recipe.
.venv\Scripts\python.exe -m brats.snowflake.submit_job --train --epochs 50 `
  --run-name baseline
.venv\Scripts\python.exe -m brats.snowflake.submit_job --train --epochs 150 `
  --run-name segresnet_full

# THE REQUIRED ABLATION: does the classification head cost segmentation Dice?
# Matched backbone, split, and schedule -- only lambda differs.
.venv\Scripts\python.exe -m brats.snowflake.submit_job --train --epochs 150 `
  --lambda-cls 0.0 --run-name seg_only_ablation

# Confound ablation: plain GAP instead of tumor-attention pooling. Measures how much
# the confound is worth. Edit `pooling: gap` in the config for this arm.
```

**Gate (smoke).** Training resumes cleanly from a stage checkpoint after a forced
kill. Not optional: SPCS maintenance windows (Sat/Sun) can cancel a running job
service and Snowflake will not restart it.

**Gate (full).** GLI average lesion-wise Dice ≥ 0.78 on validation.

### Experiment tracking (Weights & Biases)

Enabled in `configs/segresnet_base.yaml` (`wandb: true`), **offline by default**.
Offline needs no network at all: run data lands in `runs/wandb/` for a later sync.

```powershell
# Live curves instead. Needs WANDB_EAI + the API-key secret from admin_grants.sql.
.venv\Scripts\python.exe -m brats.snowflake.submit_job --train --epochs 150 `
  --wandb-mode online --wandb-secret BRATS_MRI.CORE.WANDB_API_KEY

# Or retrieve an offline run afterwards and sync it locally.
wandb sync runs\wandb\offline-run-*
```

Only rank 0 logs, so the four ranks cannot interleave into one unusable curve.
Ablation arms are auto-tagged (`ablation-seg-only`, `ablation-gap-confound`,
`uncertainty-weighting`) — use `--wandb-group` to compare them side by side. Every
tracker call is failure-tolerant: a network blip or an expired key degrades to a
no-op rather than killing a job that has been burning A10G credits for hours.

Local single-GPU or CPU debugging of the same code path:

```powershell
.venv\Scripts\python.exe -m brats.train --config configs/segresnet_base.yaml --epochs 1
```

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

| Step | State |
|---|---|
| 0. Environment | **done** — uv, Python 3.12.14, venv, core deps, CPU torch 2.13 + MONAI 1.6.0, wandb 0.29 |
| 0. `external/brats_metrics` clone | **not done** — needed before reporting any number |
| 0. Snowflake grants | **blocked** — needs `ACCOUNTADMIN` to run `scripts/admin_grants.sql` |
| 1a. Extract | **done** — 2755 cases: GLI 1251/219, MEN 1000/141, PED 99/45, all counts exact |
| 1b. Manifest | **done, PASSED** — 0 label-4s, labels exactly {0,1,2,3}, 0 integrity errors |
| 1c. Splits | **done, committed** — `splits/split_random_seed42.csv`, 0 patient leakage |
| 2. Confound pre-screen | **done** — mask geometry alone reaches 0.769 balanced acc (chance 0.333) |
| 3. Preprocess cache | **NOT RUN here** — re-run with `--overwrite --verify` if built before the z-crop fix |
| 6. Train / 7. Evaluate | code complete and unit-tested (174 tests, CPU); **not run on a GPU or on real data** |
| Lesion-wise metric | matches the official implementation exactly (Dice, TP/FP/FN); HD95 within ~0.5 mm — `verify_against_official()` |
| 4–9 | not run (4+ blocked on grants) |

All scripts are written and unit-tested (`pytest`). Resume at step **3**. The pre-screen numbers above were produced with a patient-leaky CV and should be re-run (`python -m brats.confound.probes --max-cases 300`).
