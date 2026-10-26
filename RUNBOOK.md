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

### Isolated repair and source verification

Use `scripts/rebuild_cache.py` for a previously truncated cache. It requires a new
local output directory and the exact frozen split SHA256. `--reuse-cache` may point
to an earlier incomplete repair, but existence is not sufficient: each candidate
must pass the mechanical audit and voxelwise original-source round-trip before it
is copied. Missing or invalid candidates are rebuilt from original NIfTI. Source
files, reuse candidates, manifests and splits are hash-checked for changes. No
existing cache, source image, label or split is overwritten, and no upload or
training is performed.

The crop remains image-derived unless it would discard an annotation; then the
box expands to contain image and label support. Normalization still uses only the
MRI nonzero union. Annotations outside that MRI mask are preserved and reported,
not erased or used to change image intensities. This is a data-preservation step,
not annotation correction or test-set tuning. Official validation remains unlabeled.

The output contains `cache/{GLI,MEN,PED}/*.npz`, `cache_index.csv`, and
`repair-report.json` with per-case source hashes, actions and round-trip results.
`cache-audit.json` is the completion marker and is written last only after exact
coverage and all checks pass. A failed report is not a usable training audit.
Never point training at the older `_cache` or an incomplete repair merely because
its file count appears correct. These preprocessing changes do not alter the NPZ
encoding consumed by the trainer; source provenance must still distinguish the
repair checkout from the already-built runtime image.

Completed October 26, 2026: the isolated cache at
`C:\data\brats2023\_cache_verified_20261026\dataset\cache` passed all 2,755
original-source round-trips and the complete mechanical audit (GLI 1,470;
MEN 1,141; PED 144). 2,754 files were copied from the previous repair only after
verification; `BraTS-MEN-00781-010` was rebuilt to preserve its excluded label.
The completed cache is 12,026,760,158 bytes. Source labels outside the MRI nonzero
mask were retained in 32 cases (327 voxels), not silently corrected. The maximum
normalized intensity round-trip error was 0.0196863 sigma. Original manifests and
frozen split bytes remained unchanged.

Audit: `C:\data\brats2023\_cache_verified_20261026\dataset\cache-audit.json`.
Dataset identity: `e0b7fe2fb085ceddc298ee76e7edaf18c5a6e3e9b1ab84e57f542ee7206ecb13`.
The sibling `repair-report.json` and `repair-provenance.json` record per-case and
execution evidence. The existing uploaded `spcs-20261026-runtime1` image's baked
`verify_audit_identity` successfully checked every repaired NPZ against this audit;
the encoding is compatible without changing the training reader. The new repair
code is not baked into that image. Local regression suite: 305 passed, 3 skipped.

### Verified Snowflake upload (October 26, 2026)

Uploaded all 2,755 repaired NPZ files (12,026,760,158 bytes), the three unchanged
manifest files, and the frozen split without recompression or overwrite. Every
uploaded file was downloaded to separate local storage and its exact size and
SHA256 checked against the completed audit/metadata hashes. All 2,759 passed.
The pre-existing stage inventory, including sizes, checksums and timestamps,
was unchanged. No GPU pool was resumed and no training job was submitted.
The exact uploaded runtime image (digest
`sha256:9f1378954e83d1fa4d9763726becd65791757bda80aeffd706010467873e41cf`)
also passed `verify_audit_identity` using only these downloaded cache, metadata,
split and audit bytes, with networking disabled and no checkout source override.
This establishes data/reader compatibility, not live GPU or stage-mount behavior.

Dataset prefix:
`@CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.DATASET_STAGE/versions/cache-20261026-e0b7fe2fb085/`
contains `cache/{GLI,MEN,PED}/`, `manifest/`, and `splits/`.

Audit:
`@CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.OUTPUT_STAGE/audits/cache-20261026-e0b7fe2fb085/cache-audit.json`
was published only after remote data verification. The same output prefix holds
`repair-report.json`, `repair-provenance.json`, `cache_index.csv`, and
`upload-receipt.json`; each was downloaded and SHA256-verified. Repair provenance
is the unchanged pre-upload snapshot; the upload receipt records publication.
Audit SHA256: `49b561138e2dc53c9d77d32e75fde5b7df05227b0db7f7f85c5d37c5a196f364`.
Receipt SHA256: `92b1f98925c32ebb6ba1d5b4a2dfe6434680ee854322ba0e0eb61ac1902520fe`.
Local transfer evidence:
`C:\data\brats2023\_cache_verified_20261026\upload-20261026-e0b7fe2fb085\`.

For the existing stage mounts, the next rendered specification must use:

```yaml
BRATS_CACHE_ROOT: "/mnt/data/versions/cache-20261026-e0b7fe2fb085/cache"
BRATS_MANIFEST_DIR: "/mnt/data/versions/cache-20261026-e0b7fe2fb085/manifest"
BRATS_SPLITS_CSV: "/mnt/data/versions/cache-20261026-e0b7fe2fb085/splits/split_random_seed42.csv"
BRATS_CACHE_AUDIT_PATH: "/mnt/output/audits/cache-20261026-e0b7fe2fb085/cache-audit.json"
BRATS_EXPECTED_SPLIT_SHA256: "3e0efe90585fbb98be94f2035901d646a012a472090c8a9153344528df55f51e"
```

Do not use `/mnt/data` as the cache root: it now contains both old and versioned
NPZ files, so exact coverage verification will reject it. The subsequent local
specification update sets all five values above and references the already-uploaded
`spcs-20261026-runtime1` image. Its updated preparation template was uploaded to
`@CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.SPECS_STAGE/prepared/cache-20261026-e0b7fe2fb085-runtime1/service_spec.yaml`
and downloaded for SHA256 verification:
`d59eb361b41f78d66d5c78774602d9950877b192c2d6e38dea8f4b7af96913fb`.
Older staged specifications remain unchanged. This template still contains run,
attempt and deadline placeholders; render a separate launch specification before
submission. No training has started.

Historical initial-cache commands (not the isolated repair workflow):

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

### Prepare specifications without starting a job

The image `brats-train:spcs-20261026-runtime1` has been pushed and its registry
index digest reverified on October 26 as
`sha256:9f1378954e83d1fa4d9763726becd65791757bda80aeffd706010467873e41cf`.
The external `spcs/service_spec.yaml` now references that image and the confirmed
W&B entity `banerjee-s-corewell`. The supplied project page is
<https://forge.coreweave.com/wandb/banerjee-s-corewell/brats2023>.
This browser URL is not an SDK API endpoint: do not set `WANDB_BASE_URL` to it.
Confirm the SDK endpoint/egress with the workspace configuration before launch.

`spcs/prepare_job.py` renders local YAML only. It cannot authenticate, upload,
create a secret, resume a pool, or submit a job. It requires an explicit mode,
existing SECRET object name (not the key), completed audit path, frozen split
SHA256 and runtime limit; missing values/placeholders fail instead of defaulting.
Generated run/attempt IDs are unique, and existing output files are never replaced.
The renderer validates input syntax, **not** remote existence, privileges or audit
contents; the training image rechecks current NPZ hashes against the audit.

Create an ignored local output directory with `mkdir -p runs/spcs`. In the example
below, replace the uppercase input tokens with verified values; the example is
intentionally not runnable unchanged. Use the existing image as the local Python
runtime, with networking disabled:

```bash
docker run --rm --network none --platform linux/amd64 \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/repo:ro" \
  -v "$PWD/runs/spcs:/prepared" brats-train:spcs-20261026-runtime1 \
  python /repo/spcs/prepare_job.py --mode smoke \
  --secret-name CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.WANDB_API_KEY \
  --cache-audit-path /mnt/output/audits/VERIFIED_AUDIT.json \
  --split-sha256 VERIFIED_64_CHARACTER_SPLIT_SHA256 \
  --runtime-seconds APPROVED_SMOKE_SECONDS \
  --output /prepared/smoke.yaml
```

This prepares the dedicated four-rank GPU smoke harness, **not** the 100-epoch
trainer. The default is two updates per attention phase (six per rank) and one
validation volume. It retains full128/batch2/bf16, strict online tracking, and the
same stage mounts. `--smoke-lambda-cls 0.0` prepares the separate classifier-disabled
smoke gate. `--interrupt-after-publication` prepares deliberate exit 75 after a
completed checkpoint. For recovery, render another file with `--resume --run-id`
set to the original smoke ID; a new attempt ID is generated. Do not change modes,
lambda, or update counts when resuming the same run.

After smoke acceptance and a separate full-run decision, prepare full training by
using `--mode train`, its own explicit runtime limit, and a different output file.
That spec retains 100 epochs, validation every 10 epochs, and a fresh non-smoke
run prefix. There is no automatic promotion or submission. Resume only an existing
run of the same mode with a new attempt ID. The supervisor's TERM/KILL grace can
extend beyond its runtime timer; the external hard deadline must allow for that
and independently cover queueing/image pull.

These are **external launch-file changes**: the pushed image already supports
the environment overrides and both entry points. No rebuild/push is needed to
use the rendered YAML. Its bundled template is older; render from the checkout
(`/repo/spcs/prepare_job.py` above), not from `/app/spcs` in that image.

At eventual submission, attach `CARE_ML_RSRCH_INTERNET_EAI` using the job's
`EXTERNAL_ACCESS_INTEGRATIONS` property, not an invented YAML field. Upload/rendering
alone does not attach egress. No submission command is executed by this helper.

Before any job launch, resolve these gates:

- Check the active `/guardrails` scope before restricted operations. The earlier
  session blocked EAI inspection; the credential-setup session on October 26, 2026
  reported no active RSS. This does not verify EAI privileges or allowed hosts.
  Do not bypass a restriction by changing roles or using alternate CLI SQL.
- The W&B entity is `banerjee-s-corewell`, project `brats2023`. On October 26, 2026,
  `CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.WANDB_API_KEY` was created as GENERIC_STRING
  and its ownership by `SFK_CARE_ML_RSRCH_ADM` verified without retrieving its value.
  The generic secret can be referenced by other experiments with appropriate
  parent-object and secret access; no additional roles were granted access here.
  The local specification injects it into `WANDB_API_KEY` through `containers.secrets`.
  Credentials must enter through a secure masked/no-echo prompt, never chat,
  spec/build arguments or source. Local credential storage and Snowflake SECRET
  provisioning are separate. Creation does not prove W&B authentication or delivery.
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

### Current operator launch steps (prepared, not executed)

The current image, repaired data, frozen split and audit are uploaded and verified.
The research role passed `EXPLAIN_PRIVILEGES` for job submission; GPU_M remains
suspended with zero jobs, and the W&B secret exists. The EAI is enabled and the
role has USAGE. Its network-rule host list is still not inspectable by this role.
Actual GPU fit/NCCL, W&B authentication/history delivery, stage-mounted persistence
and forced-interruption recovery remain unverified. The SQL tool's compile-only
wrapper cannot compile `EXECUTE JOB SERVICE` (it rejected `EXECUTE`); the privilege
check is not service-spec validation or scheduling evidence. No launch was attempted.

A concrete first-smoke candidate is already rendered locally at
`runs/spcs-20261026-runtime1/smoke.yaml`. It uses the new image and verified data,
a generated `smoke-f8bd16af7f7645b198e04973158a684f` run ID, a unique attempt ID,
and a **proposed** 1,800-second container runtime limit. Rendering is not paid-run
approval. The stage file under `prepared/.../service_spec.yaml` remains a template;
do not launch it directly. Snowflake CLI can submit the rendered local YAML instead.

#### 1. Start the first smoke only when accepting its cost and deadlines

Proposed operator limits: cancel if not RUNNING within 15 minutes of submission;
cancel if not terminal within 50 minutes of submission. These are manual operator
deadlines, not provisioned Snowflake limits or a credit cap. The container's own
30-minute timer starts after image pull and can add 120 seconds of shutdown grace.
Record submission time and stay present, observing at least once a minute. Capture
the job owner/created_on from DESCRIBE before any cancellation. Do not suspend the
shared pool to stop your individual job.

Run from WSL using the installed Windows Snowflake CLI. The `--spec-path` must be
a Windows path, not `/mnt/c/...`. The following command **starts paid GPU compute**:

```bash
SNOW="/mnt/c/Program Files/Snowflake CLI/snow.exe"
REPO_WIN="C:/Users/SOU66793/OneDrive - Corewell Health/Desktop/Projects/Brain_Tumor_MRI"
JOB="CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.BRATS_SMOKE_F8BD16AF7F7645B198E04973158A684F"
CONNECTION=(--connection corewell-ml --authenticator externalbrowser --role SFK_CARE_ML_RSRCH_ADM --database CARE_ML_RESEARCH_WKSP --schema BRAIN_TUMOR_SEG)
"$SNOW" spcs service execute-job "$JOB" \
  --compute-pool CARE_ML_RSRCH_GPU_M \
  --spec-path "$REPO_WIN/runs/spcs-20261026-runtime1/smoke.yaml" \
  --eai-name CARE_ML_RSRCH_INTERNET_EAI --async "${CONNECTION[@]}"
```

Submit this candidate once only. A repeat/recovery attempt requires a newly rendered
specification, new job name and attempt ID; retain the run ID only for explicit resume.
Do not set four replicas: the one instance already starts four local DDP ranks.
No manual pool resume is needed when its verified `auto_resume=true` remains set.

#### 2. Observe it and cancel only that job if necessary

```bash
"$SNOW" spcs service describe "$JOB" "${CONNECTION[@]}"
"$SNOW" spcs service list-containers "$JOB" "${CONNECTION[@]}"
"$SNOW" spcs service logs "$JOB" --container-name train --instance-id 0 \
  --num-lines 200 "${CONNECTION[@]}"
```

For an expired deadline, stuck job or manual stop, run the following in Snowsight
as `SFK_CARE_ML_RSRCH_ADM`, after confirming this is the exact job you submitted:

```sql
SELECT CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.BRATS_SMOKE_F8BD16AF7F7645B198E04973158A684F!SPCS_CANCEL_JOB();
```

The documented helper requires OPERATE on the job (normally available to its owning
role); its invocation cannot be verified until the job exists. Keep checking DESCRIBE
and container state until cancellation/completion is confirmed. Do not assume closing
the terminal or cancelling the returned asynchronous SQL query stops the GPU job.

Accept the first smoke only after status DONE, a
`gpu_compute_and_checkpoint_gate_passed` log, six successful updates per rank,
four peak-VRAM values, a hash-complete checkpoint bundle under
`@CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.OUTPUT_STAGE/training/smoke-f8bd16af7f7645b198e04973158a684f/`,
and visible numeric W&B history for the generated run. Inspect W&B's file inventory
for the privacy contract; SDK init/finish alone does not prove delivery. The smoke
logs successful-update history to W&B; its validation/VRAM metrics are in the stage
checkpoint evidence, not a demonstration of full training-loss curves.

Complete the separately described classifier-disabled and recovery gates before
calling full training validated. `--interrupt-after-publication` is a controlled
exit after publication, not the forced mid-next-epoch interruption test.

#### 3. Prepare full training only after acceptance and a duration decision

The 100-epoch duration is not measured yet. Choose the limit after inspecting the
smoke and accounting for full epochs, 50-volume validation and checkpoint I/O;
do not interpret the smoke's 30-minute limit as a full-run limit. A smoke alone is
not a precise full-run cost forecast. The limit applies to one attempt; recovery
continues from the last completed epoch with `--resume` and a new attempt ID.

From WSL, render a fresh full-training file (this command is local-only):

```bash
REPO="/mnt/c/Users/SOU66793/OneDrive - Corewell Health/Desktop/Projects/Brain_Tumor_MRI"
read -r -p "Approved full-training attempt limit in seconds (>120): " TRAIN_SECONDS
docker run --rm --network none --platform linux/amd64 \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$REPO:/repo:ro" \
  -v "$REPO/runs/spcs-20261026-runtime1:/prepared" \
  brats-train@sha256:9f1378954e83d1fa4d9763726becd65791757bda80aeffd706010467873e41cf \
  python /repo/spcs/prepare_job.py --mode train \
  --secret-name CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.WANDB_API_KEY \
  --cache-audit-path /mnt/output/audits/cache-20261026-e0b7fe2fb085/cache-audit.json \
  --split-sha256 3e0efe90585fbb98be94f2035901d646a012a472090c8a9153344528df55f51e \
  --runtime-seconds "$TRAIN_SECONDS" --output /prepared/train.yaml
```

Check that rendering succeeded and record its generated run/attempt IDs. If the
file already exists, the renderer refuses to overwrite; use a new filename and
review which attempt you mean to launch. Then, **only after approving this run**:

```bash
JOB="CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.BRATS_TRAIN_$(date -u +%Y%m%d_%H%M%S)"
"$SNOW" spcs service execute-job "$JOB" \
  --compute-pool CARE_ML_RSRCH_GPU_M \
  --spec-path "$REPO_WIN/runs/spcs-20261026-runtime1/train.yaml" \
  --eai-name CARE_ML_RSRCH_INTERNET_EAI --async "${CONNECTION[@]}"
```

Use the same observe commands with the new `$JOB`, and set explicit startup/hard
operator deadlines for this attempt. Job names and artifact run IDs are distinct.
For cancellation use `<exact full-training job FQN>!SPCS_CANCEL_JOB()`, not the smoke
name. No `--resume` is used on this fresh run.

Full training remains 100 epochs, BF16, 128-cubed patches, batch 2/rank (global 8),
validation at epochs 10,20,...,100 on 50 cases (GLI 20/MEN 20/PED 10), and checkpoints
after each completed epoch. Training losses, validation ET/TC/WT Dice and cohort
accuracy, LR, timings and runtime metrics are logged. **Validation loss is not
implemented.** W&B is `banerjee-s-corewell/brats2023`; MRI, case identifiers, code
and checkpoints stay in local/Snowflake storage. Full-training artifacts publish
under `OUTPUT_STAGE/training/<generated-train-run-id>/`.

References: [execute-job CLI](https://docs.snowflake.com/en/developer-guide/snowflake-cli/command-reference/spcs-commands/service-commands/execute-job),
[job cancellation](https://docs.snowflake.com/en/sql-reference/functions/spcs_cancel_job).
These instructions were checked against CLI 3.16.0 help and current documentation;
the launch, monitoring and cancellation commands were not executed on a live job.

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
were not independently inspected. The original SPCS image was
`brats-train:spcs-20261015-local2` (linux/amd64), with Python 3.12.3,
PyTorch 2.13.0+cu126, MONAI 1.6.1 and W&B 0.23.0. Dependency checks and a CPU
model forward/backward pass succeeded. The versioned image was subsequently pushed
and its registry digest verified; `latest` was left unchanged. No remote W&B run or
paid GPU job was performed. Independent final-review workers could not start
(`ENOENT: uv_cwd`); the parent performed the final review instead.
The subsequent local launch-preparation changes passed 277 tests (3 skipped)
using this image with the checkout mounted and networking disabled. That does
not verify Snowflake permissions, staged-cache integrity, or live W&B delivery.
The October 26 credential-only update created the generic W&B secret and updated
its external specification reference; 37 deployment/preparation tests passed
with networking disabled. No spec upload or job submission accompanied that update.

| Step | State |
|---|---|
| 0. Environment | SPCS local image validated as above; host Python 3.14 is outside the supported project range |
| 0. `external/brats_metrics` clone | **not done** — needed before reporting any number |
| 0. Snowflake launch | **Smoke candidate prepared, not submitted** - runtime1 image and updated template, repaired dataset/audit, secret and launch-privilege checks complete; GPU/W&B/recovery acceptance and full-run deadline/spend decision remain |
| 1a. Extract | **done** — 2755 cases: GLI 1251/219, MEN 1000/141, PED 99/45, all counts exact |
| 1b. Manifest | **done, PASSED** — 0 label-4s, labels exactly {0,1,2,3}, 0 integrity errors |
| 1c. Splits | **done, committed** — `splits/split_random_seed42.csv`, 0 patient leakage |
| 2. Confound pre-screen | **done** — mask geometry alone reaches 0.769 balanced acc (chance 0.333) |
| 3. Preprocess cache | **Repair and upload PASSED** - 2,755 source-verified cases uploaded under `DATASET_STAGE/versions/cache-20261026-e0b7fe2fb085/cache/`; every remote file downloaded and SHA256-verified; old cache preserved |
| 6. Train / 7. Evaluate | Uploaded runtime1 image: 282 tests passed, 3 skipped at build; current image-reference update: 37 focused tests passed; distributed CUDA execution remains unverified |
| Lesion-wise metric | Internal approximate HD95; official-reference tests skipped because the reference repository is absent |
| SPCS rollout | No remote launch; follow sections 4-6, including cache audit and bounded GPU smoke |

The historical pre-screen numbers above used patient-leaky CV and should be rerun
before scientific interpretation. Do not regenerate the frozen split. Use the
verified versioned cache and audit from section 3, not the preserved original cache.
