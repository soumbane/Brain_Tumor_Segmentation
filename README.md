# Brain Tumor Segmentation + Tumor-Type Classification on BraTS 2023

Multi-task 3D deep learning on the ASNR-MICCAI BraTS 2023 challenge data: voxel-wise
segmentation of tumor sub-regions across three cohorts (adult glioma, meningioma,
pediatric glioma) plus a volume-level 3-class cohort classifier sharing the same encoder.

**Status:** the pipeline is implemented end to end (data → train → inference → post-processing →
lesion-wise evaluation → confound diagnostics) but **no model has been trained to completion yet**.
This document is the original plan and design rationale; where the code disagrees, the code wins.
See `RUNBOOK.md` for the commands and the current state of each step. Known deviations from this
plan: the classification-confound pre-screen was run, preprocessing/training/evaluation are written
and unit-tested but not yet run on real data, and Swin UNETR, the site-held-out split and the LRP
relevance maps are not implemented.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Critical constraints discovered before planning](#2-critical-constraints-discovered-before-planning)
3. [Dataset inventory (verified)](#3-dataset-inventory-verified)
4. [Label semantics — read this before writing any code](#4-label-semantics--read-this-before-writing-any-code)
5. [What the literature actually says](#5-what-the-literature-actually-says)
6. [Compute architecture](#6-compute-architecture)
7. [Environment and libraries](#7-environment-and-libraries)
8. [Data pipeline](#8-data-pipeline)
9. [Data splits](#9-data-splits)
10. [Model design](#10-model-design)
11. [The classification task is confounded — mandatory protocol](#11-the-classification-task-is-confounded--mandatory-protocol)
12. [Loss design](#12-loss-design)
13. [Training recipe](#13-training-recipe)
14. [Inference and post-processing — the single highest-leverage stage](#14-inference-and-post-processing--the-single-highest-leverage-stage)
15. [Evaluation](#15-evaluation)
16. [Performance targets](#16-performance-targets)
17. [Execution phases](#17-execution-phases)
18. [Planned repository layout](#18-planned-repository-layout)
19. [Risk register](#19-risk-register)
20. [References](#20-references)

---

## 1. Executive summary

**Segmentation task.** Given 4 co-registered MRI sequences per patient (T1n, T1c, T2w, T2f/FLAIR)
at 1 mm³ in SRI24 atlas space, predict three *nested* tumor regions: enhancing tumor (ET),
tumor core (TC), whole tumor (WT).

**Classification task.** Given the same volume, predict tumor cohort ∈ {GLI, MEN, PED} from a
head attached to the shared encoder bottleneck.

**Design decisions locked in:**

| Decision | Choice | Rationale |
|---|---|---|
| Compute | Snowflake SPCS GPU compute pool | No local NVIDIA GPU exists; keeps work inside the Corewell account |
| Final evaluation | Held-out **labeled** test split from TrainingData | Official ValidationData has no `seg.nii.gz`, so local Dice is impossible on it |
| Model scope | One joint cohort-conditioned model | Enables the classifier head; lets PED (99 cases) borrow features from GLI/MEN |
| Architectures | SegResNet (primary), Swin UNETR (secondary) | SegResNet is BraTS-native and best accuracy-per-VRAM; Swin UNETR as ensemble partner |
| Formulation | Multi-label, 3 sigmoid channels | Regions are nested/overlapping — softmax is *wrong* here |

**The counter-intuitive headline from the research:** architecture choice is worth roughly
**±0.01** lesion-wise Dice among the top backbones. Post-processing is worth **+0.10**.
Plan the effort budget accordingly — see §14.

---

## 2. Critical constraints discovered before planning

These were verified on this machine and in this Snowflake account, not assumed.

### 2.1 This workstation cannot train the model

```
CPU     AMD Ryzen 5 PRO 215, 6 cores / 12 threads
GPU     AMD Radeon 740M Graphics (integrated), ~1 GB VRAM, no CUDA
RAM     32 GB
Disk    C: 331 GB free
Python  not installed
CUDA    not present
```

There is no NVIDIA GPU. A 3D BraTS model at 128³ patches needs 16–24 GB of VRAM and
GPU-days of compute. Training here is not slow — it is infeasible. **The workstation's role is
authoring, EDA on a handful of cases, QC visualization, and job submission only.**

### 2.2 The official ValidationData has no ground truth

Verified by inspecting the archives: TrainingData cases contain **5** files
(4 sequences + `seg.nii.gz`); ValidationData cases contain **4** files (no `seg.nii.gz`).

Consequence: the original framing of "train on TrainingData, test on ValidationData" cannot
produce a Dice number locally. The plan therefore carves a labeled test split out of
TrainingData (§9). Predictions on the official ValidationData are still generated, but for
qualitative review and optional Synapse submission only.

### 2.3 The data is still zipped, and OneDrive is the wrong place for it

All six archives are unextracted. Both the dataset and this project live under
`OneDrive - Corewell Health`, which will attempt to sync every extracted NIfTI and every
cache file. **Extraction target must be a non-synced local path** (e.g. `D:\data\brats2023`
or `C:\data\brats2023`), and the training cache must live on the remote compute, not here.

### 2.4 Label `4` became label `3` in BraTS 2023

Any code, pretrained weights, or post-processing carried over from BraTS ≤2021 that keys ET
off `label == 4` will silently produce an **empty ET channel** — and ET is the hardest and
most heavily weighted region. This is the single most likely silent bug in the project.
See §4.

---

## 3. Dataset inventory (verified)

Root: `C:\Users\SOU66793\OneDrive - Corewell Health\Desktop\Datasets\BraTS_2023\`

| Cohort | Archive | Cases | Size | Has labels |
|---|---|---:|---:|:--:|
| **GLI** — adult glioma | `...GLI-Challenge-TrainingData.zip` | **1251** | 12.27 GB | yes |
| GLI | `...GLI-Challenge-ValidationData.zip` | **219** | 2.20 GB | **no** |
| **MEN** — meningioma | `...MEN-Challenge-TrainingData.zip` | **1000** | 18.79 GB | yes |
| MEN | `...MEN-Challenge-ValidationData.zip` | **141** | 2.64 GB | **no** |
| **PED** — pediatric glioma | `...PED-Challenge-TrainingData.zip` | **99** | 1.87 GB | yes |
| PED | `...PED-Challenge-ValidationData.zip` | **45** | 0.87 GB | **no** |
| | **Total labeled** | **2350** | 32.9 GB | |
| | **Total unlabeled** | **405** | 5.7 GB | |

Counts derived from archive entry counts (`files / 5` for training, `files / 4` for validation)
and cross-checked against directory counts. Consistent with the participant papers.

### Per-case structure

```
BraTS-GLI-00324-000/
├── BraTS-GLI-00324-000-t1n.nii.gz    # T1 native
├── BraTS-GLI-00324-000-t1c.nii.gz    # T1 post-contrast (gadolinium)
├── BraTS-GLI-00324-000-t2w.nii.gz    # T2 weighted
├── BraTS-GLI-00324-000-t2f.nii.gz    # T2 FLAIR
└── BraTS-GLI-00324-000-seg.nii.gz    # annotation (training only)
```

All volumes are already **co-registered to SRI24, resampled to 1 mm³ isotropic, and
skull-stripped** by the FeTS pipeline. Nominal shape 240 × 240 × 155.

`-000` is the timepoint suffix. Verify during EDA whether any patient ID appears with
multiple timepoints — if so, splits must be grouped by patient, not by study.

### Extreme class imbalance

The three cohorts are imbalanced **12.6 : 10.1 : 1**. PED has 99 cases against GLI's 1251.
This drives three design choices: cohort-balanced sampling for the classifier head,
class-weighted classification loss, and GLI→PED transfer (§10.4).

### Ancillary files

- `BraTS-GLI/BraTS2023_2017_GLI_Mapping.xlsx` — maps 2023 case IDs to legacy 2017 IDs.
  **Useful for leakage control**: cases present in older BraTS releases may be in public
  pretrained weights. Worth checking before using any BraTS-pretrained checkpoint.
- `BraTS-MEN/Meningioma supplementary clinical data and imaging parameters...` — clinical
  metadata and **acquisition parameters**. Directly relevant to the confound analysis in §11:
  this file may let us measure how much of the cohort signal is scanner/protocol.

---

## 4. Label semantics — read this before writing any code

### 4.1 Integer values in `seg.nii.gz` (BraTS 2023 convention)

| Value | GLI / general | MEN | PED |
|---:|---|---|---|
| 0 | background | background | background |
| **1** | NCR — necrotic / non-enhancing core | NETC — non-enhancing tumor core | **NC — non-enhancing + cystic + necrosis** |
| **2** | ED — peritumoral edema / invaded tissue | SNFH — surrounding non-enhancing FLAIR hyperintensity | ED — peritumoral edema |
| **3** | **ET — GD-enhancing tumor** | ET — enhancing tumor | ET — enhancing tumor |

**Label 3 is ET. Not label 4.** (See §2.4.)

### 4.2 Region composition — identical across all three cohorts

```
WT (whole tumor)  = {1, 2, 3}
TC (tumor core)   = {1, 3}
ET (enhancing)    = {3}
```

These regions are **nested**. The correct formulation is therefore **3-channel multi-label
with sigmoid activation**, not 4-class softmax. The BraTS 2023 winners' configs do exactly
this. A sigmoid formulation additionally lets you compute loss over only the regions that
are present, which matters because many PED cases have genuinely empty ET and/or ED.

### 4.3 Cohort-specific facts that change the design

**PED.** Annotators produced four sub-regions (ET, NET, **cystic component**, ED), but the
2023 release collapses cystic component and necrosis into label 1 — explicitly so that teams
could reuse adult glioma data. Two consequences:
- **Direct support for our joint-training decision** from the challenge organizers themselves.
- PED papers from 2024/2025 break cystic component back out as a separate label. **Their label
  sets are not ours.** Do not port their post-processing verbatim.

For diffuse midline glioma / DIPG, necrosis is rare or unclear and the tumor **may not enhance
at all**. Many PED cases have empty ET ground truth. Under the lesion-wise metric, correctly
predicting *nothing* scores Dice 1.0, and hallucinating a small blob scores 0. This makes
volume-ratio gating on ET (§14) not a hack but the correct response to the metric.

**MEN.** Two things:
- "Whole tumor" is a misnomer — SNFH is vasogenic edema that typically contains no tumor.
  Empirically WT is *harder* than ET/TC for meningioma, inverted relative to glioma.
- **90.3% of MEN cases (1286/1424) have tumor voxels touching the edge of the skull-stripped
  brain mask.** Meningiomas are extra-axial and extend through the skull; skull-stripping
  deleted extracranial tumor. Expect truncated tumors at the mask boundary and do **not**
  treat boundary-clipped predictions as errors. This is also a confound vector (§11).

### 4.4 Channel ordering trap

The MONAI model-zoo `brats_mri_segmentation` bundle outputs channels in **TC, WT, ET** order,
whereas BraTS reporting convention is **ET, TC, WT**. If we borrow anything from that bundle,
this is a silent metric-scrambling bug. Fix the project's canonical order once, assert it in
code, and never reorder implicitly.

---

## 5. What the literature actually says

### 5.1 BraTS 2023 podium

| Sub-challenge | 1st | 2nd | 3rd |
|---|---|---|---|
| GLI | Ferreira et al. — nnU-Net + Swin UNETR ensemble + GAN/registration synthetic data | NVAUTO (MONAI Auto3DSeg / SegResNet) | BiomedMBZ (SegResNet + MedNeXt) |
| MEN | **NVAUTO (Auto3DSeg / SegResNet)** | blackbean (STU-Net) | CNMC_PMI2023 (nnU-Net + Swin UNETR) |
| PED | **CNMC_PMI2023 (nnU-Net + Swin UNETR)** | NVAUTO (Auto3DSeg / SegResNet) | SherlockZyb (nnU-Net, self-supervised) |

**Every podium finish is a CNN encoder-decoder.** No transformer-primary method won anything.
The two teams that used Swin UNETR used it strictly as an *ensemble partner*; standalone it was
dramatically worse (CNMC, MEN validation ET: Swin UNETR 0.640 vs nnU-Net 0.818).

This validates SegResNet as primary and demotes Swin UNETR to secondary/ensemble — matching the
chosen architecture pair, but with the expectation ordering made explicit.

MONAI's own Auto3DSeg/SegResNet took **1st in MEN, 2nd in GLI, 2nd in PED, 1st in Brain
Metastasis, 1st in BraTS-Africa** with an essentially unchanged recipe across all five. That is
strong evidence the MONAI stack is competitive here and that a single recipe generalizes.

### 5.2 The lesion-wise metric changes everything

BraTS 2023 replaced legacy overlap Dice with **lesion-wise Dice/HD95**:

```
LW Dice = Σ Dice(lesion_i) / (TP + FN + FP)
```

- Ground-truth lesions are isolated by dilating the GT mask and running 3D connected components.
- GT lesions < 50 voxels are excluded from scoring.
- A predicted component counts as TP if it overlaps GT by ≥ 1 voxel.
- **Every FP and FN scores Dice = 0 and HD95 = 374 mm.**

Because FPs appear in the denominator *and* contribute a zero to the numerator, one spurious
60-voxel blob in a single-lesion case drops that case from ~0.95 to ~0.475. Under legacy Dice
the same blob costs a fraction of a percent.

**BraTS 2023 leaderboards look worse than 2021's because the metric changed, not because models
got worse.** The cleanest published demonstration, from BiomedMBZ, on *identical predictions*:

| Post-processing | LW Dice | Legacy Dice |
|---|---:|---:|
| none | 78.08 | 90.60 |
| + 8-flip TTA | 81.25 | 90.70 |
| + component size filter | 87.74 | 90.91 |
| + confidence filter | 87.66 | 91.04 |
| **+ both** | **88.05** | 90.94 |

**Legacy Dice moved +0.34. Lesion-wise Dice moved +9.97.** This is the most important
single fact in this document.

### 5.3 Multi-task seg + classification

Precedent for one shared 3D encoder driving both a segmentation decoder and classification
heads is well established for glioma:

- **van der Voort et al. 2020** (arXiv 2010.04425) — single 3D CNN, shared trunk, seg decoder +
  IDH/1p19q/grade heads. 1508 train / 240 test patients across 29 institutes. IDH AUC 0.90,
  WT Dice 0.84.
- **MTS-UNET** (arXiv 2503.06828) — Multi-Task **Swin UNETR**, closest architectural match to
  our secondary backbone. Introduces *Tumor-Aware Feature Encoding* (TAFE): multi-scale
  tumor-focused pooling rather than plain global average pooling. Dice 84%, IDH AUC 90.6%.
  Ablations confirm TAFE is necessary.
- **GMMAS** (arXiv 2501.17758) — shared encoder, simultaneous seg + subtype + IDH + 1p19q,
  using **uncertainty-based** multi-task loss weighting.
- **Decuyper et al.** (arXiv 2005.11965) — the *decoupled* two-stage alternative: segment first,
  then crop the tumor ROI into a separate classifier. Useful as a baseline to beat.
- **SegResNet's** own VAE regularization branch (arXiv 1810.11654) is already a second head on
  the encoder bottleneck — architectural precedent for exactly what we intend, in exactly the
  network we chose.

**Honest caveat: none of these papers runs the ablation that matters to us.** They report
classification AUCs and a respectable Dice, but no matched single-task-vs-multi-task
segmentation comparison. Whether the classification head *costs* Dice is an open question we
must answer ourselves — hence the mandatory λ=0 ablation in §17 Phase 4.

### 5.4 Cross-cohort transfer

**Every top team trained one model per cohort.** No podium finisher submitted a joint model.
But this was partly *forced*: PED 2023 rules explicitly prohibited additional data and
pretrained models, "to allow for a direct and fair comparison." **So the PED leaderboard tells
us almost nothing about whether GLI→PED transfer works.**

Evidence that it does:

- **NVAUTO** trained everything from scratch *except* BraTS-Africa, "which is very small.
  There, we initialized models from the checkpoints trained on the Glioma segmentation
  subtask." They won that sub-challenge. Direct evidence from the strongest team that
  GLI-pretrain → small-cohort-finetune is right when permitted.
- **DA-nnUNet** (arXiv 2406.16848) — unsupervised domain adaptation GLI (n=1251) → PED (n=99)
  via gradient-reversal domain classifier, **no target labels**. ~32% better TC Dice than
  adult-only training, and **not statistically distinguishable from the supervised upper bound
  on TC**. Code at `github.com/Fjr9516/DA_nnUNet`. This is the strongest published evidence
  on our exact question.
- **A New Logic For Pediatric Brain Tumor Segmentation** (arXiv 2411.01390) — a PED-designed
  model reaching 0.877 WT Dice on adult glioma, i.e. genuine transfer in both directions.

**Net read: our joint-model decision is defensible and well-supported, but it is a deviation
from what won the challenge.** The plan therefore keeps per-cohort fine-tuning as an explicit
Phase 6 option (§17) so we can measure the cost of jointness rather than assume it away.

### 5.5 Self-supervision and foundation models: not decisive here

- SSL and foundation models won nothing in BraTS 2023.
- **SherlockZyb** (PED 3rd) is listed as "nnU-Net (self-supervised)" and had by far the best
  WT HD95 (6.11 ± 4.50 vs 18–24 for everyone else) — suggesting SSL markedly reduced spurious
  lesions. **Their method paper could not be located**; likely in the MICCAI 2023 BrainLes
  LNCS proceedings rather than on arXiv. This is the most valuable literature gap to close.
- **STU-Net** (TotalSegmentator-pretrained, up to 1.4B params) placed 2nd MEN / 4th PED — real,
  but did not beat from-scratch SegResNet. Its weights are **CT**-pretrained; transfer to
  4-channel brain MRI needs stem surgery.
- **SAM-Med3D** and **VISTA3D** do not appear in any BraTS 2023 podium submission.

**Do not expect a promptable/interactive 3D foundation model to be competitive with a tuned
SegResNet ensemble on this benchmark.** The highest-leverage "extra data" trick in BraTS 2023
was not SSL but **synthetic data** — the GLI winner's entire contribution was 23,049
registration-warped plus 23,049 GAN-generated samples (~2 weeks of compute). Noted as a
stretch goal, not a core plan item.

### 5.6 Confounding and shortcut learning

Deferred to §11, because it is the intellectual crux of the classification task rather than
background reading.

---

## 6. Compute architecture

**Target: Snowflake SPCS GPU compute pool**, account `SPECTRUMHEALTH-ANALYTICS`.

### 6.1 Available GPU instance families (queried live from the account)

| Instance family | GPU | Count | VRAM | vCPU | RAM | Notes |
|---|---|---:|---:|---:|---:|---|
| `GPU_NV_S` | NVIDIA A10G | 1 | 24 GB | 6 | 27 GB | backs `SYSTEM_COMPUTE_POOL_GPU` |
| `GPU_NV_M` | NVIDIA A10G | 4 | 24 GB | 44 | 178 GB | |
| `GPU_L40S_G1_16` | NVIDIA L40S | 1 | 48 GB | 14 | 116 GB | |
| **`GPU_R6K_G1_16`** | **NVIDIA RTX PRO 6000** | 1 | **96 GB** | 14 | 116 GB | **recommended** |
| `GPU_R6K_G1_48` | NVIDIA RTX PRO 6000 | 2 | 96 GB | 44 | 490 GB | for ensembles |
| `GPU_NV_L` | NVIDIA A100 | 8 | 40 GB | 92 | 1112 GB | |
| `GPU_NV_XL` / `2XL` | H100 / H200 | 8 | 80 / 141 GB | 188 | 1843 GB | **by reservation only** |

**Recommendation: `GPU_R6K_G1_16`.** 96 GB of VRAM removes memory as a design constraint
entirely — no gradient checkpointing needed, batch size 4–8 at 128³ instead of 1–2, and
Swin UNETR becomes comfortable rather than marginal. Its 14 vCPU also matters: 3D augmentation
is CPU-bound, and the 6 vCPU on `GPU_NV_S` would starve the GPU.

The pre-existing `SYSTEM_COMPUTE_POOL_GPU` (`GPU_NV_S`, A10G 24 GB, currently SUSPENDED) is the
zero-setup fallback and is adequate for Phase 1–2 smoke tests.

**Cost is unverified.** Per-credit rates for these families must be checked against the
Snowflake Service Consumption Table before committing to a long run. Budget explicitly, set
`AUTO_SUSPEND_SECS` aggressively, and attach the pool to a Snowflake budget.

### 6.2 Storage — a real constraint

Compute pool nodes have only **93.13 GiB of local storage**. A naive float32 preprocessed cache
would exceed this:

```
cropped 160³ × 4 channels × 4 bytes  ≈  65 MB/case
× 2350 cases                         ≈  154 GB   ✗ does not fit
```

Three mitigations, in order of preference:

1. **float16 + tight per-case cropping + uint8 labels** → ~37 MB/case → ~87 GB. Fits, but with
   almost no headroom. Risky.
2. **SPCS block storage volume** for the cache, sized independently of node storage.
   **This is the recommended approach.**
3. **Stage-mount the `.nii.gz` files and decode on the fly.** 14 vCPU with
   `PersistentDataset`-style caching may be sufficient. Simplest; benchmark it first, because
   if throughput is adequate this removes the whole cache-sizing problem.

Note SPCS limits stage volume mounts to **8 per compute pool node**.

### 6.3 Platform split — and why the Windows caveats mostly evaporate

| Where | Role |
|---|---|
| **Local Windows workstation** | Code authoring, git, EDA on ~10 cases, QC screenshots, job submission |
| **SPCS GPU container (Linux)** | All preprocessing, training, inference, metric computation |

This matters more than it looks. SPCS containers are **Linux**, so the well-known Windows
deep-learning constraints do not apply to the training path:

- `torch.distributed` NCCL backend is unsupported on Windows (gloo only, prototype status).
  On Linux SPCS, NCCL works — so multi-GPU on `GPU_R6K_G1_48` or `GPU_NV_L` is available if
  needed.
- Windows lacks `fork`, so DataLoader workers `spawn`, re-import the parent module, and
  re-pickle the dataset every epoch. On Linux this is a non-issue and `num_workers` can be set
  to ~12 rather than ~4.
- MONAI `CacheDataset` duplicates its RAM cache per spawned worker on Windows. On Linux,
  copy-on-write `fork` makes `CacheDataset` viable.
- `torch.compile` on Windows requires MSVC + a Triton Windows wheel and has historically been
  unreliable. On Linux it is straightforward.

**Design implication: do not architect around Windows limitations.** Write Linux-first code and
accept that the local workstation runs only tiny CPU smoke tests. Guard entry points with
`if __name__ == "__main__":` anyway so local smoke tests work.

### 6.4 Job submission

**Snowflake ML Jobs** (`snowflake-ml-python`) is the intended mechanism: decorate a Python
entry point, submit to the compute pool, stream logs back. Checkpoints and predictions written
to an internal stage.

Do **not** compile the `sliding_window_inference` path — `mode="reduce-overhead"` uses CUDA
graphs, which interact badly with the dynamic shapes sliding-window inference produces.

Checkpoint to the stage every epoch. SPCS maintenance windows (Sat/Sun 20:00–08:00 local) can
**cancel running job services**, and Snowflake will not restart them. Every long training run
must be resumable from a stage checkpoint. This is not optional.

---

## 7. Environment and libraries

### 7.1 Core stack

| Library | Purpose | Notes |
|---|---|---|
| **PyTorch** | framework | Get the CUDA wheel index URL from pytorch.org — the default CUDA build changes often. Install **before** anything else. |
| **MONAI 1.6.0** | 3D medical imaging | Released 2026-06-10. `SegResNet`, `SegResNetDS`, `SwinUNETR`, `DynUNet`, `DiceCELoss`, `DiceFocalLoss`, `sliding_window_inference`, `CacheDataset`/`PersistentDataset`, Auto3DSeg. |
| **nibabel** | NIfTI I/O | MONAI's default reader; also used directly for EDA/QC |
| **panoptica** | instance-wise 3D metrics | v2.1.7, actively maintained (commits within the week) |
| **BraTS-2023-Metrics** | official lesion-wise metric | Git clone only — **no pip package** |
| **snowflake-ml-python** | job submission to SPCS | |
| **TensorBoard** | experiment tracking | MONAI has first-class handlers |
| SimpleITK | *optional* | only if N4 bias correction or registration is needed |
| MedNeXt | *optional, Phase 6* | standalone `nn.Module`, native gradient checkpointing |

### 7.2 Install (Linux container)

```bash
# 1. PyTorch FIRST. Verify the CUDA suffix at pytorch.org/get-started/locally
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu<XXX>

# 2. MONAI with only the extras we need (avoid monai[all] — pulls openslide/cucim)
pip install "monai[nibabel,tqdm,einops,tensorboard]==1.6.0"

# 3. Evaluation
pip install panoptica
git clone https://github.com/rachitsaluja/BraTS-2023-Metrics.git external/brats_metrics

# 4. Snowflake
pip install snowflake-ml-python snowflake-connector-python

# 5. Utilities
pip install numpy scipy scikit-learn pandas matplotlib tensorboard pyyaml
pip install openpyxl        # to read the .xlsx metadata files
pip install connected-components-3d   # fast CC for post-processing
```

Local workstation needs a Python install first (3.11 or 3.12; nnU-Net requires ≥3.10). Prefer
`uv` or Miniforge over the Microsoft Store Python shim, which is currently the only thing on PATH.

### 7.3 MONAI 1.6.0 specifics that will bite

Drawn from the 1.6.0 changelog — these are behavior changes, not trivia:

- **`sliding_window_inference` gained strict shape validation** (PR #8645). Calls that silently
  tolerated mismatched shapes in 1.5.x now raise.
- **`SwinUNETR` has spatial shape constraints** (PR #8817) — spatial dims must be divisible by
  the patch × window product, commonly multiples of 32. A frequent source of runtime shape
  errors. Read the docstring before choosing ROI size.
- **`PersistentDataset` gained `track_meta` / `weights_only`** for MetaTensor support (PR #8628).
- **`NibabelReader` no longer makes an eager C-order copy** (PR #8825) — a genuine
  memory/throughput win for large NIfTI volumes.
- `MaskedDiceLoss` changed activation/masking execution order (PR #8704).
- `GeneralizedWassersteinDiceLoss` batch-size broadcasting bug fixed (PR #8744).
- New `AUC-Margin` loss (PR #8719) — potentially useful for the imbalanced classifier head.

`monai.apps.nnUNet` / `nnUNetV2Runner` exists and is maintained (PR #8716), but **its 1.6.0
constructor signature is unverified** — check `docs.monai.io/en/stable/apps.html` before
writing code against it.

### 7.4 Deliberately excluded

- **nnU-Net v2** — would likely give the best pure-segmentation Dice, and Windows is now
  officially supported. But it is architecturally committed to supervised semantic
  segmentation with no first-class classification-head extension point; multi-task work
  requires a custom trainer subclass against a fast-moving `master` (only 4 tagged releases,
  latest tag ~2 years old and far behind `master`). **Kept as an optional Phase 6
  segmentation-only reference number, not the main line.**
- **STU-Net** — dormant ~2 years; pins `torch==1.10` / `nnUNet==1.7.0`; weights are CT-only and
  distributed via Google Drive / Baidu. Expect breakage.
- **MedNeXt** — dormant ~2 years, built on nnU-Net **v1**. Usable as a standalone `nn.Module`
  (`create_mednext_v1`), which is how we would use it in Phase 6 if at all.
- **MONAI model-zoo `brats_mri_segmentation` bundle** — trained on **BraTS 2018**, not 2023;
  channel order TC/WT/ET; untouched for 2–3 years. Reference only, and note the label-4
  problem (§2.4) applies to its weights.

---

## 8. Data pipeline

### 8.1 Stage 0 — extract and inventory (local, one-time)

1. Extract all six archives to a **non-OneDrive** path.
2. Build a manifest CSV: `case_id, cohort, timepoint, split_source, path_t1n, path_t1c,
   path_t2w, path_t2f, path_seg`.
3. **Integrity checks — fail loudly:**
   - all 5 files present for every training case, 4 for every validation case
   - all 4 sequences and the label share identical shape and affine per case
   - shape is 240 × 240 × 155 (log any deviation rather than silently resampling)
   - label volumes contain **only** values ⊆ {0, 1, 2, 3} — **assert no 4s**, which directly
     tests §2.4 and §4.1 against the actual data rather than trusting the literature
   - report per-cohort label-value histograms and per-region volume distributions
   - count cases with **empty ET** and **empty ED**, per cohort (expect many in PED)
   - check whether any patient ID appears at more than one timepoint

### 8.2 Stage 1 — EDA (local, ~10 cases + manifest statistics)

- Intensity distributions per sequence per cohort — **this is also the first confound probe**
  (§11). If GLI/MEN/PED are separable from a histogram alone, the classification task is
  substantially confounded and we learn it on day one for free.
- Brain-mask volume and bounding-box distributions per cohort — the skull-stripping shape
  shortcut vector.
- Tumor volume distributions per region per cohort.
- Count MEN cases with tumor voxels on the brain-mask boundary (expect ~90%).
- Overlay montages for visual QC.

### 8.3 Stage 2 — preprocessing

Follow the winners: minimal, well-understood preprocessing.

1. **Stack** the 4 sequences into a 4-channel volume. Fix and assert channel order once
   (proposal: `t1n, t1c, t2w, t2f`).
2. **Crop** to the nonzero brain-mask bounding box (with a small margin) — cuts volume ~40%.
3. **Z-score normalize per channel over nonzero voxels only**, leaving background at 0. This
   is what every top team did. Do **not** normalize over the whole volume including background.
4. **Convert labels** to 3 nested binary channels (ET, TC, WT) per §4.2.
5. **Do not resample.** Data is already 1 mm³ isotropic in SRI24 space. Note MedNeXt's warning
   that nnU-Net-style median-spacing resampling is untested and may degrade performance.
6. **Do not** N4 bias-correct or skull-strip — already done by FeTS.
7. Cache as float16 images + uint8 labels (§6.2).

Consider **deferring harmonization**. It is tempting as a confound fix, but the harmonization
literature warns that ComBat applied to the whole dataset before ML **causes leakage and
falsely inflates performance** (arXiv 2211.04125, 36 datasets / 1740 subjects). If we harmonize
at all, it must be fit **inside each CV fold**. Treat as a Phase 5 experiment, not baseline
preprocessing.

### 8.4 Stage 3 — augmentation (training only)

Matching the winners' recipes:

- `RandSpatialCropd` / `RandCropByPosNegLabeld` to the training patch size
- random flips on all three axes
- random affine (rotation + scaling)
- random intensity scale and shift
- random Gaussian noise and blur
- *(optional, Phase 5)* random gamma, elastic deformation, per-channel dropout to build
  robustness to missing sequences

---

## 9. Data splits

### 9.1 Structure

The official ValidationData is unusable for scoring (§2.2), so **all splits derive from the
2350 labeled TrainingData cases**:

```
labeled TrainingData (2350)
├── train  80%  ~1880   → gradient updates
├── val    10%   ~235   → hyperparameters, checkpoint selection, post-proc thresholds
└── test   10%   ~235   → touched ONCE, at the very end
```

Per-cohort: GLI ≈ 1000/125/126, MEN ≈ 800/100/100, PED ≈ 79/10/10.

**PED test at n=10 is statistically near-useless for a point estimate.** Report bootstrap
confidence intervals, never a bare PED number. For context, the official PED test set was only
n=24 and the top four teams were statistically indistinguishable (p = 0.10–0.45). Treat PED
differences under ~0.05 Dice as noise.

### 9.2 Rules

- **Stratify by cohort**, so all three appear in all three splits.
- **Group by patient ID**, not study — if multiple timepoints exist, all go to one split.
- **Fixed seed, splits committed to the repo as CSV.** Never regenerated.
- The **test split is opened exactly once**, after all hyperparameters and post-processing
  thresholds are frozen on validation. Every peek invalidates it.
- 5-fold CV on train+val is the Phase 6 upgrade for ensembling; the single split is for
  Phases 1–5 iteration speed.

### 9.3 Site-held-out splits — required, not optional

**In addition** to the random split above, construct a **site-held-out** split for the
classification evaluation, using whatever site/institution information can be recovered from
the MEN supplementary metadata and the GLI legacy mapping. Reasons:

- A random split lets the model memorize site-specific acquisition signatures and report an
  optimistically high classification accuracy that will not transfer.
- The PED challenge itself demonstrated this: train/val were CBTN + Boston, but **all 24 test
  cases came from Yale**. NVAUTO saw internal 5-fold Dice 0.9196 collapse to 0.8236 on the
  hidden test set.

If site labels cannot be recovered for a cohort, **say so in the results** rather than quietly
reporting only the random-split number.

---

## 10. Model design

### 10.1 Shared backbone

```
                                    ┌─────────────────────────┐
 4-ch patch                         │ segmentation decoder    │→ 3 ch (ET,TC,WT), sigmoid
 (128³)  ──→  ENCODER  ──→ bottleneck │ + deep supervision     │
              (SegResNet)           └─────────────────────────┘
                  │
                  └──→ ┌──────────────────────────┐
                       │ classification head      │→ 3 logits (GLI/MEN/PED), softmax
                       │ mask-conditioned pooling │
                       └──────────────────────────┘
```

**Primary: `SegResNet` / `SegResNetDS`** (MONAI). BraTS-native, memory-efficient, deep
supervision, batch norm. Its existing VAE branch is precedent for a second head on the
bottleneck.

**Secondary: `SwinUNETR`.** Same two-head structure. Expect it to underperform SegResNet
standalone (§5.1) and to earn its place only in an ensemble. Mind the spatial shape
constraints (§7.3).

### 10.2 Segmentation head

- 3 output channels, **sigmoid**, nested regions per §4.2.
- **Deep supervision** with loss weights `1/2^i` for sub-level `i` (3 outputs with the shipped `blocks_down=(1,2,2,4)`), targets
  downsampled nearest-neighbor. Used by every top team.

### 10.3 Classification head — mask-conditioned pooling, not plain GAP

This is the most consequential architectural decision in the project, and it is driven
entirely by §11.

Three options, in increasing robustness:

| Option | Mechanism | Confound risk |
|---|---|---|
| A. Plain GAP | global average pool over bottleneck → linear | **High.** At a 128³ patch the bottleneck is 16³ (three downsamplings); a flat GAP is dominated by whole-brain/background statistics, which is exactly the channel through which cohort identity leaks |
| B. Mask-conditioned pooling | pool encoder features **only where the predicted tumor mask is positive** | **Low.** Restricting the classifier's receptive field to tumor voxels removes the dominant leakage path (brain contour, FOV, skull-strip geometry) |
| C. TAFE-style attention pooling | multi-scale tumor-focused attention pooling (MTS-UNET, arXiv 2503.06828) | Low, and better accuracy; more complex |

**Plan: implement B as the default, A as an explicit ablation to quantify how much the confound
is worth, C as a Phase 5 upgrade.**

Deliberately **excluded**: fusing age or other clinical covariates. Chakrabarty et al.
(arXiv 2210.03779) show this helps in general — but the PED cohort is *defined by age*, so
adding age would inject precisely the confound we are trying to control. This is a case where
a known-good technique from the literature is wrong for our specific question.

Because the classifier is volume-level but training is patch-level, resolve the mismatch
explicitly: aggregate patch logits per volume, or run the classification head only on
sliding-window inference at validation. Note that patch sampling itself leaks cohort
information via patch location statistics — document the chosen aggregation.

### 10.4 Cohort conditioning and PED transfer

The classifier makes the model cohort-*aware*. Whether to also make the segmentation decoder
cohort-*conditioned* (e.g. FiLM on a cohort embedding) is a Phase 5 experiment, since it risks
the segmentation inheriting the classifier's confound.

For PED specifically, run the literature-supported path (§5.4): **pretrain on GLI, then
fine-tune on PED**, and consider a **PED ET-only specialist** to ensemble in — both PED podium
finishers did this independently. Gradient-reversal domain adaptation (arXiv 2406.16848) is a
Phase 6 stretch goal.

---

## 11. The classification task is confounded — mandatory protocol

### 11.1 The problem, stated plainly

In pooled BraTS 2023, the tumor-type label **is** the cohort folder the case came from. The
three cohorts were assembled by different consortia, at different institutions, on different
scanners, with different protocols, on different patient populations (PED is defined by age),
and passed through defacing/skull-stripping with cohort-specific behavior.

**A near-perfect tumor-type accuracy should be treated as evidence of confounding, not as
evidence of a good model, until proven otherwise.** This is not hedging; the mechanisms are
documented for this exact dataset.

### 11.2 Evidence that the shortcut is real, not hypothetical

- **Tinauer et al., arXiv 2501.15831** (Oct 2025). A 3D CNN classifying Alzheimer's from 990
  matched ADNI T1w scans retained **full accuracy on binarized images** — proving it was not
  using tissue texture at all. Layer-wise Relevance Propagation showed it relied on **brain
  contours introduced by skull-stripping**. The authors call it a Clever Hans effect.
- **The BraTS 2023 meningioma challenge's own analysis paper (arXiv 2405.09787)** reports that
  **1286 of 1424 MEN cases (90.3%) have tumor voxels abutting the edge of the skull-stripped
  image**, and calls for "further investigation into optimal pre-processing face anonymization
  steps." Cohort-specific edge behavior is documented by the organizers.
- **Shortcut learning is not confined to classification heads** — arXiv 2403.06748 documents it
  in segmentation.
- Site/batch effects are large and well characterized — arXiv 2507.16962 (harmonization survey)
  concludes site invariance is achievable but **preservation of biological information remains
  under-verified**. Harmonization is not a free fix.

Combine the first two points: the brain-mask shape is a plausible near-perfect cohort
fingerprint for this dataset. Mask-conditioned pooling (§10.3 option B) exists specifically to
close that channel.

### 11.3 Required diagnostics — cheap and decisive, run them early

| Test | What it proves | Reference |
|---|---|---|
| **Intensity-histogram cohort classifier** | If a classical model on histograms alone separates cohorts, the task is confounded at the intensity level. Free — falls out of §8.2 EDA. | — |
| **Brain-mask-shape cohort classifier** | Predict cohort from mask geometry alone (volume, bbox, surface area), with no tumor information. Tests the Tinauer pathway directly. | 2501.15831 |
| **Binarization control** | Retrain the real model on intensity-binarized volumes. **If accuracy survives, the model is reading shape/geometry, not tumor phenotype.** | 2501.15831 |
| **Site probe on the encoder embedding** | Train a site/cohort classifier on the learned embedding. Near-perfect predictability ⇒ the tumor-type head is very likely reading cohort. Standard failure metric in the harmonization literature. | 2402.06875, 2601.08193 |
| **Relevance maps on the classifier head** | LRP + spectral clustering. If relevance concentrates on brain contour or background rather than lesion, the answer is in. | 2501.15831 |
| **Label-permutation control** | Run the full pipeline with permuted cohort labels to expose pipeline-induced optimism. The "Same Analysis Approach." | 1703.06670 |
| **Report a Confounding Index** | Gives a single reportable scalar rather than a narrative caveat. | 1905.08871 |
| **DeepRepViz** | Purpose-built framework for identifying confounders in DL predictions from learned representations. Most directly reusable tool available. | 2309.15551 |

### 11.4 Mitigations if the diagnostics come back bad

- Mask-conditioned pooling (§10.3 B) — already the default.
- **Site-held-out splits** (§9.3).
- Adversarial confound regression / gradient reversal on a site head (arXiv 2205.02885).
- Dependence-measure minimization between representation and confounder — arXiv 2407.18792
  benchmarks *which* dependence measure actually works, which is the practical question.
- Fold-internal harmonization only (arXiv 2211.04125).

### 11.5 How this gets reported

**The classification head is framed as an auxiliary/regularizing task with an explicitly
stated confound caveat — not as a clinical tumor-typing claim.** If the diagnostics show the
signal is largely acquisition-driven, that is a legitimate and interesting finding to report,
not a failure to hide. Reporting "99.4% tumor-type accuracy" without §11.3 would be
scientifically indefensible, and it is exactly the number this dataset will hand us.

---

## 12. Loss design

```
L_total = L_seg + λ · L_cls
```

**`L_seg`** — Dice + Focal, or Dice + CE. Both podiumed with no clear winner (NVAUTO and
BiomedMBZ used Dice+Focal; CNMC/nnU-Net used Dice+CE). Start with
Dice + **per-channel binary cross-entropy** on the 3 nested sigmoid channels (`SegLoss("dice_ce")`), then
try `DiceFocalLoss`. Do **not** use MONAI's `DiceCELoss` here: with more than one output channel its CE
term is a softmax *across* the channels, so a background voxel contributes zero loss and the nested
regions compete for one unit of probability (verified numerically; see `brats/losses.py`).
Use **batch Dice** rather than per-sample Dice — the winners did, and it stabilizes cases with
empty ET. Summed over deep-supervision levels with `1/2^i` weights.

**`L_cls`** — cross-entropy with class weights inverse to cohort frequency (12.6 : 10.1 : 1),
and/or cohort-balanced batch sampling. MONAI 1.6.0's new AUC-Margin loss is worth an
experiment.

**Weighting λ.** Dice and CE live on different scales, so this matters.

1. **Fixed λ, small sweep** (e.g. 0.05 / 0.1 / 0.3 / 1.0). Start here.
2. **Homoscedastic uncertainty weighting** — Kendall, Gal & Cipolla (arXiv 1705.07115); learns
   a per-task log-variance. This is what GMMAS (arXiv 2501.17758) adopted for exactly our task
   family. Second experiment.
3. **GradNorm** (arXiv 1711.02257) — deprioritized. A direct three-way comparison
   (arXiv 2607.03304) found GradNorm "consistently underperformed" on one of two tasks and
   incurred higher compute and memory cost, with the best strategy depending on backbone.

**λ = 0 is a required ablation**, not an afterthought — it is the only way to answer whether the
classification head costs segmentation Dice, and §5.3 establishes that no published paper
answers this for us.

**Numerics.** Prefer **bf16** over fp16 (RTX PRO 6000 and A10G are both Ampere-or-newer). bf16
needs no `GradScaler` and is far more forgiving for Dice-family losses, whose small denominators
are a known fp16 NaN source. Mixing a Dice-scale and a CE-scale loss under fp16 makes the
scaler's job harder — another reason for bf16.

---

## 13. Training recipe

Baseline synthesized from the BraTS 2023 winners, adapted for `GPU_R6K_G1_16` (96 GB).

| Hyperparameter | Value | Source / note |
|---|---|---|
| Patch size | **128³** | CNMC (PED), BiomedMBZ; also 128×160×112 (Ferreira, CNMC MEN) |
| Batch size | **4–8** | 96 GB VRAM lifts the usual 1–2 limit; Ferreira used 5 on 128×160×112 |
| Normalization | z-score, nonzero voxels only | all top teams |
| Loss | Dice+CE or Dice+Focal, batch Dice, deep supervision `1/2^i` | §12 |
| Optimizer | **AdamW**, LR **2e-4**, wd **1e-5** | NVAUTO (SegResNet lineage) |
| Alt. optimizer | SGD+Nesterov, LR 0.01, mom 0.99, wd 3e-5, poly LR | nnU-Net lineage |
| Schedule | **cosine annealing to zero**, linear warmup ~8 epochs | NVAUTO; BiomedMBZ |
| Epochs | **150** dev → **600** final | BiomedMBZ 150; NVAUTO/CNMC-Swin 600 |
| Precision | **bf16** autocast | §12 |
| Norm layer | batch norm (batch ≥ 4 makes this viable) | Ferreira chose BN over GN once batch 5 was affordable |
| Augmentation | affine, flips ×3 axes, intensity scale/shift, noise, blur | NVAUTO |
| `num_workers` | ~12 (Linux) | §6.3 |
| Checkpointing | **every epoch to Snowflake stage** | SPCS maintenance can kill jobs (§6.4) |
| Checkpoint selection | keep best-average **and best-ET, best-TC, best-WT separately** | NVAUTO: "the best average checkpoint may not be the best in all 3 sub-regions" |

**Batch size is a genuine open question, not a settled value.** Every published recipe used
batch 1–5 because of 16–48 GB VRAM limits — nobody tuned these LR/schedule values at batch 8.
Larger batches will likely need LR rescaling. Treat batch size as an early sweep, and do not
assume the literature's LR transfers.

---

## 14. Inference and post-processing — the single highest-leverage stage

**Read §5.2 first.** Post-processing is worth ~+10 lesion-wise Dice; architecture choice among
top backbones is worth ~+0.01. **This section deserves more engineering time than §10 and §13
combined.** This is the plan's central strategic claim.

### 14.1 Inference

- **`sliding_window_inference`**, window = training patch size, **overlap 0.5**,
  `mode="gaussian"`.
- **TTA: sweep, do not assume.** The evidence is genuinely split — BiomedMBZ gained from 8-flip
  TTA (+3.17 LW Dice), while Ferreira found **disabling** nnU-Net's TTA gave better validation
  results and was ~8× faster, letting them ensemble more models instead. Measure it.

### 14.2 Post-processing, in priority order

1. **Per-channel probability thresholds — not 0.5 everywhere.** BiomedMBZ's final values were
   **(TC 0.5, WT 0.5, ET 0.4)**. Cheap, and tune on validation.
2. **Connected-component filtering on joint size + mean confidence.** This is the big one.
   BiomedMBZ's `FilterObjects` keeps a component if
   `(size ≥ S_upper AND mean_prob ≥ P_upper) OR (S_lower ≤ size < S_upper AND mean_prob ≥ P_mid)`.
   Their final values: `WT(2000, 100, 0.85, 0.925)`, `ET(95, 70, 0.71, 0.5)`, `TC(350, 350, 0, 0)`.
   **Filtering on size *and* confidence beat size alone**, because a size-only cutoff discards
   small-but-confident true lesions.
3. **Simple per-region voxel thresholds** as the baseline version of (2): Ferreira shipped
   **WT 250 / TC 150 / ET 100**; CNMC used **130 (PED) / 110 (MEN)**.
4. **PED ET volume-ratio gating.** CNMC: **if ET/WT < 0.04, relabel ET → NC**. This single rule
   took PED validation ET lesion-wise Dice from **0.466 → 0.733** and ET HD95 from 158.89 →
   75.93. **The largest single improvement anywhere in the BraTS 2023 literature.** Non-optional
   for PED.
5. **Legacy ET→NCR relabeling** (ET voxel count < ~200). Tuned for the *old* metric; worth only
   ~+0.30 LW Dice in BiomedMBZ's ablation. Low priority — and a good example of a widely-cited
   trick that the metric change made nearly obsolete.

### 14.3 Two traps

- **Validation needs larger thresholds than training.** Ferreira found this consistently. Do
  not tune small-component thresholds on training-fold predictions and ship them.
- **Do not tune to mean Dice — tune to the BraTS rank.** Ferreira reimplemented the challenge
  ranking scheme locally to choose their submission, because "selecting the solution with the
  best DSC and/or HD95 is not the best approach." A threshold of WT 1450 scored best of all on
  their validation but they judged it too risky and shipped WT 250.

---

## 15. Evaluation

### 15.1 Metrics

| Metric | Role |
|---|---|
| **Lesion-wise Dice** per region (ET/TC/WT) | **primary** — the official BraTS 2023 metric |
| **Lesion-wise HD95** per region | **primary** |
| Legacy overlap Dice / HD95 | reported alongside, for comparison with pre-2023 literature |
| **Median as well as mean**, per region | non-negotiable — see below |
| Bootstrap CIs | mandatory for PED (n≈10 test) |
| Classification: balanced accuracy, macro-F1, per-class AUC, confusion matrix | plus every §11.3 diagnostic |

**Always report median alongside mean.** The MEN winner scored mean ET Dice 0.899 but **median
0.976**, with mean HD95 23.9 mm and **median 0.96 mm**. On a typical case the segmentation is
essentially perfect; the mean is dragged by a handful of catastrophic cases (calcified
non-enhancing meningiomas — one NVAUTO case scored ET 0.00 / TC 0.00 / WT 0.338). Reporting
only the mean hides the actual behavior of the model, and chasing those cases with threshold
tuning is wasted effort.

### 15.2 Implementation

- **Official**: `github.com/rachitsaluja/BraTS-2023-Metrics` — git clone, no pip package. Use
  this for leaderboard-comparable numbers. Note the repo's **per-challenge volumetric thresholds
  and dilation factors** (`figs/BraTS_ThreshDil.png`), set by radiologists.
- **Cross-check**: `panoptica` (v2.1.7, actively maintained). **Do not silently substitute one
  for the other** — thresholds and dilation differ.
- **The organizers' own papers describe the dilation inconsistently**: the MEN paper says
  1-voxel symmetric dilation with 26-connectivity; the PED paper says dilate by 3 pixels in all
  directions. **Read the code, not the prose.**

### 15.3 Reporting

Per cohort × per region × {mean, median, IQR}, on the held-out test split, with the
single-model and ensemble numbers separated, and every post-processing stage ablated in the
style of §5.2's table. Plus predictions on the official ValidationData for qualitative review.

---

## 16. Performance targets

Calibrated to BraTS 2023 published results. **Lesion-wise unless stated.** Note our numbers
come from a self-made test split, not the hidden challenge test set, so they are not strictly
comparable to the leaderboard — expect ours to be optimistic (same-institution split).

### GLI (test ≈ 126)

| Region | Baseline (Phase 3) | Target (Phase 5) | Podium reference |
|---|---|---|---|
| WT | 0.80 | **0.86–0.90** | 0.836 (test) – 0.900 (val) |
| TC | 0.78 | **0.83–0.87** | 0.828 – 0.867 |
| ET | 0.75 | **0.81–0.85** | 0.808 – 0.851 |
| avg | 0.78 | **0.84–0.87** | 0.824–0.831 test |

### MEN (test ≈ 100)

| Region | Target mean | Target median | Podium reference (mean / median) |
|---|---|---|---|
| ET | **0.85–0.90** | **≥ 0.95** | 0.899 / 0.976 |
| TC | **0.85–0.90** | **≥ 0.95** | 0.904 / 0.976 |
| WT | **0.80–0.87** | **≥ 0.93** | 0.871 / 0.964 |

If median ≥ 0.95 and mean ≈ 0.85, we are in the *same regime as the winner* and the residual
loss is concentrated in a few pathological cases. Recognize that state and stop optimizing.

### PED (test ≈ 10 — report CIs, treat Δ < 0.05 as noise)

| Region | Target | Podium reference |
|---|---|---|
| WT | **0.78–0.84** | 0.81–0.84 |
| TC | **0.74–0.81** | 0.77–0.81 |
| **ET** | **0.50–0.73** | 0.53–0.65 |

**PED ET is the whole game** — it accounts for essentially all the spread between teams, and
it is where post-processing pays (§14.2 item 4).

### Classification

No accuracy target is set, deliberately. The meaningful deliverables are the §11.3 diagnostics
and an honest statement of how much of the signal is tumor phenotype versus acquisition
confound. **A high accuracy number here is the expected outcome and the least interesting
result.**

### Cohort difficulty ordering

Using NVAUTO's single recipe as a control across all five 2023 cohorts:

```
MEN (0.891) > BraTS-Africa (0.850) > GLI (0.824) > PED (0.721) > METS (0.625)
```

---

## 17. Execution phases

Each phase has an explicit exit criterion. Do not advance without meeting it.

### Phase 0 — Environment and data (local + Snowflake)
- Install Python + toolchain locally; verify `snow` CLI / `snowflake-ml-python` auth.
- Extract archives to non-OneDrive storage; build and validate the manifest (§8.1).
- **Run the label-value assertion** — confirm labels ⊆ {0,1,2,3} and **no 4s**.
- Provision the compute pool; verify GPU visibility and bf16 in a container; benchmark
  stage-mount read throughput vs. block-volume caching (§6.2).
- Stage data to Snowflake.
- **Exit:** a container on `GPU_R6K_G1_16` reads a case from the stage, prints its shape,
  affine, and label histogram, and reports `torch.cuda.is_available() == True`.

### Phase 1 — EDA and confound pre-screen
- §8.2 EDA. **Plus the two cheap confound probes**: intensity-histogram and brain-mask-shape
  cohort classifiers (§11.3).
- **Exit:** a written EDA note that states, with numbers, how separable the cohorts are from
  intensity and mask geometry *before any deep model is trained*. This result shapes §10.3.

### Phase 2 — Segmentation-only baseline
- SegResNet, 3-channel sigmoid, single split, short schedule (~50 epochs), no post-processing.
- Wire up official + panoptica metrics; verify they agree on a toy case.
- **Exit:** a lesion-wise Dice number per cohort per region that is plausible (WT > 0.7 on GLI),
  and a training loop that resumes cleanly from a stage checkpoint after a forced kill.

### Phase 3 — Full segmentation recipe
- §13 recipe at full length; deep supervision; per-region checkpoint tracking; TTA sweep.
- **Exit:** GLI avg LW Dice ≥ 0.78 on validation.

### Phase 4 — Add the classification head
- Mask-conditioned pooling (§10.3 B); λ sweep; then uncertainty weighting.
- **Run the λ = 0 ablation** — matched backbone, split, and schedule.
- **Run the full §11.3 diagnostic battery**, including the binarization control and site probe.
- **Exit:** a quantified answer to "does the classification head cost segmentation Dice?" and a
  quantified answer to "how much of the classification signal is confound?"

### Phase 5 — Post-processing and threshold optimization ← **highest expected return**
- Per-channel probability thresholds; joint size + confidence component filtering; PED ET
  volume-ratio gating.
- Tune on validation only. Reimplement the BraTS ranking scheme locally and tune to rank.
- Reproduce §5.2's ablation table on our own predictions.
- **Exit:** ≥ +0.05 avg LW Dice over Phase 3, from post-processing alone.

### Phase 6 — Optional strengthening, in descending value-per-effort
1. **GLI→PED fine-tuning** and a **PED ET-only specialist** (both PED podium teams did this).
2. **5-fold CV + per-sub-region ensembling** (NVAUTO's approach — keep best-ET/TC/WT separately).
3. **Swin UNETR** trained and ensembled with SegResNet.
4. Per-cohort fine-tuning from the joint checkpoint — **measures the cost of jointness** (§5.4).
5. Cohort-conditioned decoder (FiLM); TAFE-style attention pooling.
6. nnU-Net v2 as a segmentation-only reference number.
7. Fold-internal harmonization; gradient-reversal domain adaptation.
8. MedNeXt as a standalone module; synthetic-data augmentation (very expensive).

### Phase 7 — Final evaluation and write-up
- **Open the test split once.** Full metrics per §15.3.
- Predict on official ValidationData; qualitative review; optional Synapse submission.
- Report confound findings honestly per §11.5.

---

## 18. Planned repository layout

```
Brain_Tumor_Segmentation/
├── README.md                      # this document
├── pyproject.toml
├── configs/
│   ├── data.yaml                  # paths, cohorts, channel order
│   ├── segresnet_base.yaml
│   ├── swinunetr_base.yaml
│   └── postproc.yaml              # thresholds — the Phase 5 artifact
├── splits/
│   ├── split_random_seed42.csv    # committed, never regenerated
│   └── split_site_holdout.csv
├── src/brats/
│   ├── data/
│   │   ├── extract.py             # archives → local disk
│   │   ├── manifest.py            # inventory + integrity assertions (§8.1)
│   │   ├── preprocess.py          # crop, z-score, label→regions, cache
│   │   ├── transforms.py          # MONAI train/val pipelines
│   │   └── splits.py
│   ├── models/
│   │   ├── multitask.py           # backbone + seg decoder + cls head
│   │   └── pooling.py             # GAP / mask-conditioned / TAFE
│   ├── losses.py                  # seg + cls, fixed & uncertainty weighting
│   ├── train.py
│   ├── inference.py               # sliding window + TTA
│   ├── postprocess.py             # thresholds, CC filtering, PED ET gating
│   ├── metrics/
│   │   ├── lesionwise.py          # wraps official repo
│   │   └── ranking.py             # local BraTS rank reimplementation
│   ├── confound/                  # §11 — a first-class module, not a notebook
│   │   ├── probes.py              # histogram / mask-shape / embedding probes
│   │   ├── binarize_control.py
│   │   ├── relevance.py           # LRP
│   │   └── permutation.py
│   └── snowflake/
│       ├── stage_data.py
│       └── submit_job.py          # ML Jobs entry point
├── notebooks/                     # EDA and QC only, never the source of truth
├── external/
│   └── brats_metrics/             # cloned, not vendored
└── scripts/
```

Note `src/brats/confound/` is a first-class module. If the confound analysis lives in a
notebook it will not be maintained, and §11 is not optional.

---

## 19. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Label 3 vs 4** — silently empty ET | **Critical** | Hard assertion in Phase 0; label histograms in EDA |
| 2 | **Channel-order scramble** (§4.4) | **Critical** | Single canonical order, asserted in code; never reorder implicitly |
| 3 | **Classification result is confounded and uninterpretable** | **High** | §11 protocol; mask-conditioned pooling; framed as auxiliary from the outset |
| 4 | SPCS maintenance kills a long job | High | Per-epoch stage checkpoints; resumable-by-design; verified in Phase 2 exit |
| 5 | Node local storage (93 GiB) too small for cache | High | Block volume, or stage-mount + on-the-fly decode; benchmarked in Phase 0 |
| 6 | Snowflake GPU credit cost overruns | High | Verify rates before long runs; aggressive `AUTO_SUSPEND_SECS`; attach a budget |
| 7 | **PED test n≈10 → meaningless point estimates** | Medium-High | Bootstrap CIs; treat Δ < 0.05 as noise; never report a bare PED number |
| 8 | OneDrive syncs 33 GB of NIfTI / thrashes the machine | Medium | Non-synced extraction path, enforced in Phase 0 |
| 9 | Joint model underperforms per-cohort specialists | Medium | Phase 6 item 4 measures this explicitly rather than assuming |
| 10 | Test split contaminated by repeated peeking | Medium | Opened once, in Phase 7; enforced by convention and code review |
| 11 | Multi-task head degrades segmentation Dice | Medium | λ=0 ablation is a Phase 4 exit criterion, not optional |
| 12 | Patch-level training vs volume-level classification mismatch | Medium | Explicit aggregation strategy, documented (§10.3) |
| 13 | Public pretrained weights saw our test cases | Medium | Check the GLI 2017 mapping xlsx before using any BraTS-pretrained checkpoint |
| 14 | MONAI 1.6.0 strict shape validation / SwinUNETR shape constraints | Low-Medium | Pin 1.6.0; read §7.3; assert ROI divisibility |
| 15 | MEN boundary-abutting tumors misread as errors | Low-Medium | Documented in §4.3; do not "fix" boundary-clipped predictions |
| 16 | Batch size 8 invalidates the literature's LR/schedule | Low-Medium | Early LR sweep; do not assume transfer (§13) |
| 17 | Harmonization applied globally → leakage and inflated scores | Low-Medium | Fold-internal only, Phase 5+ |

---

## 20. References

### Challenge and dataset
- **Analysis of the BraTS 2023 Intracranial Meningioma Segmentation Challenge** — arXiv 2405.09787 (MELBA 2025:002). *Official MEN results; the 90.3% boundary-abutting finding.*
- **BraTS-PEDs: Results of the Multi-Consortium International Pediatric Brain Tumor Segmentation Challenge 2023** — arXiv 2407.08855. *Official PED results; label collapse; no-pretraining rule.*
- The ASNR-MICCAI BraTS Challenge 2023: Intracranial Meningioma — arXiv 2305.07642
- BraTS Challenge 2023: Focus on Pediatrics — arXiv 2305.17033
- ⚠ **No official GLI 2023 analysis paper appears to exist.** GLI podium ordering is reconstructed from three mutually consistent team self-reports.

### Winning methods
- **How we won BraTS 2023 Adult Glioma challenge? Just faking it!** — Ferreira et al., arXiv 2402.17317. *GLI 1st.*
- **Auto3DSeg for Brain Tumor Segmentation from 3D MRI in BraTS 2023** — Myronenko et al. (NVIDIA), arXiv 2510.25058. *MEN 1st, GLI 2nd, PED 2nd.*
- **Model Ensemble for Brain Tumor Segmentation in MRI** — Capellán-Martín et al., arXiv 2409.08232. *PED 1st; the ET/WT gating result.*
- **Advanced Tumor Segmentation… BraTS 2023 Adult Glioma and Pediatric** — Maani, Hashmi et al., arXiv 2403.09262. *GLI 3rd; the post-processing ablation table.*

### Architectures
- SegResNet — Myronenko, arXiv 1810.11654
- Swin UNETR — arXiv 2201.01266; SSL pretraining — arXiv 2111.14791
- UNETR — arXiv 2103.10504
- nnU-Net — Isensee et al., *Nature Methods* 18:203–211 (2021); **nnU-Net Revisited** — arXiv 2404.09556
- MedNeXt — arXiv 2303.09975
- STU-Net — arXiv 2304.06716
- **A Unified Benchmark of DL Models for Multi-task 3D Brain Tumor Segmentation** — arXiv 2607.28858. *Head-to-head 3D U-Net / SegResNet / Swin UNETR / SegMamba under identical conditions, incl. BraTS 2023 MEN.*

### Multi-task segmentation + classification
- WHO 2016 subtyping and automated segmentation of glioma using multi-task DL — arXiv 2010.04425
- **MTS-UNET / TAFE** — arXiv 2503.06828
- GMMAS (uncertainty-weighted multi-task) — arXiv 2501.17758
- Two-stage seg-then-classify baseline — arXiv 2005.11965
- 2.5D hybrid multi-task CNN with tabular fusion — arXiv 2210.03779
- MAG-Net (2D tumor-type multi-task; cautionary precedent) — arXiv 2107.12321

### Loss weighting
- **Uncertainty weighting** — Kendall, Gal & Cipolla, arXiv 1705.07115
- GradNorm — arXiv 1711.02257
- Empirical comparison of weighting strategies — arXiv 2607.03304

### Transfer and domain adaptation
- **Unsupervised Domain Adaptation for Pediatric Brain Tumor Segmentation** — arXiv 2406.16848; code `github.com/Fjr9516/DA_nnUNet`
- A New Logic For Pediatric Brain Tumor Segmentation — arXiv 2411.01390
- Unified HT-CNNs — arXiv 2412.08240

### Confounding and shortcut learning (§11)
- **Skull-stripping induces shortcut learning in MRI-based AD classification** — Tinauer et al., arXiv 2501.15831
- Shortcut Learning in Medical Image Segmentation — arXiv 2403.06748
- Harmonization in MRI: A Survey — arXiv 2507.16962
- **Efficacy of MRI data harmonization… (ComBat leakage)** — arXiv 2211.04125, *Sci Data* 11:115 (2024)
- Detect and Correct Bias in Multi-Site Neuroimaging Datasets — arXiv 2002.05049
- Benchmarking Dependence Measures to Prevent Shortcut Learning — arXiv 2407.18792
- DeepRepViz — arXiv 2309.15551
- Confounding Index — arXiv 1905.08871
- The Same Analysis Approach (permutation control) — arXiv 1703.06670
- Adversarial confound regression — arXiv 2205.02885
- DLEST — arXiv 2402.06875; MMH — arXiv 2601.08193

### Tooling
- MONAI 1.6.0 — `github.com/Project-MONAI/MONAI/releases/tag/1.6.0`
- **Official metrics** — `github.com/rachitsaluja/BraTS-2023-Metrics`
- Synapse harness — `github.com/Sage-Bionetworks-Challenges/brats2023`
- **panoptica** — `github.com/BrainLesion/panoptica`, arXiv 2312.02608
- GaNDLF — `github.com/mlcommons/GaNDLF`
- FeTS preprocessing — `github.com/FETS-AI/Front-End`
- nnU-Net v2 — `github.com/MIC-DKFZ/nnUNet`

### Open literature gaps
- **SherlockZyb** (PED 3rd, "nnU-Net self-supervised", best WT HD95 by 3×) — method paper not located; likely MICCAI 2023 BrainLes LNCS. **Highest-value gap to close.**
- **blackbean's** BraTS-specific write-up is not on arXiv; its recipe here is second-hand via the MEN organizers.
- **NVAUTO's patch size and post-processing** are not printed in their paper (Auto3DSeg auto-configures); inspect the generated config to replicate.
