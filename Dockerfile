# =============================================================================
# BraTS 2023 multi-task training image for Snowpark Container Services (SPCS).
#
# Base: Docker Hub official PyTorch image (CUDA 12.1 + cuDNN + NCCL).
# We use Docker Hub rather than NGC because corporate TLS inspection blocks
# nvcr.io. The pytorch/pytorch image ships the same CUDA/cuDNN/NCCL stack.
#
# Data is NOT baked in. The preprocessed .npz cache (~11.8 GB) lives on a
# Snowflake internal stage and is mounted at /mnt/data at runtime via the
# service spec. Checkpoints write to /mnt/ckpt (another stage mount).
#
# Build:
#   docker build -t brats-train .
#
# Local dry-run (CPU, no data):
#   docker run --rm brats-train python -m brats.train --help
#
# SPCS: push to the Snowflake image registry, then CREATE SERVICE with the
# spec in spcs/service_spec.yaml.
# =============================================================================

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

LABEL maintainer="soumbane <s.banerjee@wayne.edu>"
LABEL description="BraTS 2023 segmentation + classification training"

# Avoid interactive prompts during apt-get.
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies that nibabel and cc3d may need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python dependencies ----
# Install in a separate layer so code changes don't re-install deps.
# The Docker Hub base has torch, torchvision, numpy, tqdm.
COPY pyproject.toml /app/

# Install core deps + the torch optional group (MONAI, wandb).
# --no-build-isolation because hatchling is only needed if building the wheel;
# we install in editable-like mode via PYTHONPATH instead.
RUN pip install --no-cache-dir \
        "monai[nibabel,tqdm,einops]>=1.4" \
        "nibabel>=5.2" \
        "connected-components-3d>=3.12" \
        "scikit-learn>=1.4" \
        "scipy>=1.11" \
        "pandas>=2.1" \
        "pyyaml>=6.0" \
        "wandb>=0.17" \
        "openpyxl>=3.1" \
        "matplotlib>=3.8"

# ---- Project code ----
COPY src/      /app/src/
COPY configs/  /app/configs/
COPY splits/   /app/splits/
COPY scripts/  /app/scripts/

# Make the brats package importable.
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# ---- Runtime defaults ----
# These can be overridden in the SPCS service spec or via docker run -e.
ENV OMP_NUM_THREADS=8
ENV NCCL_DEBUG=WARN
ENV WANDB_MODE=offline

# Stage mount points (created so the paths exist even without mounts, avoiding
# FileNotFoundError if someone runs the container without SPCS).
RUN mkdir -p /mnt/data /mnt/ckpt /mnt/predictions

# Default: print help. The service spec overrides this with the actual command.
CMD ["python", "-m", "brats.train", "--help"]
