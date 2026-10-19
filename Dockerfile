# Immutable linux/amd64 runtime candidate. Driver/GPU fit remains a smoke gate.
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4

ARG SOURCE_REVISION=unrecorded
ARG SOURCE_TREE_SHA256=unrecorded
ARG BUILD_TAG=unrecorded
LABEL org.opencontainers.image.title="BraTS numeric-tracked finite jobs" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      brats.source-tree-sha256=$SOURCE_TREE_SHA256 \
      brats.build-tag=$BUILD_TAG
WORKDIR /app
# The runtime uses externally managed system Python. Reuse its CUDA packages
# without modifying the base interpreter; inherited pip installs into this venv.
RUN python -m venv --without-pip --system-site-packages /opt/brats-venv
ENV VIRTUAL_ENV="/opt/brats-venv" PATH="/opt/brats-venv/bin:$PATH"
COPY spcs/requirements.txt /app/spcs/requirements.txt
# Exact torch constraint prevents pip replacing the base CUDA stack.
RUN python -c "import sys, torch; assert (3,11) <= sys.version_info[:2] < (3,13), sys.version; assert torch.__version__.split('+')[0] == '2.13.0'; assert torch.version.cuda == '12.6'" \
    && python -m pip install --no-cache-dir -r /app/spcs/requirements.txt \
    && python -m pip check \
    && python -m pip freeze --all > /app/dependencies.lock.txt
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY splits/ /app/splits/
COPY scripts/ /app/scripts/
COPY spcs/ /app/spcs/
ENV PYTHONPATH="/app/src" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    NCCL_DEBUG=WARN WANDB_CONSOLE=off WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true \
    WANDB_SAVE_CODE=false PYTHONUNBUFFERED=1
RUN mkdir -p /mnt/data /mnt/output /work \
    && python /app/spcs/image_check.py --provenance /app/build-provenance.json
CMD ["python", "-m", "brats.train", "--help"]
