#!/usr/bin/env bash
# =============================================================================
# Build, tag, and push the BraTS training image to the Snowflake image registry.
#
# Prerequisites:
#   1. Docker Desktop running with WSL 2 integration
#   2. Snowflake image repository exists:
#        CARE_ML_RESEARCH_WKSP.BRAIN_TUMOR_SEG.TRAINING_IMAGES
#   3. Docker logged in to the Snowflake registry:
#        snow spcs image-registry login --connection <conn>
#        -- or --
#        docker login spectrumhealth-analytics.registry.snowflakecomputing.com
#
# Usage:
#   ./spcs/push.sh                  # uses defaults
#   ./spcs/push.sh --tag v0.2      # custom tag
# =============================================================================
set -euo pipefail

# ---- Defaults ----
ACCOUNT="${SNOWFLAKE_ACCOUNT:-SPECTRUMHEALTH-ANALYTICS}"
DB="${SNOWFLAKE_DB:-CARE_ML_RESEARCH_WKSP}"
SCHEMA="${SNOWFLAKE_SCHEMA:-BRAIN_TUMOR_SEG}"
REPO="TRAINING_IMAGES"
IMAGE_NAME="brats-train"
TAG="latest"

# ---- Parse flags ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)  ACCOUNT="$2"; shift 2 ;;
        --db)       DB="$2"; shift 2 ;;
        --schema)   SCHEMA="$2"; shift 2 ;;
        --repo)     REPO="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        *)          echo "Unknown flag: $1"; exit 1 ;;
    esac
done

REGISTRY="${ACCOUNT}.registry.snowflakecomputing.com"
FULL_TAG="${REGISTRY}/${DB}/${SCHEMA}/${REPO}/${IMAGE_NAME}:${TAG}"
# Snowflake registry requires lowercase
FULL_TAG=$(echo "$FULL_TAG" | tr '[:upper:]' '[:lower:]')

echo "============================================================"
echo "BraTS SPCS image push"
echo "============================================================"
echo "  account  : ${ACCOUNT}"
echo "  registry : ${REGISTRY}"
echo "  image    : ${FULL_TAG}"
echo ""

# ---- Build ----
echo "[1/3] Building image..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
docker build -t "${IMAGE_NAME}:${TAG}" "$PROJECT_DIR"

# ---- Tag ----
echo "[2/3] Tagging for Snowflake registry..."
docker tag "${IMAGE_NAME}:${TAG}" "$FULL_TAG"

# ---- Push ----
echo "[3/3] Pushing to ${REGISTRY}..."
docker push "$FULL_TAG"

echo ""
echo "Done. Image available at:"
echo "  ${FULL_TAG}"
echo ""
echo "Next steps:"
echo "  1. Upload data:  PUT 'file:///path/to/cache/*' @${DB}.${SCHEMA}.DATASET_STAGE/ AUTO_COMPRESS=FALSE PARALLEL=10;"
echo "  2. Upload spec:  PUT 'file:///path/to/service_spec.yaml' @${DB}.${SCHEMA}.SPECS_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"
echo "  3. Run training:"
echo "     USE ROLE SFK_CARE_ML_RSRCH_ADM;"
echo "     EXECUTE JOB SERVICE"
echo "       IN COMPUTE POOL SYSTEM_COMPUTE_POOL_GPU"
echo "       NAME = BRATS_TRAIN_V1"
echo "       FROM @${DB}.${SCHEMA}.SPECS_STAGE"
echo "       SPEC = 'service_spec.yaml';"
