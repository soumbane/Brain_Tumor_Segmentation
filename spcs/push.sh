#!/usr/bin/env bash
# =============================================================================
# Build, tag, and push the BraTS training image to the Snowflake image registry.
#
# Prerequisites:
#   1. Docker Desktop running with WSL 2 integration
#   2. Snowflake image repository created:
#        CREATE IMAGE REPOSITORY <DB>.<SCHEMA>.BRATS_REPO;
#   3. Docker logged in to the Snowflake registry:
#        docker login <ORG>-<ACCOUNT>.registry.snowflakecomputing.com \
#          -u <USERNAME>
#
# Usage:
#   ./spcs/push.sh                          # uses defaults
#   ./spcs/push.sh --account MYORG-MYACCT   # override account
#   ./spcs/push.sh --tag v0.2               # custom tag
# =============================================================================
set -euo pipefail

# ---- Defaults (override via flags or env vars) ----
ACCOUNT="${SNOWFLAKE_ACCOUNT:-<ORG>-<ACCOUNT>}"
DB="${SNOWFLAKE_DB:-BRATS_MRI}"
SCHEMA="${SNOWFLAKE_SCHEMA:-CORE}"
REPO="BRATS_REPO"
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

echo "============================================================"
echo "BraTS SPCS image push"
echo "============================================================"
echo "  account  : ${ACCOUNT}"
echo "  registry : ${REGISTRY}"
echo "  image    : ${FULL_TAG}"
echo ""

# ---- Validate ----
if [[ "$ACCOUNT" == *"<"* ]]; then
    echo "ERROR: ACCOUNT still contains placeholder. Set --account or SNOWFLAKE_ACCOUNT."
    exit 1
fi

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
echo "  1. Stage data:   python -m brats.snowflake.stage_data --stage @${DB}.${SCHEMA}.DATA"
echo "  2. Update spcs/service_spec.yaml with your account details"
echo "  3. CREATE SERVICE brats_train IN COMPUTE POOL <pool> FROM SPECIFICATION_FILE='service_spec.yaml';"
