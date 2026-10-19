#!/usr/bin/env bash
# Push ONLY an already built/reviewed image. No build, login, stages or jobs.
set -euo pipefail
IMAGE_NAME="brats-train"
TAG="${1:?Usage: bash spcs/push.sh immutable-tag --approve-image-push}"
[[ "${2:-}" == --approve-image-push ]] || { printf '%s\n' 'Explicit image-push approval required' >&2; exit 2; }
[[ "$TAG" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ && "$TAG" != latest ]] || exit 2
REGISTRY="spectrumhealth-analytics.registry.snowflakecomputing.com"
FULL_TAG="$REGISTRY/care_ml_research_wksp/brain_tumor_seg/training_images/$IMAGE_NAME:$TAG"
docker image inspect "$IMAGE_NAME:$TAG" >/dev/null
# Requires a separately authorized registry login. Never overwrite a visible tag.
if docker manifest inspect "$FULL_TAG" >/dev/null 2>&1; then
    printf '%s\n' "Refusing existing remote tag: $FULL_TAG" >&2
    exit 2
fi
# A failed lookup is not proof of absence (it could be authorization/network).
[[ "${BRATS_REMOTE_TAG_ABSENCE_VERIFIED:-}" == "$FULL_TAG" ]] || {
    printf '%s\n' 'Verify tag absence with the authorized registry, then set BRATS_REMOTE_TAG_ABSENCE_VERIFIED to its exact path' >&2
    exit 2
}
docker tag "$IMAGE_NAME:$TAG" "$FULL_TAG"
docker push "$FULL_TAG"
