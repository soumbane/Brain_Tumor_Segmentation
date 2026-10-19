#!/usr/bin/env bash
# Local-only build: no authentication, push, uploads, or job submission.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:?Usage: bash spcs/build.sh immutable-tag}"
[[ "$TAG" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]+$ && "$TAG" != latest ]] || { printf '%s\n' 'Require non-latest immutable tag' >&2; exit 2; }
IMAGE="brats-train:${TAG}"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    printf '%s\n' "Refusing to overwrite existing image tag: $IMAGE" >&2
    exit 2
fi
REVISION="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
# Hash actual working-tree contents, not only git HEAD (uncommitted edits matter).
TREE_HASH="$(python "$PROJECT_DIR/spcs/image_check.py" --source-only)"
docker build --platform linux/amd64 --build-arg "SOURCE_REVISION=$REVISION" \
    --build-arg "SOURCE_TREE_SHA256=$TREE_HASH" --build-arg "BUILD_TAG=$TAG" \
    -t "$IMAGE" "$PROJECT_DIR"
docker run --rm --platform linux/amd64 --network none "$IMAGE" python -m pip check
docker run --rm --platform linux/amd64 --network none "$IMAGE" python /app/spcs/image_check.py --model-smoke
docker image inspect "$IMAGE" --format 'Image={{.Id}} Platform={{.Os}}/{{.Architecture}} Source={{index .Config.Labels "brats.source-tree-sha256"}}'
