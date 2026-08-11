#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-ubuntu:24.04}"

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required but not found." >&2
    exit 1
fi

if [[ ! -d /etc/accel-config ]]; then
    echo "==> Creating /etc/accel-config on host"
    sudo mkdir -p /etc/accel-config
fi

echo "==> Launching ${IMAGE} container to configure DSA"
docker run --rm --privileged \
    --entrypoint "" \
    -e https_proxy \
    -e http_proxy \
    -e no_proxy \
    -v /sys:/sys \
    -v /etc/accel-config:/etc/accel-config \
    -v "${SCRIPT_DIR}:/scripts:ro" \
    "${IMAGE}" \
    bash -c '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y accel-config
    exec bash /scripts/setup_dsa.sh "$@"
  ' -- "$@"
