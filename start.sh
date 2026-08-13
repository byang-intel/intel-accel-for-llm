#!/bin/bash -e
source setvars.sh

DOCKERFILE=docker/Dockerfile.dev
DOCKERFILE_SHA=$(sha256sum "$DOCKERFILE" | awk '{print $1}')
IMAGE_SHA=$(docker image inspect -f '{{ index .Config.Labels "iaxl.dockerfile.sha" }}' \
    "$IAXL_DEV_DOCKER_IMAGE" 2>/dev/null || true)

if [[ "$DOCKERFILE_SHA" != "$IMAGE_SHA" ]]; then
    docker build -f "$DOCKERFILE" -t "$IAXL_DEV_DOCKER_IMAGE" . \
        --label "iaxl.dockerfile.sha=$DOCKERFILE_SHA" \
        --build-arg "BASE=$IAXL_BASE_DOCKER_IMAGE" \
        --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy
else
    echo "Dockerfile unchanged; skipping build of $IAXL_DEV_DOCKER_IMAGE"
fi

docker run \
    "${DOCKER_RUN_ARGS[@]}" \
    --rm \
    --privileged \
    --pid host \
    --net host \
    -it \
    --name "$CONTAINER_NAME" \
    -v /dev:/dev \
    --entrypoint "" \
    "$IAXL_DEV_DOCKER_IMAGE" \
    bash -c '
        for gid in $(id -G); do
            getent group "$gid" >/dev/null 2>&1 || groupadd -g "$gid" "hostgrp$gid" 2>/dev/null || true
        done
        git config --global --add safe.directory "$PWD"
        pip install -e . --verbose --no-build-isolation
        exec bash'
