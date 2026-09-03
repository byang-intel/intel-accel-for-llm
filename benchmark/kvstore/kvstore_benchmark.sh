#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USE_DSA=0
USE_NSYS=0
PY_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --dsa)
            USE_DSA=1
            ;;
        --nsys)
            USE_NSYS=1
            ;;
        *)
            PY_ARGS+=("$arg")
            ;;
    esac
done

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"

if [[ "$USE_DSA" == "1" ]]; then
    export IAXL_DSA_GD_ENABLE=1
fi

if [[ "$USE_NSYS" == "1" ]]; then
    export IAXL_PROFILE_MODE=nvtx
    export VLLM_NVTX_SCOPES_FOR_PROFILING=1

    rm -f /_data/nsys_report*

    numactl --cpunodebind=0 --membind=0 nsys profile -o /_data/nsys_report \
        -t cuda,nvtx,nccl,python-gil,osrt \
        --python-sampling=true \
        --python-backtrace=cuda \
        --trace-fork-before-exec=true \
        python3 "$SCRIPT_DIR/kvstore_benchmark.py" "${PY_ARGS[@]}"
else
    numactl --cpunodebind=0 --membind=0 python3 "$SCRIPT_DIR/kvstore_benchmark.py" "${PY_ARGS[@]}"
fi
