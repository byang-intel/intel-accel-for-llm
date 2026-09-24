#!/bin/bash -e
# Usage: kvstore_benchmark.sh [--ranks N] [--dsa] [--nsys] [kvstore_benchmark.py options]

export IAXL_QAT_ZIP_ENABLE=1
export IAXL_IAA_ZIP_ENABLE=1
export IAXL_CPU_ZIP_ENABLE=0
export IAXL_DSA_GD_ENABLE=1
export IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=0

RANKS=1
USE_DSA=0
USE_NSYS=0
PY_ARGS=()

while (($#)); do
    case "$1" in
        --ranks) RANKS=$2; shift 2 ;;
        --ranks=*) RANKS=${1#--ranks=}; shift ;;
        --dsa) USE_DSA=1; shift ;;
        --nsys) USE_NSYS=1; shift ;;
        *) PY_ARGS+=("$1"); shift ;;
    esac
done
if ((RANKS > 1)); then
    export TP_SIZE=$RANKS
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../setvars.sh"

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"

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
elif ((RANKS > 1)); then
    # Each rank process binds its own CPUs / accelerators via iaxl.utils.affinity.
    python3 "$SCRIPT_DIR/kvstore_benchmark_multi_ranks.py" "${PY_ARGS[@]}"
else
    numactl --cpunodebind=0 --membind=0 python3 "$SCRIPT_DIR/kvstore_benchmark.py" "${PY_ARGS[@]}"
fi
