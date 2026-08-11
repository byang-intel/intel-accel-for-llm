export IAXL_BASE_DOCKER_IMAGE=${IAXL_BASE_DOCKER_IMAGE:-"vllm/vllm-openai:v0.23.0"}
export IAXL_DEV_DOCKER_IMAGE=${IAXL_DEV_DOCKER_IMAGE:-"vllm-iaxl-dev"}
export IAXL_BUILDER_DOCKER_IMAGE=${IAXL_BUILDER_DOCKER_IMAGE:-"vllm-iaxl-builder"}

# =============================================================================
# Configurable runtime / build environment variables (all IAXL_* prefixed).
# Each keeps its default unless already set in the environment; edit as needed.
# =============================================================================

# ---- Build ------------------------------------------------------------------
export DEVICE=${DEVICE:-cuda}                 # Build backend: cuda | xpu
export IAXL_CMAKE_ARGS=${IAXL_CMAKE_ARGS:-""} # Extra cmake flags, e.g. "-DENABLE_NVTX=OFF"

# ---- Cache / compression ----------------------------------------------------
export IAXL_KV_COMPRESSION=${IAXL_KV_COMPRESSION:-1}                     # Enable QAT DEFLATE compression (0/1)
export IAXL_KV_LOSSY_TRUNC=${IAXL_KV_LOSSY_TRUNC:-0}                     # Lossy LSB truncation: 'auto', 0 (off), or N bits
export IAXL_KV_DATA_SHUFFLE=${IAXL_KV_DATA_SHUFFLE:-0}                   # Byte-shuffle before compression (0/1)
export IAXL_CACHE_DIR=${IAXL_CACHE_DIR:-_data/kvcache}                   # Base directory for persisted cache files
export IAXL_CACHE_STREAM_SYNC_ON_GET=${IAXL_CACHE_STREAM_SYNC_ON_GET:-0} # CPU-sync the GPU stream on get() (0/1)
export IAXL_CACHE_CACHEGROUP_SIZE=${IAXL_CACHE_CACHEGROUP_SIZE:-100}     # Reserved entries per cache group
export IAXL_CACHE_CACHEGROUP_NUM=${IAXL_CACHE_CACHEGROUP_NUM:-100000}    # Reserved number of cache groups
export IAXL_PREALLOC_LIMIT=${IAXL_PREALLOC_LIMIT:-0}                     # Cap pinned scratch-pool pre-allocation (0 = unlimited)
# export IAXL_DDR_POOL_SIZE_GB=...         # DDR (host) cache pool size in GB (unset = 1/10 of RAM)

# ---- Intel QAT (compression accelerator) ------------------------------------
export IAXL_QAT_DEVICES=${IAXL_QAT_DEVICES:-0}                                   # Comma-separated QAT device indices, e.g. "0,1"
export IAXL_QAT_ZIP_INSTANCES_PER_DEVICE=${IAXL_QAT_ZIP_INSTANCES_PER_DEVICE:-4} # Instances (driving threads) per device
# QAT driving threads = number of devices x instances-per-device (override to force).
export IAXL_QAT_INSTANCE_NUM=${IAXL_QAT_INSTANCE_NUM:-$(($(echo "$IAXL_QAT_DEVICES" | awk -F, '{print NF}') * IAXL_QAT_ZIP_INSTANCES_PER_DEVICE))}
export IAXL_QAT_ZIP_SRC_CAP=${IAXL_QAT_ZIP_SRC_CAP:-262144}    # Source block size in bytes (256 KiB)
export IAXL_QAT_ZIP_DST_CAP=${IAXL_QAT_ZIP_DST_CAP:-262144}    # Compressed-output cap in bytes (256 KiB)
export IAXL_QAT_ZIP_QUEUE_DEPTH=${IAXL_QAT_ZIP_QUEUE_DEPTH:-4} # In-flight requests per instance (<= 4)

# ---- Intel DSA (host<->device copy accelerator, CUDA only) ------------------
export IAXL_DSA_GD_ENABLE=${IAXL_DSA_GD_ENABLE:-0}                     # Use the Intel DSA + GDRCopy transfer path (0/1)
export IAXL_DSA_GD_RESET_ON_DESTROY=${IAXL_DSA_GD_RESET_ON_DESTROY:-0} # Large BAR with stable tensor VAs can keep this off for better performance (0/1)
export IAXL_DSA_WQS=${IAXL_DSA_WQS:-wq0.0}                             # Comma/space separated DSA work-queue names

# ---- Debug / profiling ------------------------------------------------------
export PYTHONOPTIMIZE=0                                 # Keep Python assert statements enabled (python -O removes them)
export IAXL_DEBUG=${IAXL_DEBUG:-0}                      # Python log level (0=INFO, 1=DEBUG)
export IAXL_DEBUG_LOG=${IAXL_DEBUG_LOG:-0}              # Native kv_pool verbose logging (0/1)
export IAXL_PROFILE_MODE=${IAXL_PROFILE_MODE:-disabled} # Profiling: disabled | nvtx | full

# ---- Management REST API ----------------------------------------------------
export IAXL_API_WORKER_BASE_PORT=${IAXL_API_WORKER_BASE_PORT:-18800} # Worker server port base (+ rank)
export IAXL_API_CONTROLLER_PORT=${IAXL_API_CONTROLLER_PORT:-18700}   # Controller server port
export IAXL_API_TIMEOUT=${IAXL_API_TIMEOUT:-60}                      # HTTP request timeout in seconds

HOST_IP=$(ip route get 1 | awk '{print $7}' | tr -d '\n')
export no_proxy=localhost,127.0.0.1,localaddress,.localdomain.com,.local,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12,${HOST_IP}
export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=$http_proxy

# docker
CONTAINER_NAME=iaxl.vllm
ENV_VARS=(
    # common
    no_proxy
    http_proxy
    https_proxy
    HOST_IP
    PYTHONOPTIMIZE
    # vLLM
    VLLM_CPU_OMP_THREADS_BIND
    # Backend
    DEVICE
)
while IFS= read -r var; do
    ENV_VARS+=("$var")
done < <(compgen -A variable IAXL_ | sort)

case "$DEVICE" in
    cuda)
        DOCKER_RUN_ARGS=("--runtime" "nvidia" "--gpus" "all")
        ;;
    xpu)
        DOCKER_RUN_ARGS=("--device" "/dev/dri")
        ;;
    *)
        echo "Unsupported DEVICE: $DEVICE (expected cuda or xpu)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
for var in "${ENV_VARS[@]}"; do
    [[ -n "${!var}" ]] && DOCKER_RUN_ARGS+=("-e" "$var=${!var}")
done
DOCKER_RUN_ARGS+=("-e" "HF_HOME=/_data/hf_home")
DOCKER_RUN_ARGS+=("-v" "$PWD/_data:/_data")
DOCKER_RUN_ARGS+=("-v" "/media/model-space:/media/model-space:ro")
DOCKER_RUN_ARGS+=("-v" "$PWD:$PWD" "-w" "$PWD")
