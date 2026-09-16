#!/bin/bash -e
# Start the remote_pool KVStore daemon (one KVStore process per TP rank + scheduler).
# Run on the daemon node with the RDMA env from setvars.sh; vLLM clients connect
# with IAXL_RDMA_ENABLE=1 IAXL_RDMA_DAEMON_IP=<this NIC IP> IAXL_RDMA_DAEMON_PORT=<port>.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../setvars.sh"
if [[ "$IAXL_RDMA_ENABLE" != "1" ]]; then
    echo "IAXL_RDMA_ENABLE=1 is required to run the remote KVStore daemon" >&2
    exit 1
fi
: "${IAXL_RDMA_DAEMON_IP:?Set IAXL_RDMA_DAEMON_IP (RDMA NIC IP) in setvars.sh or the environment}"

export LD_PRELOAD="/usr/local/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4${LD_PRELOAD:+:$LD_PRELOAD}"

cd "$SCRIPT_DIR/.."
python3 -X faulthandler -m iaxl.remote_pool.daemon \
    --ip "$IAXL_RDMA_DAEMON_IP" \
    --port "$IAXL_RDMA_DAEMON_PORT" \
    --tp-size "$IAXL_RDMA_TP_SIZE" \
    2>&1 | tee log.kvshrink-daemon
