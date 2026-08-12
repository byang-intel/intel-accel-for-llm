#!/usr/bin/env bash

set -euo pipefail

TP_SIZE=${1:-${TP_SIZE:-}}

if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TP_SIZE must be a positive integer, got '$TP_SIZE'" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required to detect GPU NUMA nodes" >&2
    exit 1
fi

if ! gpu_output=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits); then
    echo "ERROR: failed to query GPU PCI information with nvidia-smi" >&2
    exit 1
fi

mapfile -t gpu_rows <<<"$gpu_output"
if ((${#gpu_rows[@]} < TP_SIZE)); then
    echo "ERROR: TP_SIZE=$TP_SIZE requires $TP_SIZE GPUs, but only ${#gpu_rows[@]} were found" >&2
    exit 1
fi

mapfile -t dsa_paths < <(
    for path in /sys/bus/dsa/devices/dsa[0-9]*; do
        [[ -d "$path" ]] && echo "$path"
    done | sort -V
)
if ((${#dsa_paths[@]} < TP_SIZE)); then
    echo "ERROR: TP_SIZE=$TP_SIZE requires at least $TP_SIZE DSA devices, but only ${#dsa_paths[@]} were found" >&2
    exit 1
fi

used_dsa=()
rank_wqs=()
for ((rank = 0; rank < TP_SIZE; rank++)); do
    row=${gpu_rows[$rank]}
    IFS=, read -r gpu_index pci_bus <<<"$row"
    gpu_index=${gpu_index//[[:space:]]/}
    pci_bus=${pci_bus//[[:space:]]/}
    pci_address=$(echo "$pci_bus" | sed -E 's/^[[:xdigit:]]{4}([[:xdigit:]]{4}:)/\1/' | tr '[:upper:]' '[:lower:]')
    if [[ ! -r "/sys/bus/pci/devices/$pci_address/numa_node" ]]; then
        echo "ERROR: cannot read NUMA node for GPU $gpu_index at PCI $pci_bus" >&2
        exit 1
    fi
    gpu_numa=$(<"/sys/bus/pci/devices/$pci_address/numa_node")
    if ((gpu_numa < 0)); then
        echo "ERROR: GPU $gpu_index at PCI $pci_bus has no NUMA node" >&2
        exit 1
    fi
    echo "KVShrink GPU rank $rank: index=$gpu_index pci=$pci_bus numa=$gpu_numa" >&2

    selected_dsa=""
    selected_wq=""
    for dsa_path in "${dsa_paths[@]}"; do
        dsa_name=${dsa_path##*/}
        [[ " ${used_dsa[*]} " == *" $dsa_name "* ]] && continue
        dsa_numa=$(<"$dsa_path/numa_node")
        [[ "$dsa_numa" == "$gpu_numa" ]] || continue

        dsa_id=${dsa_name#dsa}
        mapfile -t candidate_wqs < <(
            for wq_path in /sys/bus/dsa/devices/wq"$dsa_id".*; do
                [[ -e "$wq_path" ]] || continue
                [[ "$(<"$wq_path/type")" == "user" ]] || continue
                [[ "$(<"$wq_path/state")" == "enabled" ]] || continue
                basename "$wq_path"
            done | sort -V
        )
        ((${#candidate_wqs[@]} > 0)) || continue
        selected_dsa=$dsa_name
        selected_wq=${candidate_wqs[0]}
        break
    done

    if [[ -z "$selected_dsa" ]]; then
        echo "ERROR: no unused DSA with enabled user WQs is available on NUMA $gpu_numa for GPU $gpu_index" >&2
        exit 1
    fi
    used_dsa+=("$selected_dsa")
    rank_wqs+=("$selected_wq")
    echo "KVShrink GPU rank $rank: DSA=$selected_dsa WQ=$selected_wq" >&2
done

result=$(IFS='|'; echo "${rank_wqs[*]}")
echo "$result"