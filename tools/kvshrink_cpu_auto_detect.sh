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

expand_cpu_list() {
    local part start end cpu
    for part in ${1//,/ }; do
        if [[ "$part" == *-* ]]; then
            IFS=- read -r start end <<<"$part"
            for ((cpu = start; cpu <= end; cpu++)); do
                echo "$cpu"
            done
        else
            echo "$part"
        fi
    done
}

compress_cpu_list() {
    printf '%s\n' "$@" | sort -n | awk '
        NR == 1 { start = previous = $1; next }
        $1 == previous + 1 { previous = $1; next }
        {
            printf "%s%s", separator, start == previous ? start : start "-" previous
            separator = ","
            start = previous = $1
        }
        END {
            if (NR > 0)
                printf "%s%s\n", separator, start == previous ? start : start "-" previous
        }
    '
}

declare -A ranks_per_numa=()
gpu_numas=()
gpu_indices=()
gpu_buses=()
for ((rank = 0; rank < TP_SIZE; rank++)); do
    IFS=, read -r gpu_index pci_bus <<<"${gpu_rows[$rank]}"
    gpu_index=${gpu_index//[[:space:]]/}
    pci_bus=${pci_bus//[[:space:]]/}
    pci_address=$(echo "$pci_bus" | sed -E 's/^[[:xdigit:]]{4}([[:xdigit:]]{4}:)/\1/' | tr '[:upper:]' '[:lower:]')
    if [[ ! -r "/sys/bus/pci/devices/$pci_address/numa_node" ]]; then
        echo "ERROR: cannot read NUMA node for GPU $gpu_index at PCI $pci_bus" >&2
        exit 1
    fi
    numa=$(<"/sys/bus/pci/devices/$pci_address/numa_node")
    if ((numa < 0)); then
        echo "ERROR: GPU $gpu_index at PCI $pci_bus has no NUMA node" >&2
        exit 1
    fi
    gpu_numas+=("$numa")
    gpu_indices+=("$gpu_index")
    gpu_buses+=("$pci_bus")
    ranks_per_numa[$numa]=$((${ranks_per_numa[$numa]:-0} + 1))
done

declare -A next_rank_on_numa=()
rank_cpu_groups=()
for ((rank = 0; rank < TP_SIZE; rank++)); do
    numa=${gpu_numas[$rank]}
    node_path=/sys/devices/system/node/node$numa
    if [[ ! -r "$node_path/cpulist" ]]; then
        echo "ERROR: cannot read CPU list for NUMA node $numa" >&2
        exit 1
    fi

    declare -A seen_cores=()
    core_groups=()
    while read -r cpu; do
        topology=/sys/devices/system/cpu/cpu$cpu/topology
        [[ -r "$topology/thread_siblings_list" ]] || continue
        siblings=$(<"$topology/thread_siblings_list")
        [[ -n "${seen_cores[$siblings]:-}" ]] && continue
        seen_cores[$siblings]=1
        core_groups+=("$cpu")
    done < <(expand_cpu_list "$(<"$node_path/cpulist")")

    ranks=${ranks_per_numa[$numa]}
    if ((${#core_groups[@]} < ranks)); then
        echo "ERROR: NUMA node $numa has ${#core_groups[@]} physical cores for $ranks GPU ranks" >&2
        exit 1
    fi
    local_rank=${next_rank_on_numa[$numa]:-0}
    next_rank_on_numa[$numa]=$((local_rank + 1))
    cores_per_rank=$((${#core_groups[@]} / ranks))
    extra_cores=$((${#core_groups[@]} % ranks))
    core_count=$cores_per_rank
    ((local_rank < extra_cores)) && core_count=$((core_count + 1))
    core_start=$((local_rank * cores_per_rank + (local_rank < extra_cores ? local_rank : extra_cores)))

    cpu_ids=()
    for ((offset = 0; offset < core_count; offset++)); do
        cpu_ids+=("${core_groups[$((core_start + offset))]}")
    done
    cpu_group=$(compress_cpu_list "${cpu_ids[@]}")
    rank_cpu_groups+=("$cpu_group")
    echo "KVShrink CPU rank $rank: GPU=${gpu_indices[$rank]} PCI=${gpu_buses[$rank]} NUMA=$numa CPUs=$cpu_group" >&2
    unset seen_cores
done

result=$(IFS='|'; echo "${rank_cpu_groups[*]}")
echo "$result"