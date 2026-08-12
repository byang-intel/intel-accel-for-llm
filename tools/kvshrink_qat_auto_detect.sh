#!/usr/bin/env bash

set -euo pipefail

TP_SIZE=${1:-${TP_SIZE:-}}

if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TP_SIZE must be a positive integer, got '$TP_SIZE'" >&2
    exit 1
fi

qat_devices=()
for path in /sys/bus/pci/devices/*; do
    [[ -r "$path/class" && -L "$path/driver" ]] || continue
    [[ "$(<"$path/class")" == "0x0b4000" ]] || continue
    driver=$(basename "$(readlink -f "$path/driver")")
    [[ "$driver" != *vf* ]] || continue
    [[ "$driver" =~ (qat|4xxx|420xx|c6xx|c3xxx|200xx|dh895xcc) ]] || continue
    qat_devices+=("${path##*/}")
done

if ((${#qat_devices[@]} == 0)); then
    echo "ERROR: no QAT devices bound to a supported PF driver were found" >&2
    exit 1
fi
mapfile -t qat_devices < <(printf '%s\n' "${qat_devices[@]}" | LC_ALL=C sort)

if ((${#qat_devices[@]} < TP_SIZE)); then
    echo "ERROR: TP_SIZE=$TP_SIZE requires at least $TP_SIZE QAT devices, but only ${#qat_devices[@]} were found" >&2
    exit 1
fi

devices_per_rank=$((${#qat_devices[@]} / TP_SIZE))
extra_devices=$((${#qat_devices[@]} % TP_SIZE))
next_device=0
rank_groups=()

for ((rank = 0; rank < TP_SIZE; rank++)); do
    count=$devices_per_rank
    ((rank < extra_devices)) && count=$((count + 1))

    indices=()
    pci_devices=()
    for ((offset = 0; offset < count; offset++)); do
        indices+=("$next_device")
        pci=${qat_devices[$next_device]}
        numa=$(<"/sys/bus/pci/devices/$pci/numa_node")
        pci_devices+=("$pci(numa=$numa)")
        next_device=$((next_device + 1))
    done

    group=$(IFS=,; echo "${indices[*]}")
    rank_groups+=("$group")
    echo "KVShrink QAT rank $rank: indices=$group devices=${pci_devices[*]}" >&2
done

result=$(IFS='|'; echo "${rank_groups[*]}")
echo "$result"