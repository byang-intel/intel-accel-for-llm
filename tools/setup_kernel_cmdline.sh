#!/usr/bin/env bash

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: this script must be run as root (use sudo)." >&2
    exit 1
fi

. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]] ||
    [[ ! " 22.04 24.04 26.04 " == *" ${VERSION_ID} "* ]]; then
    echo "ERROR: unsupported OS '${ID:-?} ${VERSION_ID:-?}'. Need Ubuntu 22.04/24.04/26.04." >&2
    exit 1
fi

PARAMS="intel_iommu=on,sm_on iommu=pt"
GRUB=/etc/default/grub
KEY=GRUB_CMDLINE_LINUX

cp -a "${GRUB}" "${GRUB}.bak.$(date +%s)"

cur="$(sed -n "s/^${KEY}=\"\(.*\)\"$/\1/p" "${GRUB}")"

for p in ${PARAMS}; do
    [[ " ${cur} " == *" ${p} "* ]] || cur="${cur:+${cur} }${p}"
done

if grep -q "^${KEY}=" "${GRUB}"; then
    sed -i "s|^${KEY}=.*|${KEY}=\"${cur}\"|" "${GRUB}"
else
    echo "${KEY}=\"${cur}\"" >>"${GRUB}"
fi

echo "==> ${KEY}=\"${cur}\""
update-grub
echo "==> Done. Reboot for the new kernel command line to take effect."
