#!/usr/bin/env bash

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: this script must be run as root (use sudo)." >&2
    exit 1
fi

CONFS=()
for f in /etc/4xxx_dev[0-7].conf; do
    [[ -f "$f" ]] && CONFS+=("$f")
done

if [[ ${#CONFS[@]} -eq 0 ]]; then
    echo "ERROR: no /etc/4xxx_dev[0-7].conf found." >&2
    echo "       Run this on the host, not inside a container, and make sure QAT is installed." >&2
    exit 1
fi

IOMMU=0
if grep -qE "iommu=(on|pt)" /proc/cmdline; then
    IOMMU=1
    echo "IOMMU enabled: adding SVMEnabled/ATEnabled to [GENERAL]"
else
    echo "IOMMU disabled: removing SVMEnabled/ATEnabled from [GENERAL]"
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

for CONF in "${CONFS[@]}"; do
    awk -v iommu="$IOMMU" '
        /^[[:space:]]*(SVMEnabled|ATEnabled)[[:space:]]*=/ { next }
        { print }
        iommu == 1 && /^[[:space:]]*\[GENERAL\][[:space:]]*$/ {
            print "SVMEnabled = 1"
            print "ATEnabled = 1"
        }
    ' "$CONF" >"$TMP"

    if cmp -s "$TMP" "$CONF"; then
        echo "  $CONF: already up to date"
        continue
    fi

    cp -a "$CONF" "$CONF.bak.$(date +%s)"
    cat "$TMP" >"$CONF"
    echo "  $CONF: updated"
done

echo "Done. Run 'adf_ctl restart' (or 'systemctl restart qat') to apply."
