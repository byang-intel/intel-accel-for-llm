#!/usr/bin/env bash

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: this script must be run as root (use sudo)." >&2
    exit 1
fi

GDR_VERSION="${GDR_VERSION:-2.6-1}"
CUDA_VERSION="${CUDA_VERSION:-13.3}"

echo "==> Detecting distribution"
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "ERROR: this script only supports Ubuntu (detected: ${ID:-unknown} ${VERSION_ID:-})." >&2
    exit 1
fi

case "${VERSION_ID}" in
    22.04) GDR_DISTRO="Ubuntu22_04" ;;
    24.04) GDR_DISTRO="Ubuntu24_04" ;;
    26.04) GDR_DISTRO="Ubuntu26_04" ;;
    *)
        echo "ERROR: unsupported Ubuntu version '${VERSION_ID}'. Supported: 22.04, 24.04, 26.04." >&2
        exit 1
        ;;
esac
GDR_DISTRO_LOWER="$(tr '[:upper:]' '[:lower:]' <<<"${GDR_DISTRO}")"
echo "    Ubuntu ${VERSION_ID} -> ${GDR_DISTRO}"

echo "==> Installing build prerequisites"
apt-get update
apt-get install -y wget dkms build-essential "linux-headers-$(uname -r)"

BASE_URL="https://developer.download.nvidia.com/compute/redist/gdrcopy"
DEB_NAME="gdrdrv-dkms_${GDR_VERSION}_amd64.${GDR_DISTRO}.deb"
DEB_URL="${BASE_URL}/CUDA%20${CUDA_VERSION}/${GDR_DISTRO_LOWER}/x64/${DEB_NAME}"
TMP_DEB="/tmp/${DEB_NAME}"

echo "==> Downloading ${DEB_URL}"
wget -O "${TMP_DEB}" "${DEB_URL}"

echo "==> Installing ${DEB_NAME}"
apt-get install -y "${TMP_DEB}"

echo "==> Loading gdrdrv module"
modprobe gdrdrv

echo "==> Done. Verify with:"
echo "      lsmod | grep gdrdrv"
echo "      ls -l /dev/gdrdrv"
