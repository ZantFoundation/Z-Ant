#!/usr/bin/env bash
# Installs a local Zig toolchain (default 0.15.2) into the repository.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ZIG_VERSION="${ZIG_VERSION:-${1:-0.15.2}}"

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
    Linux)
        if [[ "$ARCH" == "x86_64" ]]; then
            ZIG_PLATFORM="linux-x86_64"
        elif [[ "$ARCH" == "aarch64" ]]; then
            ZIG_PLATFORM="linux-aarch64"
        fi
        ;;
    Darwin)
        if [[ "$ARCH" == "arm64" ]]; then
            ZIG_PLATFORM="macos-aarch64"
        else
            ZIG_PLATFORM="macos-x86_64"
        fi
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

INSTALL_ROOT="${ZIG_INSTALL_ROOT:-${REPO_ROOT}/.zig-toolchain}"
ARCHIVE_NAME="zig-${ZIG_PLATFORM}-${ZIG_VERSION}.tar.xz"
DOWNLOAD_BASE="${ZIG_DOWNLOAD_BASE:-https://ziglang.org/download}"
DOWNLOAD_URL="${ZIG_DOWNLOAD_URL:-${DOWNLOAD_BASE}/${ZIG_VERSION}/${ARCHIVE_NAME}}"
TARGET_DIR="${INSTALL_ROOT}/${ZIG_VERSION}"

mkdir -p "${INSTALL_ROOT}"

if [ -x "${TARGET_DIR}/zig" ]; then
    echo "Zig ${ZIG_VERSION} is already installed at ${TARGET_DIR}"
else
    tmpdir=$(mktemp -d)
    cleanup() {
        rm -rf "${tmpdir}"
    }
    trap cleanup EXIT

    echo "Fetching Zig ${ZIG_VERSION} from ${DOWNLOAD_URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL "${DOWNLOAD_URL}" -o "${tmpdir}/${ARCHIVE_NAME}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${tmpdir}/${ARCHIVE_NAME}" "${DOWNLOAD_URL}"
    else
        echo "error: neither curl nor wget is available" >&2
        exit 1
    fi

    echo "Extracting ${ARCHIVE_NAME}"
    tar -xf "${tmpdir}/${ARCHIVE_NAME}" -C "${tmpdir}"

    extracted_dir="${tmpdir}/zig-${ZIG_PLATFORM}-${ZIG_VERSION}"
    if [ ! -d "${extracted_dir}" ]; then
        echo "error: expected directory ${extracted_dir} after extraction" >&2
        exit 1
    fi

    rm -rf "${TARGET_DIR}"
    mv "${extracted_dir}" "${TARGET_DIR}"
fi

ln -sfn "${TARGET_DIR}" "${INSTALL_ROOT}/current"

cat <<MSG
Zig ${ZIG_VERSION} installed at ${TARGET_DIR}
Add it to your PATH for this shell session with:
  export PATH="${INSTALL_ROOT}/current:\$PATH"
Then run zig commands, for example:
  zig version
  zig build test
MSG
