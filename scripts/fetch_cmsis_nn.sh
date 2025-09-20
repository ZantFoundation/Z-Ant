#!/usr/bin/env bash
# Downloads or updates the CMSIS-NN sources under third_party/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEST="${REPO_ROOT}/third_party/CMSIS-NN"
REPO_URL="${CMSIS_NN_REPO:-https://github.com/ARM-software/CMSIS-NN.git}"
REF="${CMSIS_NN_REF:-main}"
ARCHIVE="${CMSIS_NN_ARCHIVE:-}"

if [ -n "${ARCHIVE}" ]; then
    if [ ! -f "${ARCHIVE}" ]; then
        echo "error: CMSIS_NN_ARCHIVE points to a missing file: ${ARCHIVE}" >&2
        exit 1
    fi

    tmpdir=$(mktemp -d)
    cleanup() {
        rm -rf "${tmpdir}"
    }
    trap cleanup EXIT

    echo "Extracting CMSIS-NN from ${ARCHIVE}" 
    case "${ARCHIVE}" in
        *.zip)
            unzip -q "${ARCHIVE}" -d "${tmpdir}"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "${ARCHIVE}" -C "${tmpdir}"
            ;;
        *.tar.xz|*.txz)
            tar -xJf "${ARCHIVE}" -C "${tmpdir}"
            ;;
        *.tar.bz2|*.tbz2)
            tar -xjf "${ARCHIVE}" -C "${tmpdir}"
            ;;
        *)
            echo "error: unsupported archive format: ${ARCHIVE}" >&2
            exit 1
            ;;
    esac

    mkdir -p "${REPO_ROOT}/third_party"
    rm -rf "${DEST}"

    # Attempt to move the single extracted directory if present; otherwise copy
    # the entire archive contents into DEST.
    extracted_root="$(find "${tmpdir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -n "${extracted_root}" ] && [ "$(find "${tmpdir}" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]; then
        mv "${extracted_root}" "${DEST}"
    else
        mkdir -p "${DEST}"
        cp -R "${tmpdir}"/. "${DEST}/"
    fi

    echo "CMSIS-NN extracted to ${DEST}"
    exit 0
fi

if [ -d "${DEST}/.git" ]; then
    echo "Updating CMSIS-NN in ${DEST}"
    if ! git -C "${DEST}" fetch --depth 1 origin "${REF}"; then
        echo "error: failed to fetch CMSIS-NN (check your network/proxy settings)" >&2
        exit 1
    fi
    git -C "${DEST}" checkout FETCH_HEAD
else
    mkdir -p "${REPO_ROOT}/third_party"
    echo "Cloning CMSIS-NN (${REF}) into ${DEST}"
    if ! git clone --depth 1 --branch "${REF}" "${REPO_URL}" "${DEST}"; then
        echo "error: failed to clone CMSIS-NN from ${REPO_URL}" >&2
        exit 1
    fi
fi

echo "CMSIS-NN is ready under ${DEST}"
