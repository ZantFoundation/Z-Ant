#!/usr/bin/env bash
# Downloads or updates the Ethos-U core driver under third_party/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEST="${REPO_ROOT}/third_party/ethos-u-core-driver"
REPO_URL="${ETHOS_U_REPO:-https://github.com/ARM-software/ethos-u.git}"
REF="${ETHOS_U_REF:-main}"
ARCHIVE="${ETHOS_U_ARCHIVE:-}"

if [ -n "${ARCHIVE}" ]; then
    if [ ! -f "${ARCHIVE}" ]; then
        echo "error: ETHOS_U_ARCHIVE points to a missing file: ${ARCHIVE}" >&2
        exit 1
    fi

    tmpdir=$(mktemp -d)
    cleanup() {
        rm -rf "${tmpdir}"
    }
    trap cleanup EXIT

    echo "Extracting Ethos-U driver from ${ARCHIVE}"
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

    extracted_root="$(find "${tmpdir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -n "${extracted_root}" ] && [ "$(find "${tmpdir}" -mindepth 1 -maxdepth 1 | wc -l)" -eq 1 ]; then
        mv "${extracted_root}" "${DEST}"
    else
        mkdir -p "${DEST}"
        cp -R "${tmpdir}"/. "${DEST}/"
    fi

    echo "Ethos-U driver extracted to ${DEST}"
    exit 0
fi

if [ -d "${DEST}/.git" ]; then
    echo "Updating Ethos-U driver in ${DEST}"
    if ! git -C "${DEST}" fetch --depth 1 origin "${REF}"; then
        echo "error: failed to fetch Ethos-U driver (check your network/proxy settings)" >&2
        exit 1
    fi
    git -C "${DEST}" checkout FETCH_HEAD
else
    mkdir -p "${REPO_ROOT}/third_party"
    echo "Cloning Ethos-U driver (${REF}) into ${DEST}"
    if ! git clone --depth 1 --branch "${REF}" "${REPO_URL}" "${DEST}"; then
        echo "error: failed to clone Ethos-U driver from ${REPO_URL}" >&2
        exit 1
    fi
fi

echo "Ethos-U driver is ready under ${DEST}"
