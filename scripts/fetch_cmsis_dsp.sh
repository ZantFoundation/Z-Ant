#!/usr/bin/env bash
# Downloads or updates the CMSIS-DSP sources under third_party/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEST="${REPO_ROOT}/third_party/CMSIS-DSP"
REPO_URL="${CMSIS_DSP_REPO:-https://github.com/ARM-software/CMSIS-DSP.git}"
REF="${CMSIS_DSP_REF:-develop}"
ARCHIVE="${CMSIS_DSP_ARCHIVE:-}"

if [ -n "${ARCHIVE}" ]; then
    if [ ! -f "${ARCHIVE}" ]; then
        echo "error: CMSIS_DSP_ARCHIVE points to a missing file: ${ARCHIVE}" >&2
        exit 1
    fi
    tmpdir=$(mktemp -d)
    cleanup() { rm -rf "${tmpdir}"; }
    trap cleanup EXIT
    case "${ARCHIVE}" in
        *.zip) unzip -q "${ARCHIVE}" -d "${tmpdir}" ;;
        *.tar.gz|*.tgz) tar -xzf "${ARCHIVE}" -C "${tmpdir}" ;;
        *.tar.xz|*.txz) tar -xJf "${ARCHIVE}" -C "${tmpdir}" ;;
        *.tar.bz2|*.tbz2) tar -xjf "${ARCHIVE}" -C "${tmpdir}" ;;
        *) echo "error: unsupported archive format: ${ARCHIVE}" >&2; exit 1 ;;
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
    echo "CMSIS-DSP extracted to ${DEST}"
    exit 0
fi

if [ -d "${DEST}/.git" ]; then
    echo "Updating CMSIS-DSP in ${DEST}"
    candidates=("${REF}" develop main master)
    fetched=0
    for r in "${candidates[@]}"; do
        if git -C "${DEST}" -c http.https://github.com/.extraheader= fetch --depth 1 origin "$r"; then
            fetched=1
            break
        fi
    done
    if [ "$fetched" -ne 1 ]; then
        echo "error: failed to fetch CMSIS-DSP (tried: ${candidates[*]})" >&2
        exit 1
    fi
    git -C "${DEST}" checkout FETCH_HEAD
else
    mkdir -p "${REPO_ROOT}/third_party"
    echo "Cloning CMSIS-DSP (${REF}) into ${DEST}"
    candidates=("${REF}" develop main master)
    cloned=0
    for r in "${candidates[@]}"; do
        if git -c http.https://github.com/.extraheader= clone --depth 1 --branch "$r" "${REPO_URL}" "${DEST}"; then
            cloned=1
            break
        fi
    done
    if [ "$cloned" -ne 1 ]; then
        echo "error: failed to clone CMSIS-DSP from ${REPO_URL} (tried: ${candidates[*]})" >&2
        exit 1
    fi
fi

echo "CMSIS-DSP is ready under ${DEST}"


