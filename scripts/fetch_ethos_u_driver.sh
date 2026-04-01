#!/usr/bin/env bash
# Downloads or updates the Ethos-U core driver under third_party/.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEST="${REPO_ROOT}/third_party/ethos-u-core-driver"
REPO_URL="${ETHOS_U_REPO:-https://github.com/meta-pytorch/ethos-u-core-driver-mirror.git}"
REF="${ETHOS_U_REF:-main}"
ARCHIVE="${ETHOS_U_ARCHIVE:-}"
ML_REPO_URL="${ETHOS_U_ML_REPO:-https://git.mlplatform.org/ml/ethos-u/ethos-u-core-driver.git}"

fallback_download_extract() {
    echo "Falling back to ZIP download from codeload.github.com"
    tmpzip=$(mktemp /tmp/ethos-u-XXXXXX.zip)
    # Derive owner/repo slug from REPO_URL
    slug="${REPO_URL}"
    slug="${slug#git@github.com:}"
    slug="${slug#https://github.com/}"
    slug="${slug#http://github.com/}"
    slug="${slug%.git}"

    try_download() {
        local url="$1"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "${url}" -o "${tmpzip}" && return 0
        elif command -v wget >/dev/null 2>&1; then
            wget -qO "${tmpzip}" "${url}" && return 0
        fi
        return 1
    }

    # Try requested REF as head/tag, then main/master heads
    candidates=()
    if [ -n "${REF}" ]; then
        candidates+=("https://codeload.github.com/${slug}/zip/refs/heads/${REF}")
        candidates+=("https://codeload.github.com/${slug}/zip/refs/tags/${REF}")
    fi
    candidates+=("https://codeload.github.com/${slug}/zip/refs/heads/main")
    candidates+=("https://codeload.github.com/${slug}/zip/refs/heads/master")

    downloaded=0
    for url in "${candidates[@]}"; do
        if try_download "${url}"; then
            downloaded=1
            break
        fi
    done
    if [ "${downloaded}" -ne 1 ]; then
        echo "error: ZIP fallback failed for repo ${slug}; tried: ${candidates[*]}" >&2
        return 1
    fi

    tmpdir=$(mktemp -d)
    cleanup() {
        rm -rf "${tmpdir}" "${tmpzip}"
    }
    trap cleanup EXIT

    unzip -q "${tmpzip}" -d "${tmpdir}"
    mkdir -p "${REPO_ROOT}/third_party"
    rm -rf "${DEST}"
    extracted_root="$(find "${tmpdir}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    if [ -n "${extracted_root}" ]; then
        mv "${extracted_root}" "${DEST}"
    else
        echo "error: failed to locate extracted root directory" >&2
        return 1
    fi
    echo "Ethos-U driver extracted to ${DEST}"
}

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
    if ! git -C "${DEST}" -c http.https://github.com/.extraheader= fetch --depth 1 origin "${REF}"; then
        echo "warn: git fetch from origin failed; trying mlplatform mirror" >&2
        # Switch remote to mlplatform mirror and retry
        if git -C "${DEST}" remote set-url origin "${ML_REPO_URL}" && git -C "${DEST}" fetch --depth 1 origin "${REF}"; then
            :
        else
            echo "warn: mlplatform fetch failed; attempting ZIP fallback" >&2
            fallback_download_extract || exit 1
            exit 0
        fi
    fi
    git -C "${DEST}" checkout FETCH_HEAD
else
    mkdir -p "${REPO_ROOT}/third_party"
    echo "Cloning Ethos-U driver (${REF}) into ${DEST}"
    if ! git -c http.https://github.com/.extraheader= clone --depth 1 --branch "${REF}" "${REPO_URL}" "${DEST}"; then
        echo "warn: git clone failed from ${REPO_URL}; trying mlplatform mirror" >&2
        if ! git clone --depth 1 --branch "${REF}" "${ML_REPO_URL}" "${DEST}"; then
            echo "warn: mlplatform clone failed; attempting ZIP fallback" >&2
            fallback_download_extract || exit 1
            exit 0
        fi
    fi
fi

echo "Ethos-U driver is ready under ${DEST}"
