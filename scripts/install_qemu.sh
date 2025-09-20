#!/usr/bin/env bash
# Installs qemu-system-arm (suitable for the STM32N6 regression harness).
set -euo pipefail

# Determine whether qemu-system-arm is already available.
if command -v qemu-system-arm >/dev/null 2>&1; then
    existing=$(command -v qemu-system-arm)
    echo "qemu-system-arm is already available at ${existing}" 
    exit 0
fi

run_with_privilege() {
    if [ "${EUID}" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    elif command -v doas >/dev/null 2>&1; then
        doas "$@"
    else
        echo "error: administrator privileges are required to run '$*'." >&2
        echo "       Re-run scripts/install_qemu.sh as root or install qemu-system-arm manually." >&2
        exit 1
    fi
}

install_with_apt() {
    echo "Detected apt-get (Debian/Ubuntu). Installing qemu-system-arm..."
    if [ "${QEMU_SKIP_APT_UPDATE:-0}" != "1" ]; then
        run_with_privilege apt-get update
    fi
    run_with_privilege apt-get install -y qemu-system-arm
}

install_with_dnf() {
    echo "Detected dnf (Fedora/RHEL). Installing qemu-system-arm..."
    run_with_privilege dnf install -y qemu-system-arm
}

install_with_yum() {
    echo "Detected yum (CentOS/RHEL). Installing qemu-system-arm..."
    run_with_privilege yum install -y qemu-system-arm
}

install_with_pacman() {
    echo "Detected pacman (Arch/Manjaro). Installing qemu-arch-extra..."
    run_with_privilege pacman -Sy --noconfirm qemu-arch-extra
}

install_with_zypper() {
    echo "Detected zypper (openSUSE). Installing qemu-arm..."
    run_with_privilege zypper install -y qemu-arm
}

install_with_brew() {
    echo "Detected Homebrew (macOS). Installing qemu..."
    brew install qemu
}

if command -v apt-get >/dev/null 2>&1; then
    install_with_apt
elif command -v dnf >/dev/null 2>&1; then
    install_with_dnf
elif command -v yum >/dev/null 2>&1; then
    install_with_yum
elif command -v pacman >/dev/null 2>&1; then
    install_with_pacman
elif command -v zypper >/dev/null 2>&1; then
    install_with_zypper
elif command -v brew >/dev/null 2>&1; then
    install_with_brew
else
    cat <<'MSG' >&2
error: unable to determine package manager.
       Install qemu-system-arm manually using your distribution's tools,
       or download a release archive from https://www.qemu.org/download/.
MSG
    exit 1
fi

if command -v qemu-system-arm >/dev/null 2>&1; then
    echo "qemu-system-arm installed successfully: $(command -v qemu-system-arm)"
else
    echo "warning: qemu-system-arm is still not in PATH; installation may have failed." >&2
    exit 1
fi
