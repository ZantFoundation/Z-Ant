#!/usr/bin/env bash
set -euo pipefail

rm -f -- nicla_weights.bin nicla_internal.bin
rm -rf -- build

FQBN=arduino:mbed_nicla:nicla_vision
SKETCH_DIR="$PWD"
# keep this layout so your flash script still finds the ELF where it expects it
BUILD_DIR="$SKETCH_DIR/build/arduino.mbed_nicla.nicla_vision"
LD="$SKETCH_DIR/custom.ld"

arduino-cli compile \
    --fqbn "${FQBN}" \
    --export-binaries \
    --libraries ~/Arduino/libraries \
    --build-property "compiler.c.elf.extra_flags=-Wl,-T$PWD/custom.ld"

exec "$SKETCH_DIR/flash_nicla_xip.sh"

