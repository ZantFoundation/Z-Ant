#!/bin/bash
set -e

# --- Configuration for STMicroelectronics STM32 Core ---

# Definizione della board (la tua Nucleo-64)
FQBN="STMicroelectronics:stm32:Nucleo_64"

# Ricerca del percorso del toolchain ARM GCC (installato da Arduino CLI)
# Nota: Il percorso esatto può variare leggermente tra le versioni, ma segue questo schema.
# Il percorso di base per i tool installati si trova in AppData/Local/Arduino15/packages
STM32_BASE_PATH="$HOME/AppData/Local/Arduino15/packages/STMicroelectronics"

# Trova la versione più recente della toolchain GCC per STM32
ARM_GCC_TOOL_PATH=$(find "$STM32_BASE_PATH/tools/xpack-arm-none-eabi-gcc" -maxdepth 1 -type d | sort -r | head -n 1)

if [ -z "$ARM_GCC_TOOL_PATH" ]; then
    echo "Error: STM32 ARM GCC toolchain not found."
    echo "Make sure you have installed the STMicroelectronics board package via arduino-cli core install stm32."
    exit 1
fi

ARD_BIN="$ARM_GCC_TOOL_PATH/bin"
OBJCOPY="$ARD_BIN/arm-none-eabi-objcopy"
ELF_NAME="cmsis_nucleo.ino.elf"

# Determina il percorso del file ELF (compilato nella cartella di build temporanea/esportata)
ELF=$(find . -name "$ELF_NAME" | head -n 1)

if [ -z "$ELF" ]; then
    echo "Error: ELF file ($ELF_NAME) not found. Compile the sketch first with '--export-binaries'."
    exit 1
fi

# Flashing via ST-Link (che gestisce l'upload via SWD)

echo "=== STM32 Nucleo Flashing Script via ST-Link/SWD ==="

# Compilazione (se non fatta)
echo "Found ELF file: $ELF"
echo "---"

# Upload utilizzando il tool integrato di Arduino CLI
echo "1. Starting upload using arduino-cli integrated tool..."

# L'upload carica il binario creato nella fase di 'compile' precedente e gestisce il flashing via SWD.
arduino-cli upload \
  --fqbn "$FQBN" \
  --port COM3 \
  --board-options upload_method=swdMethod

echo "2. Waiting for upload to complete..."

echo "=== Flashing and automatic restart complete! ==="
echo "Controlla la porta seriale alla velocità specificata nel tuo codice."