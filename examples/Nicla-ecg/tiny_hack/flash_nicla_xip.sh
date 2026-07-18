#!/bin/bash
set -e

# Configuration
if [ -d "$HOME/Library/Arduino15/packages/arduino/tools/arm-none-eabi-gcc/7-2017q4/bin" ]; then
    # macOS
    ARD="$HOME/Library/Arduino15/packages/arduino/tools/arm-none-eabi-gcc/7-2017q4/bin"
elif [ -d "$HOME/.arduino15/packages/arduino/tools/arm-none-eabi-gcc/7-2017q4/bin" ]; then
    # linux
    ARD="$HOME/.arduino15/packages/arduino/tools/arm-none-eabi-gcc/7-2017q4/bin"
else
    echo "Error: Arduino installation not found"
    echo "Check your Arduino installation"
    exit 1
fi
READELF="$ARD/arm-none-eabi-readelf"
OBJCOPY="$ARD/arm-none-eabi-objcopy"
ELF="./build/arduino.mbed_nicla.nicla_vision/tiny_hack.ino.elf"

echo "=== Nicla Vision XIP Flashing Script ==="

# Check if ELF exists
if [ ! -f "$ELF" ]; then
    echo "Error: ELF file not found at $ELF"
    echo "Make sure you've compiled with Arduino CLI first"
    exit 1
fi

# Check if tools exist
if [ ! -f "$OBJCOPY" ]; then
    echo "Error: objcopy not found at $OBJCOPY"
    echo "Check your Arduino installation"
    exit 1
fi

echo "1. Inspecting ELF sections..."
"$READELF" -S "$ELF" | grep -E "\.flash_weights|\.text|Name" || true

echo "2. Creating internal firmware binary (without weights)..."
"$OBJCOPY" -O binary -R .flash_weights "$ELF" nicla_internal.bin

echo "3. Creating weights binary..."
"$OBJCOPY" -O binary -j .flash_weights "$ELF" nicla_weights.bin

echo "4. File sizes:"
ls -lh nicla_*.bin

echo "5. Checking DFU devices..."
dfu-util -l

echo "6. Please put Nicla Vision in DFU mode (double-tap RESET)"
read -p "Press Enter when ready..."

echo "7. Flashing internal firmware..."
dfu-util -a 0 -s 0x08040000:leave -D nicla_internal.bin

echo "8. Put Nicla Vision in DFU mode again (double-tap RESET)"
read -p "Press Enter when ready..."

echo "9. Flashing weights to external flash..."
dfu-util -a 1 -s 0x90000000:leave -D nicla_weights.bin

echo "=== Flashing complete! ==="
echo "Your device should now boot with XIP weights."
