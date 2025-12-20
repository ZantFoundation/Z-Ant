#!/bin/bash

# ==============================================================================
#  STM32 NUCLEO FLASHING SCRIPT (Cross-Platform & Robust)
#  - Compiles the sketch
#  - Auto-detects the Mass Storage Drive (Drag & Drop)
#  - Flashes the binary
#  - Auto-detects the Serial Port
#  - Opens the Serial Monitor
# ==============================================================================

# --- CONFIGURATION ---
BAUDRATE="115200"

# Board Definition
FQBN="STMicroelectronics:stm32:Nucleo_64:pnum=NUCLEO_L476RG"

# Build Output Directory
BUILD_DIR="build"

# --- FUNCTIONS ---

# Function to detect the Mass Storage Drive
# Returns ONLY the path to the drive
detect_drive() {
    # Files that typically exist on a Nucleo drive
    CHECK_FILES=("MBED.HTM" "DETAILS.TXT")
    
    # 1. Check for Windows/Git Bash drives (d through g)
    for drive in {d..g}; do
        path="/$drive"
        if [ -d "$path" ]; then
            for file in "${CHECK_FILES[@]}"; do
                if [ -f "$path/$file" ]; then
                    echo "$path"
                    return 0
                fi
            done
        fi
    done

    # 2. Check for macOS (/Volumes)
    if [ -d "/Volumes" ]; then
        for path in /Volumes/*; do
            for file in "${CHECK_FILES[@]}"; do
                if [ -f "$path/$file" ]; then
                    echo "$path"
                    return 0
                fi
            done
        done
    fi

    # 3. Check for Linux (/media or /run/media)
    for root in /media /run/media; do
        if [ -d "$root" ]; then
            found=$(find "$root" -maxdepth 3 -name "DETAILS.TXT" 2>/dev/null | head -n 1)
            if [ ! -z "$found" ]; then
                dirname "$found"
                return 0
            fi
        fi
    done

    return 1
}

# Function to detect the Serial Port
detect_port() {
    # Ask arduino-cli for the list of boards and grep for "Nucleo"
    # Format: "COM3  serial Serial Port (USB)  Nucleo-64 ..."
    
    BOARD_LINE=$(arduino-cli board list | grep "Nucleo" | head -n 1)
    
    if [ -z "$BOARD_LINE" ]; then
        # Fallback: Try looking for generic STM32 or just grab the first USB serial
        BOARD_LINE=$(arduino-cli board list | grep "Serial Port (USB)" | head -n 1)
    fi

    if [ ! -z "$BOARD_LINE" ]; then
        # Extract the first column (the port)
        echo "$BOARD_LINE" | awk '{print $1}'
        return 0
    fi

    return 1
}

# --- MAIN EXECUTION ---

echo "--------------------------------------------------------"
echo "[1/4] CLEANING AND COMPILING..."
echo "Target FQBN: $FQBN"

# Clean previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Compile
arduino-cli compile \
  --fqbn "$FQBN" \
  --output-dir "$BUILD_DIR" \
  .

# Check compilation result
if [ $? -ne 0 ]; then
    echo "[ERROR] Compilation failed!"
    exit 1
fi

echo "--------------------------------------------------------"
echo "[2/4] AUTO-DETECTING DRIVE..."

# Call the function (now silent) to get just the path
TARGET_DRIVE=$(detect_drive)

if [ -z "$TARGET_DRIVE" ]; then
    echo "[ERROR] Nucleo Drive not found!"
    echo "Please ensure the board is connected."
    exit 1
fi

echo "-> Found Nucleo Drive at: $TARGET_DRIVE"

echo "--------------------------------------------------------"
echo "[3/4] FLASHING BINARY..."

# Copy the .bin file to the detected drive
# Note: Using quotes around variables handles paths with spaces correctly
cp "$BUILD_DIR"/*.bin "$TARGET_DRIVE/"

if [ $? -ne 0 ]; then
    echo "[ERROR] Copy/Flash failed!"
    exit 1
fi

echo "-> Flash successful."
echo "-> The board is restarting. Waiting 5 seconds for USB re-enumeration..."

sleep 5

echo "--------------------------------------------------------"
echo "[4/4] STARTING SERIAL MONITOR..."

DETECTED_PORT=$(detect_port)

if [ -z "$DETECTED_PORT" ]; then
    echo "[WARNING] Could not auto-detect the serial port."
    echo "Please check the connection manually."
    exit 1
fi

echo "-> Connected to port: $DETECTED_PORT"
echo "-> Press CTRL+C to exit."
echo "--------------------------------------------------------"

# Open monitor
# arduino-cli monitor -p "$DETECTED_PORT" --config baudrate="$BAUDRATE"
arduino-cli monitor -p "COM3" --config baudrate="115200"