#!/bin/bash

FQBN="arduino:mbed_nicla:nicla_vision"

arduino-cli compile --fqbn "$FQBN" --export-binaries --libraries ~/Arduino/libraries

./flash_nicla.sh