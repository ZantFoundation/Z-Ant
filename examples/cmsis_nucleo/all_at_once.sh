#!/bin/bash

FQBN="STMicroelectronics:stm32:Nucleo_64" 

arduino-cli compile --fqbn "$FQBN" --export-binaries --libraries ~/Arduino/libraries

./flash_nucleo.sh