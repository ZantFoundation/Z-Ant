# Flash nucleo64

$ MODEL_NAME="your_model_name"

$ zig build lib-gen -Dmodel="$MODEL_NAME" -Ddo_export 

$ zig build lib -Dmodel="$MODEL_NAME$" -Denable_CMSIS -Dtarget=thumb-freestanding-eabihf -Dcpu=cortex_m4 -Doptimize=ReleaseSmall

- ONLY ONCE: copy the directories containing all files "examples/cmsis_nicla/ZantLib" in "~/Arduino/libraries/"
    $ cp zig-out/$MODEL_NAME/freestanding/libzant.a ~/Arduino/libraries/ZantLib/src/cortex-m4

$ cd examples/cmsis_nucleo 

- only once: 
    $ chmod +x flash_nucleo.sh 
    $ chmod +x all_at_once.sh 

- Connect the Nucleo64 board to your PC via USB ( use " $ arduino-cli board list " to verify that the board is detected )

- flash the board:
    $ ./flash_nucleo.sh

- Note: executing the flash command the serial monitor will be open automatically.