/* Place .flash_weights into external QSPI (XIP) */
MEMORY
{
    QSPI (rx) : ORIGIN = 0x90000000, LENGTH = 16M
}

SECTIONS
{
    .flash_weights :
    {
    . = ALIGN(32);
    __flash_weights_start__ = .;
    KEEP(*(.flash_weights))
    KEEP(*(.flash_weights.*))
    KEEP(*(.rodata.flash_weights*))
    . = ALIGN(32);
    __flash_weights_end__ = .;
    } > QSPI
}

PROVIDE(__flash_weights_size__ =
 __flash_weights_end__ - __flash_weights_start__);