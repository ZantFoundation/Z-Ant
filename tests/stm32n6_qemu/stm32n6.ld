MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 0x00400000
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 0x00080000
}

ENTRY(Reset_Handler)

SECTIONS
{
    .isr_vector :
    {
        KEEP(*(.isr_vector))
    } > FLASH

    .text :
    {
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
        __etext = .;
    } > FLASH

    .ARM.exidx :
    {
        *(.ARM.exidx*)
        *(.ARM.extab*)
    } > FLASH

    .data : AT(__etext)
    {
        __sdata = .;
        *(.data*)
        . = ALIGN(4);
        __edata = .;
    } > RAM

    .bss (NOLOAD) :
    {
        __sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        __ebss = .;
    } > RAM

    .heap (NOLOAD) :
    {
        __heap_start__ = .;
        . += 0x1000;
        __heap_end__ = .;
    } > RAM

    . = ALIGN(8);
    __end__ = .;
}

__sidata = __etext;
PROVIDE(__stack_top__ = ORIGIN(RAM) + LENGTH(RAM));
PROVIDE(_estack = __stack_top__);
PROVIDE(_sidata = __sidata);
PROVIDE(_sdata = __sdata);
PROVIDE(_edata = __edata);
PROVIDE(_sbss = __sbss);
PROVIDE(_ebss = __ebss);
