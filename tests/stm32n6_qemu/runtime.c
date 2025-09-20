#include "semihost.h"

#include <stdint.h>

extern int main(void);

extern uint32_t __stack_top__;
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

static void default_handler(void);
void Reset_Handler(void);

__attribute__((section(".isr_vector"), used))
void (*const g_vector_table[])(void) = {
    (void (*)(void))(&__stack_top__),
    Reset_Handler,
    default_handler,
    default_handler,
    default_handler,
    default_handler,
};

static void zero_bss(void)
{
    for (uint32_t *ptr = &_sbss; ptr < &_ebss; ++ptr) {
        *ptr = 0u;
    }
}

static void init_data(void)
{
    const uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }
}

void Reset_Handler(void)
{
    init_data();
    zero_bss();

    int code = main();
    semihost_exit(code);
    for (;;) {
    }
}

static void default_handler(void)
{
    semihost_write0("Unhandled interrupt\n");
    semihost_exit(1);
    for (;;) {
    }
}
