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
extern void support_emit_heap_report(void);

static void enable_fpu_and_mve(void);

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
    semihost_write0("[runtime] reset enter\n");
    enable_fpu_and_mve();
    init_data();
    zero_bss();
    semihost_write0("[runtime] reset init done\n");

    int code = main();
    support_emit_heap_report();
    semihost_exit(code);
    for (;;) {
    }
}

static void default_handler(void)
{
    support_emit_heap_report();
    semihost_write0("Unhandled interrupt\n");
    semihost_exit(1);
    for (;;) {
    }
}

static void enable_fpu_and_mve(void)
{
    volatile uint32_t *const scb_cpacr = (uint32_t *)0xE000ED88u;
    const uint32_t cp10_cp11_full_access = 0xFu << 20;
    const uint32_t mve_enable = 1u << 0;
    *scb_cpacr |= cp10_cp11_full_access | mve_enable;
    __asm__ volatile("dsb 0xf" ::: "memory");
    __asm__ volatile("isb 0xf" ::: "memory");
}
