#include "stm32n6_common.h"

static bool g_cmsis_used = false;
static bool g_ethos_used = false;

void zant_stm32n6_mark_cmsis_used(void) { g_cmsis_used = true; }

void zant_stm32n6_mark_ethos_used(void) { g_ethos_used = true; }

void zant_stm32n6_reset_test_state(void) {
    g_cmsis_used = false;
    g_ethos_used = false;
}

bool zant_stm32n6_cmsis_was_used(void) { return g_cmsis_used; }

bool zant_stm32n6_ethos_was_used(void) { return g_ethos_used; }
