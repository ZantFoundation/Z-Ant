#ifndef ZANT_STM32N6_COMMON_H
#define ZANT_STM32N6_COMMON_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void zant_stm32n6_mark_cmsis_used(void);
void zant_stm32n6_mark_ethos_used(void);
void zant_stm32n6_reset_test_state(void);
bool zant_stm32n6_cmsis_was_used(void);
bool zant_stm32n6_ethos_was_used(void);

#ifdef __cplusplus
}
#endif

#endif // ZANT_STM32N6_COMMON_H
