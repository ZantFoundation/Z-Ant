#ifndef ZANT_STM32N6_SEMIHOST_H
#define ZANT_STM32N6_SEMIHOST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void semihost_write0(const char *msg);
void semihost_exit(int status);

#ifdef __cplusplus
}
#endif

#endif // ZANT_STM32N6_SEMIHOST_H
