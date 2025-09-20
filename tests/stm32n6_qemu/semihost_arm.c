#include "semihost.h"

#include <stddef.h>

#if defined(STM32N6_HOST)

#include <stdio.h>
#include <stdlib.h>

void semihost_write0(const char *msg) {
    if (msg == NULL) return;
    fputs(msg, stdout);
    fflush(stdout);
}

void semihost_exit(int status) {
    fflush(stdout);
    exit(status);
}

#else

#define SEMIHOST_SYS_WRITE0 0x04u
#define SEMIHOST_SYS_EXIT   0x18u
#define ADP_STOPPED_APPLICATION_EXIT 0x20026u

extern long semihost_call(unsigned int op, void *param);

void semihost_write0(const char *msg) {
    if (msg == NULL) return;
    semihost_call(SEMIHOST_SYS_WRITE0, (void *)msg);
}

void semihost_exit(int status) {
    struct {
        unsigned int reason;
        unsigned int value;
    } args = {ADP_STOPPED_APPLICATION_EXIT, (unsigned int)status};
    semihost_call(SEMIHOST_SYS_EXIT, &args);
    while (1) {
    }
}

#endif

