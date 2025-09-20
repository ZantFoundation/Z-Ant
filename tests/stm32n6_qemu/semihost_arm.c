#include "semihost.h"

#include <stddef.h>
#include <stdint.h>

#if defined(STM32N6_HOST)

#include <stdio.h>
#include <stdlib.h>

void semihost_write0(const char *msg) {
  if (msg == NULL) {
    return;
  }
  fputs(msg, stdout);
  fflush(stdout);
}

void semihost_exit(int status) {
  fflush(stdout);
  exit(status);
}

#else

#define SEMIHOST_SYS_WRITE0 0x04u
#define SEMIHOST_SYS_EXIT 0x18u
#define SEMIHOST_SYS_EXIT_EXTENDED 0x20u
#define ADP_STOPPED_APPLICATION_EXIT 0x20026u

static long semihost_call(unsigned int op, void *param) {
  long result;
  __asm__ volatile("mov r0, %1\n\t"
                   "mov r1, %2\n\t"
                   "bkpt #0xAB\n\t"
                   "mov %0, r0"
                   : "=r"(result)
                   : "r"(op), "r"(param)
                   : "r0", "r1", "memory");
  return result;
}

void semihost_write0(const char *msg) {
  if (msg == NULL) {
    return;
  }
  semihost_call(SEMIHOST_SYS_WRITE0, (void *)msg);
}

void semihost_exit(int status) {
  struct {
    unsigned int reason;
    unsigned int value;
  } args = {ADP_STOPPED_APPLICATION_EXIT, (unsigned int)status};

  semihost_call(SEMIHOST_SYS_EXIT, &args);

  // Some QEMU builds only honour the extended exit request.
  semihost_call(SEMIHOST_SYS_EXIT_EXTENDED, &args);

  for (;;) {
    __asm__ volatile("wfi");
  }
}

#endif
