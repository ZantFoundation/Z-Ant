#include "semihost.h"

#include <stddef.h>
#include <stdint.h>

extern int32_t predict(float *input, uint32_t *input_shape, uint32_t shape_len,
                       float **result);

static void fill_input(float *buf, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    buf[i] = 0.0f;
  }
}

int main(void) {
  /* Beer model expects input shape [1, 96, 96, 1] */
  static float input[96u * 96u];
  uint32_t shape[4] = {1u, 96u, 96u, 1u};
  float *output = NULL;

  fill_input(input, (uint32_t)(96u * 96u));

  const int32_t status = predict(input, shape, 4u, &output);
  if (status != 0) {
    semihost_write0("beer FAIL\n");
    return (int)status;
  }

  if (output == NULL) {
    semihost_write0("beer FAIL\n");
    return -1;
  }

  semihost_write0("beer PASS\n");
  return 0;
}
