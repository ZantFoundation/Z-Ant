#include "conv_kernels.h"
#include "semihost.h"

#include <stdbool.h>
#include <stddef.h>

typedef bool (*conv_fn)(const float *, const size_t *, const float *,
                        const size_t *, float *, const size_t *, const float *,
                        size_t, const size_t *, const size_t *, const size_t *,
                        size_t, size_t, size_t);

static const float kInput[] = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f,
};

static const size_t kInputShape[] = {1u, 1u, 3u, 3u};

static const float kWeights[] = {
    1.0f, 0.0f,
    0.0f, 1.0f,
};

static const size_t kWeightShape[] = {1u, 1u, 2u, 2u};

static const size_t kStride[] = {1u, 1u};
static const size_t kPads[] = {0u, 0u, 0u, 0u};
static const size_t kDilations[] = {1u, 1u};

static const size_t kOutputShape[] = {1u, 1u, 2u, 2u};
static const float kExpected[] = {6.0f, 8.0f, 12.0f, 14.0f};
static const float kTolerance = 1e-4f;
static float g_output[4];

struct ConvCase {
  const char *name;
  conv_fn fn;
  bool require_cmsis;
  bool require_ethos;
};

#if defined(ZANT_HAS_ETHOS_U)
#define VARIANT_NAME "ethos"
static const struct ConvCase kCases[] = {
    {"ethos", zant_stm32n6_conv_f32_ethos, false, true},
};
#elif defined(ZANT_HAS_CMSIS_DSP) && defined(ZANT_HAS_CMSIS_NN)
#define VARIANT_NAME "helium"
static const struct ConvCase kCases[] = {
    {"helium", zant_stm32n6_conv_f32_helium, true, false},
};
#else
#define VARIANT_NAME "reference"
static const struct ConvCase kCases[] = {
    {"reference", zant_stm32n6_conv_f32, false, false},
};
#endif

static const char kPassMessage[] = "stm32n6 " VARIANT_NAME " PASS\n";
static const char kFailMessage[] = "stm32n6 " VARIANT_NAME " FAIL\n";
static const char kOutputMismatch[] = "stm32n6 output mismatch\n";
static const char kUsageErrorCmsis[] = "stm32n6 expected CMSIS path\n";
static const char kUsageErrorEthos[] = "stm32n6 expected Ethos path\n";

static float float_abs(float value) {
  return (value < 0.0f) ? -value : value;
}

static bool check_output(const float *actual) {
  for (size_t idx = 0; idx < 4u; ++idx) {
    const float delta = float_abs(actual[idx] - kExpected[idx]);
    if (delta > kTolerance) {
      return false;
    }
  }
  return true;
}

static bool run_conv(conv_fn fn) {
  for (size_t idx = 0; idx < 4u; ++idx) {
    g_output[idx] = 0.0f;
  }

  if (!fn(kInput, kInputShape, kWeights, kWeightShape, g_output, kOutputShape,
          NULL, 0u, kStride, kPads, kDilations, 1u, 1u, 1u)) {
    return false;
  }

  if (!check_output(g_output)) {
    semihost_write0(kOutputMismatch);
    return false;
  }

  return true;
}

int main(void) {
  bool all_passed = true;

  for (size_t idx = 0; idx < (sizeof(kCases) / sizeof(kCases[0])); ++idx) {
    zant_stm32n6_reset_test_state();
    if (!run_conv(kCases[idx].fn)) {
      all_passed = false;
      break;
    }

    if (kCases[idx].require_cmsis && !zant_stm32n6_cmsis_was_used()) {
      semihost_write0(kUsageErrorCmsis);
      all_passed = false;
      break;
    }

    if (kCases[idx].require_ethos && !zant_stm32n6_ethos_was_used()) {
      semihost_write0(kUsageErrorEthos);
      all_passed = false;
      break;
    }
  }

  if (all_passed) {
    semihost_write0(kPassMessage);
    return 0;
  }

  semihost_write0(kFailMessage);
  return 1;
}
