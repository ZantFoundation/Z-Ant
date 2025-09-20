#include "semihost.h"
#include "conv_kernels.h"

#include <stdbool.h>
#include <stddef.h>

typedef bool (*conv_fn)(
    const float *,
    const size_t *,
    const float *,
    const size_t *,
    float *,
    const size_t *,
    const float *,
    size_t,
    const size_t *,
    const size_t *,
    const size_t *,
    size_t,
    size_t,
    size_t);

static const float g_input[] = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f,
};

static const size_t g_input_shape[] = { 1u, 1u, 3u, 3u };

static const float g_weights[] = {
    1.0f, 0.0f,
    0.0f, 1.0f,
};

static const size_t g_weight_shape[] = { 1u, 1u, 2u, 2u };

static float g_output[4];
static const size_t g_output_shape[] = { 1u, 1u, 2u, 2u };
static const float g_expected[] = { 6.0f, 8.0f, 12.0f, 14.0f };

static const size_t g_stride[] = { 1u, 1u };
static const size_t g_pads[] = { 0u, 0u, 0u, 0u };
static const size_t g_dilations[] = { 1u, 1u };

static float float_abs(float value)
{
    return (value < 0.0f) ? -value : value;
}

static bool check_output(void)
{
    for (size_t idx = 0; idx < 4u; ++idx) {
        const float delta = float_abs(g_output[idx] - g_expected[idx]);
        if (delta > 1e-4f) {
            return false;
        }
    }
    return true;
}

static bool run_conv(conv_fn fn)
{
    for (size_t idx = 0; idx < 4u; ++idx) {
        g_output[idx] = 0.0f;
    }

    return fn(
        g_input,
        g_input_shape,
        g_weights,
        g_weight_shape,
        g_output,
        g_output_shape,
        NULL,
        0u,
        g_stride,
        g_pads,
        g_dilations,
        1u,
        1u,
        1u) && check_output();
}

static void log_failure(const char *message)
{
    semihost_write0(message);
}

int main(void)
{
    bool ok = true;

    zant_stm32n6_reset_test_state();
    if (!run_conv(zant_stm32n6_conv_f32)) {
        log_failure("reference convolution failed\n");
        ok = false;
    }
    if (zant_stm32n6_cmsis_was_used()) {
        log_failure("reference path touched CMSIS flag\n");
        ok = false;
    }
    if (zant_stm32n6_ethos_was_used()) {
        log_failure("reference path touched Ethos flag\n");
        ok = false;
    }

#if defined(ZANT_HAS_CMSIS_DSP)
    zant_stm32n6_reset_test_state();
    if (!run_conv(zant_stm32n6_conv_f32_helium)) {
        log_failure("helium path failed\n");
        ok = false;
    }
    if (!zant_stm32n6_cmsis_was_used()) {
        log_failure("helium path failed to raise CMSIS flag\n");
        ok = false;
    }
    if (zant_stm32n6_ethos_was_used()) {
        log_failure("helium path touched Ethos flag\n");
        ok = false;
    }
#endif

#if defined(ZANT_HAS_ETHOS_U)
    zant_stm32n6_reset_test_state();
    if (!run_conv(zant_stm32n6_conv_f32_ethos)) {
        log_failure("ethos path failed\n");
        ok = false;
    }
    if (!zant_stm32n6_ethos_was_used()) {
        log_failure("ethos path failed to raise Ethos flag\n");
        ok = false;
    }
    if (zant_stm32n6_cmsis_was_used()) {
        log_failure("ethos path touched CMSIS flag\n");
        ok = false;
    }
#endif

    if (ok) {
        semihost_write0("STM32N6 QEMU harness PASS\n");
        return 0;
    }

    semihost_write0("STM32N6 QEMU harness FAIL\n");
    return 1;
}
