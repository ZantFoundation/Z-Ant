#include "conv_kernels.h"

#include <stdbool.h>

static bool g_ethos_used = false;

bool zant_stm32n6_conv_f32_ethos(
    const float *input,
    const size_t *input_shape,
    const float *weights,
    const size_t *weight_shape,
    float *output,
    const size_t *output_shape,
    const float *bias,
    size_t bias_len,
    const size_t *stride,
    const size_t *pads,
    const size_t *dilations,
    size_t group,
    size_t filters_per_group,
    size_t channels_per_group)
{
#if defined(ZANT_HAS_ETHOS_U)
    g_ethos_used = true;
    return zant_stm32n6_conv_f32(
        input,
        input_shape,
        weights,
        weight_shape,
        output,
        output_shape,
        bias,
        bias_len,
        stride,
        pads,
        dilations,
        group,
        filters_per_group,
        channels_per_group);
#else
    (void)input;
    (void)input_shape;
    (void)weights;
    (void)weight_shape;
    (void)output;
    (void)output_shape;
    (void)bias;
    (void)bias_len;
    (void)stride;
    (void)pads;
    (void)dilations;
    (void)group;
    (void)filters_per_group;
    (void)channels_per_group;
    return false;
#endif
}

void zant_stm32n6_reset_ethos_test_state(void)
{
    g_ethos_used = false;
}

bool zant_stm32n6_ethos_was_used(void)
{
    return g_ethos_used;
}

