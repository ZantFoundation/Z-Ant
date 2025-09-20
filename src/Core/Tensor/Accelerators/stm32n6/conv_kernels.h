#ifndef ZANT_STM32N6_CONV_KERNELS_H
#define ZANT_STM32N6_CONV_KERNELS_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

bool zant_stm32n6_conv_f32(
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
    size_t channels_per_group);

bool zant_stm32n6_conv_f32_helium(
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
    size_t channels_per_group);

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
    size_t channels_per_group);

void zant_stm32n6_reset_test_state(void);
bool zant_stm32n6_cmsis_was_used(void);
bool zant_stm32n6_ethos_was_used(void);

#ifdef __cplusplus
}
#endif

#endif // ZANT_STM32N6_CONV_KERNELS_H
