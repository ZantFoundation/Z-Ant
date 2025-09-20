#ifndef CMSIS_STUB_ARM_NNFUNCTIONS_H
#define CMSIS_STUB_ARM_NNFUNCTIONS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int8_t q7_t;

typedef struct {
    int32_t n;
    int32_t h;
    int32_t w;
    int32_t c;
} cmsis_nn_dims;

typedef struct {
    void *buf;
    int32_t size;
} cmsis_nn_context;

typedef struct {
    int32_t *multiplier;
    int32_t *shift;
} cmsis_nn_per_channel_quant_params;

typedef struct {
    int32_t input_offset;
    int32_t output_offset;
    cmsis_nn_dims stride;
    cmsis_nn_dims padding;
    cmsis_nn_dims dilation;
    int32_t activation_min;
    int32_t activation_max;
} cmsis_nn_conv_params;

typedef enum {
    ARM_CMSIS_NN_SUCCESS = 0,
    ARM_CMSIS_NN_ARGUMENT_ERROR = -1,
} arm_cmsis_nn_status;

int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);
int32_t arm_convolve_s8_get_buffer_size_dsp(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);
int32_t arm_convolve_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

arm_cmsis_nn_status arm_convolve_s8(
    const cmsis_nn_context *ctx,
    const cmsis_nn_conv_params *conv_params,
    const cmsis_nn_per_channel_quant_params *quant_params,
    const cmsis_nn_dims *input_dims,
    const q7_t *input_data,
    const cmsis_nn_dims *filter_dims,
    const q7_t *filter_data,
    const cmsis_nn_dims *bias_dims,
    const int32_t *bias_data,
    const cmsis_nn_dims *output_dims,
    q7_t *output_data);

#ifdef __cplusplus
}
#endif

#endif // CMSIS_STUB_ARM_NNFUNCTIONS_H
