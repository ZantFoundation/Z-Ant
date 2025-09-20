#include "conv_kernels.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#if defined(ZANT_HAS_CMSIS_DSP)
#include <arm_math.h>
#if defined(__has_include)
#if __has_include(<arm_nnfunctions.h>)
#define ZANT_HAS_CMSIS_NN 1
#include <arm_nnfunctions.h>
#endif
#endif
#endif

static bool g_cmsis_used = false;

extern void zant_stm32n6_reset_ethos_test_state(void);

typedef void (*zant_dot_fn)(const float *, const float *, size_t, float *);

static void reference_dot(const float *a, const float *b, size_t len, float *out)
{
    float acc = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        acc += a[i] * b[i];
    }
    *out = acc;
}

#if defined(ZANT_HAS_CMSIS_DSP) && defined(ZANT_HAS_CMSIS_NN)
static inline q7_t quantize_to_q7(float value)
{
    if (isnan(value)) {
        return 0;
    }
    if (value > 127.0f) {
        return 127;
    }
    if (value < -128.0f) {
        return -128;
    }
    long rounded = lrintf(value);
    if (rounded > 127) {
        return 127;
    }
    if (rounded < -128) {
        return -128;
    }
    return (q7_t)rounded;
}

static inline int32_t quantize_to_q31(float value)
{
    if (isnan(value)) {
        return 0;
    }
    if (value > (float)INT32_MAX) {
        return INT32_MAX;
    }
    if (value < (float)INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)lrintf(value);
}

static bool cmsis_helium_conv(
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
    const size_t batch_size = input_shape[0];
    const size_t in_channels = input_shape[1];
    const size_t in_height = input_shape[2];
    const size_t in_width = input_shape[3];

    const size_t out_channels = weight_shape[0];
    const size_t weight_in_channels = weight_shape[1];
    const size_t kernel_height = weight_shape[2];
    const size_t kernel_width = weight_shape[3];

    const size_t out_height = output_shape[2];
    const size_t out_width = output_shape[3];

    if (group == 0 || filters_per_group == 0 || channels_per_group == 0) {
        return false;
    }
    if (batch_size == 0) {
        return false;
    }
    if (in_channels != channels_per_group * group) {
        return false;
    }
    if (out_channels != filters_per_group * group) {
        return false;
    }
    if (weight_in_channels != channels_per_group) {
        return false;
    }
    if (bias != NULL && bias_len != out_channels) {
        return false;
    }
    if (group != 1) {
        return false;
    }
    if (pads[0] != pads[2] || pads[1] != pads[3]) {
        return false;
    }
    if (dilations[0] != 1 || dilations[1] != 1) {
        return false;
    }

    if (in_channels > (size_t)INT32_MAX || in_height > (size_t)INT32_MAX || in_width > (size_t)INT32_MAX ||
        out_channels > (size_t)INT32_MAX || out_height > (size_t)INT32_MAX || out_width > (size_t)INT32_MAX ||
        kernel_height > (size_t)INT32_MAX || kernel_width > (size_t)INT32_MAX) {
        return false;
    }

    const size_t input_plane = in_channels * in_height * in_width;
    const size_t output_plane = out_channels * out_height * out_width;
    const size_t weight_per_filter = channels_per_group * kernel_height * kernel_width;
    const size_t weight_total = out_channels * weight_per_filter;

    q7_t *weights_q7 = (q7_t *)malloc(weight_total * sizeof(q7_t));
    if (weights_q7 == NULL) {
        return false;
    }

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t kh = 0; kh < kernel_height; ++kh) {
            for (size_t kw = 0; kw < kernel_width; ++kw) {
                for (size_t ic = 0; ic < channels_per_group; ++ic) {
                    const size_t src_idx = (((oc * channels_per_group) + ic) * kernel_height + kh) * kernel_width + kw;
                    const size_t dst_idx = (((oc * kernel_height + kh) * kernel_width) + kw) * channels_per_group + ic;
                    weights_q7[dst_idx] = quantize_to_q7(weights[src_idx]);
                }
            }
        }
    }

    int32_t *bias_q31 = (int32_t *)malloc(out_channels * sizeof(int32_t));
    if (bias_q31 == NULL) {
        free(weights_q7);
        return false;
    }
    for (size_t oc = 0; oc < out_channels; ++oc) {
        const float bias_val = (bias != NULL) ? bias[oc] : 0.0f;
        bias_q31[oc] = quantize_to_q31(bias_val);
    }

    int32_t *multipliers = (int32_t *)malloc(out_channels * sizeof(int32_t));
    int32_t *shifts = (int32_t *)malloc(out_channels * sizeof(int32_t));
    if (multipliers == NULL || shifts == NULL) {
        free(shifts);
        free(multipliers);
        free(bias_q31);
        free(weights_q7);
        return false;
    }
    for (size_t oc = 0; oc < out_channels; ++oc) {
        multipliers[oc] = 1 << 30;
        shifts[oc] = 1;
    }

    q7_t *input_q7 = (q7_t *)malloc(input_plane * sizeof(q7_t));
    q7_t *output_q7 = (q7_t *)malloc(output_plane * sizeof(q7_t));
    if (input_q7 == NULL || output_q7 == NULL) {
        free(output_q7);
        free(input_q7);
        free(shifts);
        free(multipliers);
        free(bias_q31);
        free(weights_q7);
        return false;
    }

    cmsis_nn_conv_params conv_params;
    memset(&conv_params, 0, sizeof(conv_params));
    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.stride.h = (int32_t)stride[0];
    conv_params.stride.w = (int32_t)stride[1];
    conv_params.padding.h = (int32_t)pads[0];
    conv_params.padding.w = (int32_t)pads[1];
    conv_params.dilation.h = (int32_t)dilations[0];
    conv_params.dilation.w = (int32_t)dilations[1];
    conv_params.activation_min = -128;
    conv_params.activation_max = 127;

    cmsis_nn_per_channel_quant_params quant_params = {
        .multiplier = multipliers,
        .shift = shifts,
    };

    cmsis_nn_dims input_dims = {
        .n = 1,
        .h = (int32_t)in_height,
        .w = (int32_t)in_width,
        .c = (int32_t)in_channels,
    };
    cmsis_nn_dims filter_dims = {
        .n = (int32_t)out_channels,
        .h = (int32_t)kernel_height,
        .w = (int32_t)kernel_width,
        .c = (int32_t)channels_per_group,
    };
    cmsis_nn_dims bias_dims = {
        .n = 1,
        .h = 1,
        .w = 1,
        .c = (int32_t)out_channels,
    };
    cmsis_nn_dims output_dims = {
        .n = 1,
        .h = (int32_t)out_height,
        .w = (int32_t)out_width,
        .c = (int32_t)out_channels,
    };

    cmsis_nn_context ctx;
    ctx.buf = NULL;
    ctx.size = 0;

    int32_t buffer_size = 0;
#if defined(ARM_MATH_MVEI) || defined(ARM_MATH_MVEF)
    buffer_size = arm_convolve_s8_get_buffer_size_mve(&input_dims, &filter_dims);
#elif defined(ARM_MATH_DSP)
    buffer_size = arm_convolve_s8_get_buffer_size_dsp(&input_dims, &filter_dims);
#else
    buffer_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
#endif
    if (buffer_size < 0) {
        free(output_q7);
        free(input_q7);
        free(shifts);
        free(multipliers);
        free(bias_q31);
        free(weights_q7);
        return false;
    }
    if (buffer_size > 0) {
        ctx.buf = malloc((size_t)buffer_size);
        if (ctx.buf == NULL) {
            free(output_q7);
            free(input_q7);
            free(shifts);
            free(multipliers);
            free(bias_q31);
            free(weights_q7);
            return false;
        }
        ctx.size = buffer_size;
    }

    bool ok = true;
    for (size_t n = 0; n < batch_size; ++n) {
        const float *input_base = input + n * input_plane;
        float *output_base = output + n * output_plane;

        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t ih = 0; ih < in_height; ++ih) {
                for (size_t iw = 0; iw < in_width; ++iw) {
                    const size_t src_idx = ((ic * in_height) + ih) * in_width + iw;
                    const size_t dst_idx = ((ih * in_width) + iw) * in_channels + ic;
                    input_q7[dst_idx] = quantize_to_q7(input_base[src_idx]);
                }
            }
        }

        const arm_cmsis_nn_status status = arm_convolve_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            input_q7,
            &filter_dims,
            weights_q7,
            &bias_dims,
            bias_q31,
            &output_dims,
            output_q7);

        if (status != ARM_CMSIS_NN_SUCCESS) {
            ok = false;
            break;
        }

        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    const size_t src_idx = ((oh * out_width) + ow) * out_channels + oc;
                    const size_t dst_idx = ((oc * out_height) + oh) * out_width + ow;
                    output_base[dst_idx] = (float)output_q7[src_idx];
                }
            }
        }
    }

    free(ctx.buf);
    free(output_q7);
    free(input_q7);
    free(shifts);
    free(multipliers);
    free(bias_q31);
    free(weights_q7);
    return ok;
}
#endif

static bool conv_impl(
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
    size_t channels_per_group,
    zant_dot_fn dot_fn)
{
    if (group == 0 || filters_per_group == 0 || channels_per_group == 0) {
        return false;
    }

    const size_t batch_size = input_shape[0];
    const size_t in_channels = input_shape[1];
    const size_t in_height = input_shape[2];
    const size_t in_width = input_shape[3];

    const size_t out_channels = weight_shape[0];
    const size_t weight_in_channels = weight_shape[1];
    const size_t kernel_height = weight_shape[2];
    const size_t kernel_width = weight_shape[3];

    const size_t out_height = output_shape[2];
    const size_t out_width = output_shape[3];

    if (in_channels != channels_per_group * group) {
        return false;
    }
    if (out_channels != filters_per_group * group) {
        return false;
    }
    if (weight_in_channels != channels_per_group) {
        return false;
    }
    if (bias != NULL && bias_len != out_channels) {
        return false;
    }

    const size_t stride_h = stride[0];
    const size_t stride_w = stride[1];
    const size_t dilation_h = dilations[0];
    const size_t dilation_w = dilations[1];

    const size_t pad_h_begin = pads[0];
    const size_t pad_w_begin = pads[1];

    float *input_patch = NULL;
    float *weight_patch = NULL;
    if (channels_per_group > 0) {
        const size_t bytes = channels_per_group * sizeof(float);
        input_patch = (float *)malloc(bytes);
        weight_patch = (float *)malloc(bytes);
        if (input_patch == NULL || weight_patch == NULL) {
            free(input_patch);
            free(weight_patch);
            return false;
        }
    }

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t m = 0; m < out_channels; ++m) {
            const size_t current_group = m / filters_per_group;
            const size_t in_channel_start = current_group * channels_per_group;

            const float bias_val = (bias != NULL) ? bias[m] : 0.0f;

            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    float sum = bias_val;

                    const ptrdiff_t in_h_start = (ptrdiff_t)(oh * stride_h) - (ptrdiff_t)pad_h_begin;
                    const ptrdiff_t in_w_start = (ptrdiff_t)(ow * stride_w) - (ptrdiff_t)pad_w_begin;

                    for (size_t kh = 0; kh < kernel_height; ++kh) {
                        for (size_t kw = 0; kw < kernel_width; ++kw) {
                            const ptrdiff_t in_h = in_h_start + (ptrdiff_t)(kh * dilation_h);
                            const ptrdiff_t in_w = in_w_start + (ptrdiff_t)(kw * dilation_w);

                            if (in_h >= 0 && in_h < (ptrdiff_t)in_height &&
                                in_w >= 0 && in_w < (ptrdiff_t)in_width) {
                                size_t idx = 0;
                                for (size_t c = 0; c < channels_per_group; ++c) {
                                    const size_t channel = in_channel_start + c;
                                    const size_t input_idx = ((n * in_channels + channel) * in_height + (size_t)in_h) * in_width + (size_t)in_w;
                                    const size_t weight_idx = ((m * weight_in_channels + c) * kernel_height + kh) * kernel_width + kw;

                                    input_patch[idx] = input[input_idx];
                                    weight_patch[idx] = weights[weight_idx];
                                    idx += 1;
                                }

                                if (idx > 0) {
                                    float dot = 0.0f;
                                    dot_fn(input_patch, weight_patch, idx, &dot);
                                    sum += dot;
                                }
                            }
                        }
                    }

                    const size_t output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }

    free(input_patch);
    free(weight_patch);

    return true;
}

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
    size_t channels_per_group)
{
    return conv_impl(
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
        channels_per_group,
        reference_dot);
}

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
    size_t channels_per_group)
{
#if defined(ZANT_HAS_CMSIS_DSP) && defined(ZANT_HAS_CMSIS_NN)
    if (cmsis_helium_conv(
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
            channels_per_group)) {
        g_cmsis_used = true;
        return true;
    }
#endif

    return conv_impl(
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
        channels_per_group,
        reference_dot);
}

void zant_stm32n6_reset_test_state(void)
{
    g_cmsis_used = false;
    zant_stm32n6_reset_ethos_test_state();
}

bool zant_stm32n6_cmsis_was_used(void)
{
    return g_cmsis_used;
}

