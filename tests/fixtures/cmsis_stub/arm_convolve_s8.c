#include "arm_nnfunctions.h"

#include <limits.h>
#include <stddef.h>

static inline int32_t requantize(int32_t value, int32_t multiplier, int32_t shift)
{
    int64_t product = (int64_t)value * (int64_t)multiplier;
    const int32_t denom_shift = 31 - shift;
    if (denom_shift > 0) {
        int64_t rounding = (int64_t)1 << (denom_shift - 1);
        if (product < 0) {
            rounding = -rounding;
        }
        product += rounding;
        product >>= denom_shift;
    } else if (denom_shift < 0) {
        product <<= (int32_t)(-denom_shift);
    }
    if (product > INT32_MAX) {
        product = INT32_MAX;
    } else if (product < INT32_MIN) {
        product = INT32_MIN;
    }
    return (int32_t)product;
}

int32_t arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    (void)input_dims;
    (void)filter_dims;
    return 0;
}

int32_t arm_convolve_s8_get_buffer_size_dsp(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
}

int32_t arm_convolve_s8_get_buffer_size_mve(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
    return arm_convolve_s8_get_buffer_size(input_dims, filter_dims);
}

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
    q7_t *output_data)
{
    (void)ctx;
    (void)bias_dims;

    if (conv_params == NULL || quant_params == NULL || input_dims == NULL || filter_dims == NULL ||
        output_dims == NULL || input_data == NULL || filter_data == NULL || output_data == NULL) {
        return ARM_CMSIS_NN_ARGUMENT_ERROR;
    }

    const int32_t out_channels = output_dims->c;
    const int32_t out_height = output_dims->h;
    const int32_t out_width = output_dims->w;
    const int32_t in_channels = input_dims->c;
    const int32_t in_height = input_dims->h;
    const int32_t in_width = input_dims->w;
    const int32_t kernel_height = filter_dims->h;
    const int32_t kernel_width = filter_dims->w;

    const int32_t stride_h = conv_params->stride.h;
    const int32_t stride_w = conv_params->stride.w;
    const int32_t pad_h = conv_params->padding.h;
    const int32_t pad_w = conv_params->padding.w;
    const int32_t dilation_h = conv_params->dilation.h == 0 ? 1 : conv_params->dilation.h;
    const int32_t dilation_w = conv_params->dilation.w == 0 ? 1 : conv_params->dilation.w;

    for (int32_t oh = 0; oh < out_height; ++oh) {
        for (int32_t ow = 0; ow < out_width; ++ow) {
            for (int32_t oc = 0; oc < out_channels; ++oc) {
                int32_t acc = 0;
                if (bias_data != NULL) {
                    acc = bias_data[oc];
                }

                for (int32_t kh = 0; kh < kernel_height; ++kh) {
                    for (int32_t kw = 0; kw < kernel_width; ++kw) {
                        const int32_t in_h = oh * stride_h - pad_h + kh * dilation_h;
                        const int32_t in_w = ow * stride_w - pad_w + kw * dilation_w;
                        if (in_h < 0 || in_h >= in_height || in_w < 0 || in_w >= in_width) {
                            continue;
                        }

                        const size_t input_base = ((size_t)in_h * (size_t)in_width + (size_t)in_w) * (size_t)in_channels;
                        const size_t weight_base =
                            (((size_t)oc * (size_t)kernel_height + (size_t)kh) * (size_t)kernel_width + (size_t)kw) *
                            (size_t)in_channels;
                        for (int32_t ic = 0; ic < in_channels; ++ic) {
                            const int32_t lhs = (int32_t)input_data[input_base + (size_t)ic] + conv_params->input_offset;
                            const int32_t rhs = (int32_t)filter_data[weight_base + (size_t)ic];
                            acc += lhs * rhs;
                        }
                    }
                }

                int32_t value = requantize(acc, quant_params->multiplier[oc], quant_params->shift[oc]);
                value += conv_params->output_offset;
                if (value < conv_params->activation_min) {
                    value = conv_params->activation_min;
                }
                if (value > conv_params->activation_max) {
                    value = conv_params->activation_max;
                }
                if (value < -128) {
                    value = -128;
                }
                if (value > 127) {
                    value = 127;
                }

                const size_t out_index = ((size_t)oh * (size_t)out_width + (size_t)ow) * (size_t)out_channels + (size_t)oc;
                output_data[out_index] = (q7_t)value;
            }
        }
    }

    return ARM_CMSIS_NN_SUCCESS;
}
