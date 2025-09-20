#include "conv_kernels.h"

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
// Only include standard library headers when not in freestanding mode
#ifndef __STDC_HOSTED__
// Freestanding environment - provide minimal stubs
static inline void *malloc(size_t size) {
  (void)size;
  return NULL;
}
static inline void free(void *ptr) { (void)ptr; }
static inline void *memcpy(void *dest, const void *src, size_t n) {
  char *d = (char *)dest;
  const char *s = (const char *)src;
  for (size_t i = 0; i < n; i++)
    d[i] = s[i];
  return dest;
}
static inline void *memset(void *s, int c, size_t n) {
  char *p = (char *)s;
  for (size_t i = 0; i < n; i++)
    p[i] = (char)c;
  return s;
}
#endif

// Only include math.h and ARM headers when actually compiling for target, not
// during codegen. When the freestanding toolchain lacks <math.h>, fall back to
// the relevant compiler builtins for the small subset we need.
#ifndef ZANT_CODEGEN_PHASE
#if defined(__has_include)
#if __has_include(<math.h>)
#include <math.h>
#else
#define ZANT_MATH_FALLBACK 1
#endif
#else
#include <math.h>
#endif

#ifdef ZANT_MATH_FALLBACK
static inline int zant_isnan(float value) { return __builtin_isnan(value); }
static inline long zant_lrintf(float value) { return __builtin_lrintf(value); }
#define isnan  zant_isnan
#define lrintf zant_lrintf
#endif

// Forward declarations for memory functions
extern void *malloc(size_t size);
extern void free(void *ptr);

#if defined(ZANT_HAS_CMSIS_DSP)
#include <arm_math.h>
#if defined(__has_include)
#if __has_include(<arm_nnfunctions.h>)
#define ZANT_HAS_CMSIS_NN 1
#include <arm_nnfunctions.h>
#endif
#endif
#endif
#endif

static bool g_cmsis_used = false;

extern void zant_stm32n6_reset_ethos_test_state(void);

typedef void (*zant_dot_fn)(const float *, const float *, size_t, float *);

// Forward declarations
static bool cmsis_helium_conv(const float *input, const size_t *input_shape,
                              const float *weights, const size_t *weight_shape,
                              float *output, const size_t *output_shape,
                              const float *bias, size_t bias_len,
                              const size_t *stride, const size_t *pads,
                              const size_t *dilations, size_t group,
                              size_t filters_per_group,
                              size_t channels_per_group);

static bool conv_impl(const float *input, const size_t *input_shape,
                      const float *weights, const size_t *weight_shape,
                      float *output, const size_t *output_shape,
                      const float *bias, size_t bias_len, const size_t *stride,
                      const size_t *pads, const size_t *dilations, size_t group,
                      size_t filters_per_group, size_t channels_per_group,
                      zant_dot_fn dot_fn);

static void reference_dot(const float *a, const float *b, size_t len,
                          float *out) {
  float acc = 0.0f;
  for (size_t i = 0; i < len; ++i) {
    acc += a[i] * b[i];
  }
  *out = acc;
}

#if defined(ZANT_HAS_CMSIS_DSP) && defined(ZANT_HAS_CMSIS_NN) &&               \
    !defined(ZANT_CODEGEN_PHASE)
static inline q7_t quantize_to_q7(float value) {
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

static inline int32_t quantize_to_q31(float value) {
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

// Move cmsis_helium_conv function implementation inside CMSIS block
static bool cmsis_helium_conv(const float *input, const size_t *input_shape,
                              const float *weights, const size_t *weight_shape,
                              float *output, const size_t *output_shape,
                              const float *bias, size_t bias_len,
                              const size_t *stride, const size_t *pads,
                              const size_t *dilations, size_t group,
                              size_t filters_per_group,
                              size_t channels_per_group) {
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

  if (channels_per_group == 0 || filters_per_group == 0 || group == 0) {
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

  const size_t stride_h = stride[0];
  const size_t stride_w = stride[1];
  const size_t dilation_h = dilations[0];
  const size_t dilation_w = dilations[1];
  const size_t pad_h_begin = pads[0];
  const size_t pad_w_begin = pads[1];

  const size_t kernel_size = kernel_height * kernel_width * channels_per_group;
  float *packed_weights = NULL;
  float *input_patch = NULL;

  if (kernel_size > 0) {
    packed_weights = (float *)malloc(kernel_size * sizeof(float));
    input_patch = (float *)malloc(kernel_size * sizeof(float));
    if (packed_weights == NULL || input_patch == NULL) {
      free(packed_weights);
      free(input_patch);
      return false;
    }
  }

  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t m = 0; m < out_channels; ++m) {
      const size_t current_group = m / filters_per_group;
      const size_t in_channel_start = current_group * channels_per_group;
      const float bias_val = (bias != NULL) ? bias[m] : 0.0f;

      // Pack weights for this output channel once.
      size_t packed_index = 0;
      for (size_t kh = 0; kh < kernel_height; ++kh) {
        for (size_t kw = 0; kw < kernel_width; ++kw) {
          for (size_t c = 0; c < channels_per_group; ++c) {
            const size_t weight_idx =
                ((m * weight_in_channels + c) * kernel_height + kh) *
                    kernel_width +
                kw;
            packed_weights[packed_index++] = weights[weight_idx];
          }
        }
      }

      for (size_t oh = 0; oh < out_height; ++oh) {
        for (size_t ow = 0; ow < out_width; ++ow) {
          const ptrdiff_t in_h_start =
              (ptrdiff_t)(oh * stride_h) - (ptrdiff_t)pad_h_begin;
          const ptrdiff_t in_w_start =
              (ptrdiff_t)(ow * stride_w) - (ptrdiff_t)pad_w_begin;

          float sum = bias_val;
          size_t input_index = 0;

          for (size_t kh = 0; kh < kernel_height; ++kh) {
            for (size_t kw = 0; kw < kernel_width; ++kw) {
              const ptrdiff_t in_h = in_h_start + (ptrdiff_t)(kh * dilation_h);
              const ptrdiff_t in_w = in_w_start + (ptrdiff_t)(kw * dilation_w);

              for (size_t c = 0; c < channels_per_group; ++c) {
                if (in_h >= 0 && in_h < (ptrdiff_t)in_height && in_w >= 0 &&
                    in_w < (ptrdiff_t)in_width) {
                  const size_t channel = in_channel_start + c;
                  const size_t input_idx =
                      ((n * in_channels + channel) * in_height +
                       (size_t)in_h) *
                          in_width +
                      (size_t)in_w;
                  input_patch[input_index] = input[input_idx];
                } else {
                  input_patch[input_index] = 0.0f;
                }
                ++input_index;
              }
            }
          }

          float dot = 0.0f;
          arm_dot_prod_f32(packed_weights, input_patch, (uint32_t)input_index,
                           &dot);
          sum += dot;

          const size_t output_idx =
              ((n * out_channels + m) * out_height + oh) * out_width + ow;
          output[output_idx] = sum;
        }
      }
    }
  }

  free(packed_weights);
  free(input_patch);
  return true;
}

#elif defined(ZANT_CODEGEN_PHASE)
// Stub types and versions for codegen phase
typedef int8_t q7_t;
typedef int32_t q31_t;

// CMSIS-NN stub types for codegen
typedef struct {
  int32_t *multiplier;
  int32_t *shift;
} cmsis_nn_per_channel_quant_params;

typedef struct {
  int32_t n;
  int32_t h;
  int32_t w;
  int32_t c;
} cmsis_nn_dims;

typedef struct {
  int32_t min;
  int32_t max;
} cmsis_nn_activation;

typedef struct {
  int32_t input_offset;
  int32_t output_offset;
  cmsis_nn_activation activation;
  int32_t activation_min;
  int32_t activation_max;
  struct {
    int32_t h;
    int32_t w;
  } stride;
  struct {
    int32_t h;
    int32_t w;
  } padding;
  struct {
    int32_t h;
    int32_t w;
  } dilation;
} cmsis_nn_conv_params;

typedef struct {
  void *buf;
  int32_t size;
} cmsis_nn_context;

typedef enum {
  ARM_CMSIS_NN_SUCCESS = 0,
  ARM_CMSIS_NN_ARG_ERROR = -1
} arm_cmsis_nn_status;

static inline q7_t quantize_to_q7(float value) {
  return (q7_t)(value > 127.0f ? 127 : (value < -128.0f ? -128 : (int)value));
}

static inline int32_t quantize_to_q31(float value) { return (int32_t)value; }

// Stub CMSIS-NN functions for codegen
static inline int32_t
arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims,
                                const cmsis_nn_dims *filter_dims) {
  (void)input_dims;
  (void)filter_dims;
  return 0;
}

static inline arm_cmsis_nn_status
arm_convolve_s8(const cmsis_nn_context *ctx,
                const cmsis_nn_conv_params *conv_params,
                const cmsis_nn_per_channel_quant_params *quant_params,
                const cmsis_nn_dims *input_dims, const q7_t *input_data,
                const cmsis_nn_dims *filter_dims, const q7_t *filter_data,
                const cmsis_nn_dims *bias_dims, const q31_t *bias_data,
                const cmsis_nn_dims *output_dims, q7_t *output_data) {
  (void)ctx;
  (void)conv_params;
  (void)quant_params;
  (void)input_dims;
  (void)input_data;
  (void)filter_dims;
  (void)filter_data;
  (void)bias_dims;
  (void)bias_data;
  (void)output_dims;
  (void)output_data;
  return ARM_CMSIS_NN_SUCCESS;
}

// Additional stub functions for codegen phase
static inline void *malloc(size_t size) {
  (void)size;
  return NULL;
}
static inline void free(void *ptr) { (void)ptr; }
static inline void *memset(void *s, int c, size_t n) {
  (void)s;
  (void)c;
  (void)n;
  return s;
}

#endif

static bool conv_impl(const float *input, const size_t *input_shape,
                      const float *weights, const size_t *weight_shape,
                      float *output, const size_t *output_shape,
                      const float *bias, size_t bias_len, const size_t *stride,
                      const size_t *pads, const size_t *dilations, size_t group,
                      size_t filters_per_group, size_t channels_per_group,
                      zant_dot_fn dot_fn) {
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

          const ptrdiff_t in_h_start =
              (ptrdiff_t)(oh * stride_h) - (ptrdiff_t)pad_h_begin;
          const ptrdiff_t in_w_start =
              (ptrdiff_t)(ow * stride_w) - (ptrdiff_t)pad_w_begin;

          for (size_t kh = 0; kh < kernel_height; ++kh) {
            for (size_t kw = 0; kw < kernel_width; ++kw) {
              const ptrdiff_t in_h = in_h_start + (ptrdiff_t)(kh * dilation_h);
              const ptrdiff_t in_w = in_w_start + (ptrdiff_t)(kw * dilation_w);

              if (in_h >= 0 && in_h < (ptrdiff_t)in_height && in_w >= 0 &&
                  in_w < (ptrdiff_t)in_width) {
                size_t idx = 0;
                for (size_t c = 0; c < channels_per_group; ++c) {
                  const size_t channel = in_channel_start + c;
                  const size_t input_idx =
                      ((n * in_channels + channel) * in_height + (size_t)in_h) *
                          in_width +
                      (size_t)in_w;
                  const size_t weight_idx =
                      ((m * weight_in_channels + c) * kernel_height + kh) *
                          kernel_width +
                      kw;

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

          const size_t output_idx =
              ((n * out_channels + m) * out_height + oh) * out_width + ow;
          output[output_idx] = sum;
        }
      }
    }
  }

  free(input_patch);
  free(weight_patch);

  return true;
}

bool zant_stm32n6_conv_f32(const float *input, const size_t *input_shape,
                           const float *weights, const size_t *weight_shape,
                           float *output, const size_t *output_shape,
                           const float *bias, size_t bias_len,
                           const size_t *stride, const size_t *pads,
                           const size_t *dilations, size_t group,
                           size_t filters_per_group,
                           size_t channels_per_group) {
  return conv_impl(input, input_shape, weights, weight_shape, output,
                   output_shape, bias, bias_len, stride, pads, dilations, group,
                   filters_per_group, channels_per_group, reference_dot);
}

bool zant_stm32n6_conv_f32_helium(const float *input, const size_t *input_shape,
                                  const float *weights,
                                  const size_t *weight_shape, float *output,
                                  const size_t *output_shape, const float *bias,
                                  size_t bias_len, const size_t *stride,
                                  const size_t *pads, const size_t *dilations,
                                  size_t group, size_t filters_per_group,
                                  size_t channels_per_group) {
#if defined(ZANT_HAS_CMSIS_DSP) && defined(ZANT_HAS_CMSIS_NN)
  if (cmsis_helium_conv(input, input_shape, weights, weight_shape, output,
                        output_shape, bias, bias_len, stride, pads, dilations,
                        group, filters_per_group, channels_per_group)) {
    g_cmsis_used = true;
    return true;
  }
#endif

  return conv_impl(input, input_shape, weights, weight_shape, output,
                   output_shape, bias, bias_len, stride, pads, dilations, group,
                   filters_per_group, channels_per_group, reference_dot);
}

void zant_stm32n6_reset_test_state(void) {
  g_cmsis_used = false;
  zant_stm32n6_reset_ethos_test_state();
}

bool zant_stm32n6_cmsis_was_used(void) { return g_cmsis_used; }
