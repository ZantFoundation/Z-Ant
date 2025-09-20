#include "arm_math.h"

void arm_dot_prod_f32(const float *p_src_a, const float *p_src_b, uint32_t block_size, float *result)
{
    float acc = 0.0f;
    for (uint32_t i = 0; i < block_size; ++i) {
        acc += p_src_a[i] * p_src_b[i];
    }
    *result = acc;
}

