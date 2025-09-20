#ifndef ARM_MATH_H
#define ARM_MATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void arm_dot_prod_f32(const float *p_src_a, const float *p_src_b, uint32_t block_size, float *result);

#ifdef __cplusplus
}
#endif

#endif // ARM_MATH_H
