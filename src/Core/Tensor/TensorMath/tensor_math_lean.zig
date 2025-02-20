//! tensor_math_lean only contains forwarding methods and utility methods.
//! All the function return void.

// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------
//

// ---------- importing lean Element-Wise math ----------
const lean_elementWise_math_lib = @import("lib_elementWise_math.zig");
//pub const add_bias = lean_elementWise_math_lib.add_bias;
pub const sum_tensors = lean_elementWise_math_lib.lean_sum_tensors;
//pub const sub_tensors = lean_elementWise_math_lib.sub_tensors;
pub const mul = lean_elementWise_math_lib.mul_lean;
pub const div = lean_elementWise_math_lib.div_lean;
pub const sum_tensor_list = lean_elementWise_math_lib.lean_sum_tensor_list;

// ---------- importing lean Convolution methods ----------
const lean_op_convolution = @import("op_convolution.zig");
pub const im2col = lean_op_convolution.lean_im2col;
pub const OnnxConvLean = lean_op_convolution.OnnxConvLean;

// ---------- importing lean structural methods ----------
const shape_math_lib = @import("lib_shape_math.zig");
pub const reshape = shape_math_lib.reshape_lean;
pub const get_resize_output_shape = shape_math_lib.get_resize_output_shape;
pub const get_concatenate_output_shape = shape_math_lib.get_concatenate_output_shape;
pub const get_split_output_shapes = shape_math_lib.get_split_output_shapes;

// ---------- importing lean matrix algebra methods ----------
const op_mat_mul = @import("op_mat_mul.zig");
pub const lean_mat_mul = op_mat_mul.lean_mat_mul;

// ---------- importing lean gemm method ----------
const op_gemm = @import("op_gemm.zig");
pub const gemm_lean = op_gemm.lean_gemm;

// ---------- importing lean slice methods ----------
const op_slice = @import("op_slice.zig");
pub const lean_slice = op_slice.lean_slice;

// ---------- importing lean activation function methods ----------
const activation_math_lib = @import("lib_activation_function_math.zig");
//ReLU
pub const ReLU = activation_math_lib.lean_ReLU;
//Leaky ReLU
pub const leakyReLU = activation_math_lib.lean_leakyReLU;
//Sigmoid
pub const sigmoid = activation_math_lib.lean_sigmoid;
//Softmax
pub const softmax = activation_math_lib.lean_softmax;

// ---------- importing lean convolution methods ----------
const op_convolution = @import("op_convolution.zig");

pub const convolve_tensor_with_bias = op_convolution.convolve_tensor_with_bias;
pub const convolution_backward_biases = op_convolution.convolution_backward_biases;
pub const convolution_backward_weights = op_convolution.convolution_backward_weights;
pub const convolution_backward_input = op_convolution.convolution_backward_input;
pub const get_convolution_output_shape = op_convolution.get_convolution_output_shape;
