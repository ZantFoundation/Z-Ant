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
pub const tanh = lean_elementWise_math_lib.tanh_lean;
pub const sum_tensor_list = lean_elementWise_math_lib.lean_sum_tensor_list;

// ---------- importing lean Convolution methods ----------
const lean_op_convolution = @import("op_convolution.zig");
pub const im2col = lean_op_convolution.lean_im2col;
pub const OnnxConvLean = lean_op_convolution.OnnxConvLean;

// ---------- importing lean structural methods ----------
const op_reshape = @import("lib_shape_math/op_reshape.zig");
const op_gather = @import("lib_shape_math/op_grater.zig");
const op_unsqueeze = @import("lib_shape_math/op_unsqueeze.zig");
const op_resize = @import("lib_shape_math/op_resize.zig");
const op_concatenate = @import("lib_shape_math/op_concatenation.zig");
const op_split = @import("lib_shape_math/op_split.zig");

pub const reshape = op_reshape.reshape_lean;
pub const gather = op_gather.lean_gather;
pub const unsqueeze = op_unsqueeze.unsqueeze_lean;
pub const get_resize_output_shape = op_resize.get_resize_output_shape;
pub const get_concatenate_output_shape = op_concatenate.get_concatenate_output_shape;
pub const get_split_output_shapes = op_split.get_split_output_shapes;

// ---------- importing lean matrix algebra methods ----------
const op_mat_mul = @import("op_mat_mul.zig");
pub const lean_mat_mul = op_mat_mul.lean_mat_mul;

// ---------- importing lean gemm method ----------
const op_gemm = @import("op_gemm.zig");
pub const gemm_lean = op_gemm.lean_gemm;

// ---------- importing lean activation function methods ----------
const op_relu = @import("lib_activation_functions_math/op_relu.zig");
const op_leaky_relu = @import("lib_activation_functions_math/op_leakyRelu.zig");
const op_sigmoid = @import("lib_activation_functions_math/op_sigmoid.zig");
const op_softmax = @import("lib_activation_functions_math/op_softmax.zig");

//ReLU
pub const ReLU = op_relu.lean_ReLU;
//Leaky ReLU
pub const leakyReLU = op_leaky_relu.lean_leakyReLU;
//Sigmoid
pub const sigmoid = op_sigmoid.sigmoid_lean;
//Softmax
pub const softmax = op_softmax.lean_softmax;

// ---------- importing lean convolution methods ----------
const op_convolution = @import("op_convolution.zig");

pub const convolve_tensor_with_bias = op_convolution.convolve_tensor_with_bias;
pub const convolution_backward_biases = op_convolution.convolution_backward_biases;
pub const convolution_backward_weights = op_convolution.convolution_backward_weights;
pub const convolution_backward_input = op_convolution.convolution_backward_input;
pub const get_convolution_output_shape = op_convolution.get_convolution_output_shape;
