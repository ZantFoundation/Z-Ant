// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------
//

// ---------- importing standard reduction and logical methods ----------

// ---------- importing standard structural methods ----------
const op_reshape = @import("lib_shape_math/op_reshape.zig");
//---reshape
pub const reshape = op_reshape.reshape;
pub const reshape_lean = op_reshape.reshape_lean;

const op_gather = @import("lib_shape_math/op_grater.zig");

//---gather
pub const gather = op_gather.gather;
pub const gather_lean = op_gather.lean_gather;

const op_unsqueeze = @import("lib_shape_math/op_unsqueeze.zig");
//--unsqueeze
pub const unsqueeze = op_unsqueeze.unsqueeze;
pub const unsqueeze_lean = op_unsqueeze.unsqueeze_lean;

//---concatenate
const op_concatenate = @import("lib_shape_math/op_concatenation.zig");
pub const concatenate = op_concatenate.concatenate;
// TODO: pub const concatenate_lean = shape_math_lib.concatenate_lean;
pub const get_concatenate_output_shape = op_concatenate.get_concatenate_output_shape;

const op_transpose = @import("lib_shape_math/op_transpose.zig");
pub const transpose2D = op_transpose.transpose2D;
pub const transposeDefault = op_transpose.transposeDefault;
pub const transposeLastTwo = op_transpose.transposeLastTwo;

const op_padding = @import("lib_shape_math/op_padding.zig");

pub const addPaddingAndDilation = op_padding.addPaddingAndDilation;

const op_flip = @import("lib_shape_math/op_flip.zig");
pub const flip = op_flip.flip;

const op_resize = @import("lib_shape_math/op_resize.zig");
pub const resize = op_resize.resize;
pub const get_resize_output_shape = op_resize.get_resize_output_shape;

const op_split = @import("lib_shape_math/op_split.zig");
pub const split = op_split.split;
pub const get_split_output_shapes = op_split.get_split_output_shapes;

// ---------- importing matrix algebra methods ----------
const op_mat_mul = @import("op_mat_mul.zig");
//---matmul
pub const mat_mul = op_mat_mul.mat_mul;
pub const mat_mul_lean = op_mat_mul.lean_mat_mul;

pub const dot_product_tensor_flat = op_mat_mul.dot_product_tensor_flat;

// ---------- importing standard gemm method ----------
const op_gemm = @import("op_gemm.zig");
pub const gemm = op_gemm.gemm;
pub const gemm_lean = op_gemm.lean_gemm;

// ---------- importing standard Convolution methods ----------
const convolution_math_lib = @import("op_convolution.zig");
pub const multidim_convolution_with_bias = convolution_math_lib.multidim_convolution_with_bias;
pub const convolve_tensor_with_bias = convolution_math_lib.convolve_tensor_with_bias;
pub const convolution_backward_biases = convolution_math_lib.convolution_backward_biases;
pub const convolution_backward_weights = convolution_math_lib.convolution_backward_weights;
pub const convolution_backward_input = convolution_math_lib.convolution_backward_input;
pub const get_convolution_output_shape = convolution_math_lib.get_convolution_output_shape;
pub const Conv = convolution_math_lib.OnnxConv;
pub const Conv_lean = convolution_math_lib.OnnxConvLean;

// ---------- importing standard Pooling methods ----------
const pooling_math_lib = @import("op_pooling.zig");
pub const pool_tensor = pooling_math_lib.pool_tensor;
pub const multidim_pooling = pooling_math_lib.multidim_pooling;
pub const pool_forward = pooling_math_lib.pool_forward;
pub const pool_backward = pooling_math_lib.pool_backward;

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

const reduction_math_lib = @import("lib_reduction_math.zig");
pub const mean = reduction_math_lib.mean;

// ---------- importing standard Element-Wise math ----------
const elementWise_math_lib = @import("lib_elementWise_math.zig");
//--add bias
pub const add_bias = elementWise_math_lib.add_bias;
//--sum tensors
pub const sum_tensors = elementWise_math_lib.sum_tensors;
pub const sum_tensors_lean = elementWise_math_lib.lean_sum_tensors;
//--sum tensor list
pub const sum_tensor_list = elementWise_math_lib.sum_tensor_list;
pub const sum_tensor_list_lean = elementWise_math_lib.lean_sum_tensor_list;
//--sub
pub const sub_tensors = elementWise_math_lib.sub_tensors;
//TODO: pub const sub_tensors_lean = elementWise_math_lib.sub_tensors_lean;
//--mul
pub const mul = elementWise_math_lib.mul;
pub const mul_lean = elementWise_math_lib.mul_lean;
//--div
pub const div = elementWise_math_lib.div;
pub const div_lean = elementWise_math_lib.div_lean;
//--tanh
pub const tanh = elementWise_math_lib.tanh;
pub const tanh_lean = elementWise_math_lib.tanh_lean;

// ---------- importing standard basic methods ----------
const logical_math_lib = @import("lib_logical_math.zig");
pub const isOneHot = logical_math_lib.isOneHot;
pub const isSafe = logical_math_lib.isSafe;
pub const equal = logical_math_lib.equal;

// ---------- importing standard activation function methods ----------
const op_relu = @import("lib_activation_functions_math/op_relu.zig");
//ReLU
pub const ReLU = op_relu.ReLU_standard;
pub const ReLU_lean = op_relu.lean_ReLU;
pub const ReLU_backward = op_relu.ReLU_backward;

const op_leaky_relu = @import("lib_activation_functions_math/op_leakyRelu.zig");
//Leaky ReLU
pub const leakyReLU = op_leaky_relu.leakyReLU;
pub const leakyReLU_lean = op_leaky_relu.lean_leakyReLU;
pub const leakyReLU_backward = op_leaky_relu.leakyReLU_backward;

const op_sigmoid = @import("lib_activation_functions_math/op_sigmoid.zig");

//Sigmoid
pub const sigmoid = op_sigmoid.sigmoid;
pub const sigmoid_lean = op_sigmoid.sigmoid_lean;
pub const sigmoid_backward = op_sigmoid.sigmoid_backward;

//Softmax
const op_softmax = @import("lib_activation_functions_math/op_softmax.zig");

pub const softmax = op_softmax.softmax;
pub const softmax_lean = op_softmax.lean_softmax;
pub const softmax_backward = op_softmax.softmax_backward;
