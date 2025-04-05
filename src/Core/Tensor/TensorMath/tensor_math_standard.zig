// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------
//

// ---------- importing standard reduction and logical methods ----------

// ---------- importing standard structural methods ----------
//---reshape
const op_reshape = @import("lib_shape_math/op_reshape.zig");

pub const reshape = op_reshape.reshape;
pub const reshape_lean = op_reshape.reshape_lean;
pub const reshape_lean_f32 = op_reshape.reshape_lean_f32;
pub const reshape_lean_common = op_reshape.reshape_lean_common;
//---gather
const op_gather = @import("lib_shape_math/op_gather.zig");

pub const gather = op_gather.gather;
pub const gather_lean = op_gather.lean_gather;

//--unsqueeze
const op_unsqueeze = @import("lib_shape_math/op_unsqueeze.zig");

pub const unsqueeze = op_unsqueeze.unsqueeze;
pub const unsqueeze_lean = op_unsqueeze.unsqueeze_lean;
pub const get_unsqueeze_output_shape = op_unsqueeze.get_unsqueeze_output_shape;
//---concatenate
const op_concat = @import("lib_shape_math/op_concatenate.zig");

pub const concatenate = op_concat.concatenate;
pub const concatenate_lean = op_concat.lean_concatenate;
pub const get_concatenate_output_shape = op_concat.get_concatenate_output_shape;
//---identity
const op_identity = @import("lib_shape_math/op_identity.zig");

pub const identity = op_identity.identity;
pub const identity_lean = op_identity.identity_lean;
pub const get_identity_output_shape = op_identity.get_identity_shape_output;

// ---------- importing pooling methods ----------
const op_transp = @import("lib_shape_math/op_transpose.zig");
const op_padding = @import("lib_shape_math/op_padding.zig");
const op_resize = @import("lib_shape_math/op_resize.zig");
const op_split = @import("lib_shape_math/op_split.zig");
const op_neg = @import("lib_shape_math/op_neg.zig");

pub const transpose2D = op_transp.transpose2D;
pub const transposeDefault = op_transp.transposeDefault;
pub const transposeLastTwo = op_transp.transposeLastTwo;
pub const addPaddingAndDilation = op_padding.addPaddingAndDilation;

pub const neg = op_neg.neg;
pub const neg_lean = op_neg.neg_lean;
pub const get_neg_output_shape = op_neg.get_neg_output_shape;
pub const flip = op_neg.flip_matrix;
pub const flip_lean = op_neg.flip_matrix_lean;

pub const resize = op_resize.resize;
pub const get_resize_output_shape = op_resize.get_resize_output_shape;
pub const resize_lean = op_resize.rezise_lean;
pub const split = op_split.split;
pub const get_split_output_shapes = op_split.get_split_output_shapes;
pub const split_lean = op_split.split_lean;
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
pub const conv_lean = convolution_math_lib.OnnxConvLean;
pub const setLogFunctionC = convolution_math_lib.setLogFunctionC;

// ---------- importing standard Pooling methods ----------
const pooling_math_lib = @import("op_pooling.zig");
pub const pool_tensor = pooling_math_lib.pool_tensor;
pub const multidim_pooling = pooling_math_lib.multidim_pooling;
pub const pool_forward = pooling_math_lib.pool_forward;
pub const pool_backward = pooling_math_lib.pool_backward;
pub const onnx_maxpool = pooling_math_lib.onnx_maxpool;
pub const onnx_maxpool_lean = pooling_math_lib.lean_onnx_maxpool;
pub const AutoPadType = pooling_math_lib.AutoPadType;
pub const get_onnx_maxpool_output_shape = pooling_math_lib.get_onnx_maxpool_output_shape;
pub const get_pooling_output_shape = pooling_math_lib.get_pooling_output_shape;
pub const PoolingType = pooling_math_lib.PoolingType;

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

const reduction_math_lib = @import("lib_reduction_math.zig");
pub const mean = reduction_math_lib.mean;
pub const reduce_mean = reduction_math_lib.reduce_mean;
pub const reduce_mean_lean = reduction_math_lib.lean_reduce_mean;
pub const get_reduce_mean_output_shape = reduction_math_lib.get_reduce_mean_output_shape;
// ---------- importing standard Element-Wise math ----------
const add = @import("lib_elementWise_math/op_addition.zig");
//--add bias
pub const add_bias = add.add_bias;
//--sum tensors
pub const sum_tensors = add.sum_tensors;
pub const sum_tensors_lean = add.lean_sum_tensors;
//--sum tensor list
pub const sum_tensor_list = add.sum_tensor_list;
pub const sum_tensor_list_lean = add.lean_sum_tensor_list;
//--sub
const sub = @import("lib_elementWise_math/op_subtraction.zig");

pub const sub_tensors = sub.sub_tensors;
pub const sub_tensors_lean = sub.lean_sub_tensors;

//--shape
const op_shape = @import("lib_shape_math/op_shape.zig");
pub const shape_onnx = op_shape.shape_onnx;
pub const shape_onnx_lean = op_shape.lean_shape_onnx;
pub const get_shape_output_shape = op_shape.get_shape_output_shape;

//--slice
const op_slice = @import("lib_shape_math/op_slice.zig");
pub const slice_onnx = op_slice.slice_onnx;
pub const slice_onnx_lean = op_slice.lean_slice_onnx;
pub const get_slice_output_shape = op_slice.get_slice_output_shape;

const mult = @import("lib_elementWise_math/op_multiplication.zig");

//TODO: pub const sub_tensors_lean = elementWise_math_lib.sub_tensors_lean;
//--mul
pub const lean_matmul = op_mat_mul.lean_mat_mul;
pub const mul = mult.mul;
pub const mul_lean = mult.mul_lean;

//--div
const division = @import("lib_elementWise_math/op_division.zig");

pub const div = division.div;
pub const div_lean = division.div_lean;

//--tanh
const tanhy = @import("lib_elementWise_math/op_tanh.zig");
pub const tanh = tanhy.tanh;
pub const tanh_lean = tanhy.tanh_lean;

//--ceil
const Ceil = @import("lib_elementWise_math/op_ceil.zig");

pub const ceil = Ceil.ceil;
pub const ceil_lean = Ceil.ceil_lean;

// ---------- importing standard basic methods ----------
const logical_math_lib = @import("lib_logical_math.zig");
pub const isOneHot = logical_math_lib.isOneHot;
pub const isSafe = logical_math_lib.isSafe;
pub const equal = logical_math_lib.equal;

const op_relu = @import("lib_activation_function_math/op_reLU.zig");
//ReLU
pub const ReLU = op_relu.ReLU_standard;
pub const ReLU_lean = op_relu.lean_ReLU;
pub const ReLU_backward = op_relu.ReLU_backward;

const op_leaky_relu = @import("lib_activation_function_math/op_leaky_reLU.zig");
//Leaky ReLU
pub const leakyReLU = op_leaky_relu.leakyReLU;
pub const leakyReLU_lean = op_leaky_relu.lean_leakyReLU;
pub const leakyReLU_backward = op_leaky_relu.leakyReLU_backward;

const op_sigmoid = @import("lib_activation_function_math/op_sigmoid.zig");

//Sigmoid
pub const sigmoid = op_sigmoid.sigmoid;
pub const sigmoid_lean = op_sigmoid.sigmoid_lean;
pub const sigmoid_backward = op_sigmoid.sigmoid_backward;

//Softmax
const op_softmax = @import("lib_activation_function_math/op_softmax.zig");

pub const softmax = op_softmax.softmax;
pub const softmax_lean = op_softmax.lean_softmax;
pub const softmax_backward = op_softmax.softmax_backward;

//Transpose
const op_Transpose = @import("lib_shape_math/op_transpose.zig");
pub const transpose_onnx = op_Transpose.transpose_onnx;
pub const transpose_onnx_lean = op_Transpose.transpose_onnx_lean;
pub const get_transpose_output_shape = op_Transpose.get_transpose_output_shape;
