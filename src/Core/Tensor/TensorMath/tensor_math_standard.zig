// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------
//

// ---------- importing standard reduction and logical methods ----------

// ---------- importing standard structural methods ----------
const shape_math_lib = @import("lib_shape_math.zig");

//---reshape
pub const reshape = shape_math_lib.reshape;
pub const reshape_lean = shape_math_lib.reshape_lean;
pub const reshape_lean_f32 = shape_math_lib.reshape_lean_f32;
//---gather
pub const gather = shape_math_lib.gather;
pub const gather_lean = shape_math_lib.lean_gather;

//--unsqueeze
pub const unsqueeze = shape_math_lib.unsqueeze;
pub const unsqueeze_lean = shape_math_lib.unsqueeze_lean;
pub const get_unsqueeze_output_shape = shape_math_lib.get_unsqueeze_output_shape;
//---concatenate
pub const concatenate = shape_math_lib.concatenate;
// TODO: pub const concatenate_lean = shape_math_lib.concatenate_lean;
pub const get_concatenate_output_shape = shape_math_lib.get_concatenate_output_shape;

// ---------- importing pooling methods ----------

pub const calculateStrides = shape_math_lib.calculateStrides;
pub const transpose2D = shape_math_lib.transpose2D;
pub const transposeDefault = shape_math_lib.transposeDefault;
pub const transposeLastTwo = shape_math_lib.transposeLastTwo;
pub const addPaddingAndDilation = shape_math_lib.addPaddingAndDilation;
pub const flip = shape_math_lib.flip;

pub const resize = shape_math_lib.resize;
pub const get_resize_output_shape = shape_math_lib.get_resize_output_shape;

pub const split = shape_math_lib.split;
pub const get_split_output_shapes = shape_math_lib.get_split_output_shapes;

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

//--shape
pub const shape_onnx = shape_math_lib.shape_onnx;
pub const shape_onnx_lean = shape_math_lib.lean_shape_onnx;
pub const get_shape_output_shape = shape_math_lib.get_shape_output_shape;
//--slice
pub const slice_onnx = shape_math_lib.slice_onnx;
pub const slice_onnx_lean = shape_math_lib.lean_slice_onnx;
pub const get_slice_output_shape = shape_math_lib.get_slice_output_shape;

//TODO: pub const sub_tensors_lean = elementWise_math_lib.sub_tensors_lean;
//--mul
pub const lean_matmul = op_mat_mul.lean_mat_mul;
pub const mul = elementWise_math_lib.mul;
pub const mul_lean = elementWise_math_lib.mul_lean;
//--div
pub const div = elementWise_math_lib.div;
pub const div_lean = elementWise_math_lib.div_lean;
//--tanh
pub const tanh = elementWise_math_lib.tanh;
pub const tanh_lean = elementWise_math_lib.tanh_lean;
//--ceil
pub const ceil = elementWise_math_lib.ceil;
pub const ceil_lean = elementWise_math_lib.ceil_lean;

// ---------- importing standard basic methods ----------
const logical_math_lib = @import("lib_logical_math.zig");
pub const isOneHot = logical_math_lib.isOneHot;
pub const isSafe = logical_math_lib.isSafe;
pub const equal = logical_math_lib.equal;

// ---------- importing standard activation function methods ----------
const activation_math_lib = @import("lib_activation_function_math.zig");

//Reduce_mean
pub const reduce_mean = reduction_math_lib.reduce_mean;
pub const reduce_mean_lean = reduction_math_lib.lean_reduce_mean;
//ReLU
pub const ReLU = activation_math_lib.ReLU_standard;
pub const ReLU_lean = activation_math_lib.lean_ReLU;
pub const ReLU_backward = activation_math_lib.ReLU_backward;
//Leaky ReLU
pub const leakyReLU = activation_math_lib.leakyReLU;
pub const leakyReLU_lean = activation_math_lib.lean_leakyReLU;
pub const leakyReLU_backward = activation_math_lib.leakyReLU_backward;
//Sigmoid
pub const sigmoid = activation_math_lib.sigmoid;
pub const sigmoid_lean = activation_math_lib.sigmoid_lean;
pub const sigmoid_backward = activation_math_lib.sigmoid_backward;
//Softmax
pub const softmax = activation_math_lib.softmax;
pub const softmax_lean = activation_math_lib.lean_softmax;
pub const softmax_backward = activation_math_lib.softmax_backward;

//Transpose
pub const transpose_onnx = shape_math_lib.transpose_onnx;
pub const transpose_onnx_lean = shape_math_lib.transpose_onnx_lean;
pub const get_transpose_output_shape = shape_math_lib.get_transpose_output_shape;
