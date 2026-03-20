// ---------------------------------------------------------------------------
// ---------------- importing quantization dedicated methods -----------------
// ---------------------------------------------------------------------------
//

// ---------- importing quantization and dequantization methods ----------

//---quantize
const op_quantize = @import("op_quantize.zig");

pub const quantScheme = op_quantize.quantScheme;
pub const quantize = op_quantize.quantize;
pub const lean_quantize_minmax = op_quantize.lean_quantize_minmax;

//---dequantize
const op_dequantize = @import("op_dequantize.zig");

pub const dequantize = op_dequantize.dequantize;
pub const lean_dequantize = op_dequantize.lean_dequantize;

// ---------- importing standard structural methods ----------

//---reshape
const op_reshape = @import("../TensorMath/lib_shape_math/op_reshape.zig");

pub const reshape = op_reshape.reshape;
pub const reshape_lean = op_reshape.reshape_lean;
pub const reshape_lean_f32 = op_reshape.reshape_lean_f32;
pub const reshape_lean_common = op_reshape.reshape_lean_common;
pub const get_reshape_output_shape = op_reshape.get_reshape_output_shape;

//---gather
const op_gather = @import("../TensorMath/lib_shape_math/op_gather.zig");

pub const gather = op_gather.gather;
pub const gather_lean = op_gather.lean_gather;
pub const get_gather_output_shape = op_gather.get_gather_output_shape;

//--pads
const op_pads = @import("../TensorMath/lib_shape_math/op_pads.zig");

pub const pads = op_pads.pads;
pub const pads_lean = op_pads.pads_lean;
pub const get_pads_output_shape = op_pads.get_pads_output_shape;
pub const PadMode = op_pads.PadMode;

//--clip
const op_clip = @import("../TensorMath/lib_elementWise_math/op_clip.zig");

pub const clip = op_clip.clip;
pub const clip_lean = op_clip.lean_clip;

// TODO insert op_shape and op_slice ?

//--unsqueeze
const op_unsqueeze = @import("../TensorMath/lib_shape_math/op_unsqueeze.zig");

pub const unsqueeze = op_unsqueeze.unsqueeze;
pub const unsqueeze_lean = op_unsqueeze.unsqueeze_lean;
pub const get_unsqueeze_output_shape = op_unsqueeze.get_unsqueeze_output_shape;

//---concatenate
const op_concat = @import("../TensorMath/lib_shape_math/op_concatenate.zig");

pub const concatenate = op_concat.concatenate;
pub const concatenate_lean = op_concat.concatenate_lean;
pub const get_concatenate_output_shape = op_concat.get_concatenate_output_shape;

//---identity
const op_identity = @import("../TensorMath/lib_shape_math/op_identity.zig");

pub const identity = op_identity.identity;
pub const identity_lean = op_identity.identity_lean;
pub const get_identity_output_shape = op_identity.get_identity_shape_output;

// ---------- importing pooling methods ----------

// --- transpose
const op_transp = @import("../TensorMath/lib_shape_math/op_transpose.zig");

pub const transpose2D = op_transp.transpose2D;
pub const transposeDefault = op_transp.transposeDefault;
pub const transposeLastTwo = op_transp.transposeLastTwo;

// --- padding
const op_padding = @import("../TensorMath/lib_shape_math/op_padding.zig");

pub const addPaddingAndDilation = op_padding.addPaddingAndDilation;

// --- neg
const op_neg = @import("../TensorMath/lib_shape_math/op_neg.zig");

pub const neg = op_neg.neg;
pub const neg_lean = op_neg.neg_lean;
pub const get_neg_output_shape = op_neg.get_neg_output_shape;

// --- flip
pub const flip = op_neg.flip_matrix;

pub const flip_lean = op_neg.flip_matrix_lean;

// --- resize
const op_resize = @import("../TensorMath/lib_shape_math/op_resize.zig");

pub const resize = op_resize.resize;
pub const get_resize_output_shape = op_resize.get_resize_output_shape;
pub const resize_lean = op_resize.resize_lean;

// --- split
const op_split = @import("../TensorMath/lib_shape_math/op_split.zig");

pub const split = op_split.split;
pub const get_split_output_shapes = op_split.get_split_output_shapes;
pub const split_lean = op_split.split_lean;

// ---------- importing standard Pooling methods ----------
const pooling_math_lib = @import("quant_op_pooling.zig");

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

// ---------- importing matrix algebra methods ----------

// ---------------------------- DENSE OPERATIONS -----------------------------

// --- mat_mul
const quant_op_mat_mul = @import("quant_op_mat_mul.zig");

pub const quant_mat_mul = quant_op_mat_mul.quant_mat_mul;
pub const quant_mat_mul_lean = quant_op_mat_mul.quant_lean_mat_mul;
pub const quant_blocked_mat_mul = quant_op_mat_mul.quant_blocked_mat_mul;
pub const quant_blocked_mat_mul_lean = quant_op_mat_mul.quant_lean_blocked_mat_mul;
pub const get_quant_mat_mul_output_shape = quant_op_mat_mul.get_quant_mat_mul_output_shape;

// --- gemm
const quant_op_gemm = @import("quant_op_gemm.zig");

pub const quant_gemm = quant_op_gemm.quant_gemm;
pub const quant_gemm_lean = quant_op_gemm.quant_lean_gemm;

// --- Convolution
const quant_convolution_math_lib = @import("quant_op_convolution.zig");

pub const convolve_tensor_with_bias = quant_convolution_math_lib.convolve_tensor_with_bias;
pub const convolution_backward_biases = quant_convolution_math_lib.convolution_backward_biases;
pub const get_convolution_output_shape = quant_convolution_math_lib.get_convolution_output_shape;
pub const Conv = quant_convolution_math_lib.OnnxConv;
pub const conv_lean = quant_convolution_math_lib.OnnxConvLean;
pub const setLogFunctionC = quant_convolution_math_lib.setLogFunctionC;
//CONV INTEGER
pub const convInteger_lean = quant_convolution_math_lib.convInteger_lean;

// --------------------------- ELEMENT WISE MATH -----------------------------

// --- addition
const quant_add = @import("quant_op_addition.zig");

pub const quant_sum_tensors = quant_add.quant_sum_tensors;
pub const quant_sum_tensors_lean = quant_add.quant_lean_sum_tensors;
