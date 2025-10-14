/// The file is now organized into these logical sections:
/// - Standard structural methods (reshape, flatten, squeeze, etc.)
/// - Standard element-wise math (addition, subtraction, multiplication, etc.)
/// - Standard matrix algebra methods (matmul, gemm)
/// - Standard activation function methods (relu, sigmoid, softmax, etc.)
/// - Standard reduction methods (mean, reduce_mean)
/// - Standard pooling methods (maxpool, averagepool)
/// - Standard convolution methods
/// - Standard normalization methods (batch normalization)
/// - Standard quantization methods
/// - Standard utility methods (cast, onehot)
/// - Standard logical methods
// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------

// ---------- importing standard structural methods ----------

//---reshape
const op_reshape = @import("lib_shape_math/op_reshape.zig");

pub const reshape = op_reshape.reshape;
pub const reshape_lean = op_reshape.reshape_lean;
pub const reshape_lean_f32 = op_reshape.reshape_lean_f32;
pub const reshape_lean_common = op_reshape.reshape_lean_common;
pub const get_reshape_output_shape = op_reshape.get_reshape_output_shape;

//---flatten
const op_flatten = @import("lib_shape_math/op_flatten.zig");

pub const flatten = op_flatten.flatten;
pub const flatten_lean = op_flatten.flatten_lean;
pub const get_flatten_output_shape = op_flatten.get_flatten_output_shape;

//---squeeze
const op_squeeze = @import("lib_shape_math/op_squeeze.zig");

pub const squeeze = op_squeeze.squeeze;
pub const squeeze_lean = op_squeeze.squeeze_lean;
pub const get_squeeze_output_shape = op_squeeze.get_squeeze_output_shape;

//---gather
const op_gather = @import("lib_shape_math/op_gather.zig");

pub const gather = op_gather.gather;
pub const gather_lean = op_gather.lean_gather;
pub const get_gather_output_shape = op_gather.get_gather_output_shape;

//---gathernd
const op_gathernd = @import("op_gathernd.zig");

pub const gathernd = op_gathernd.gathernd;
pub const gathernd_lean = op_gathernd.gathernd_lean;
pub const get_gathernd_output_shape = op_gathernd.get_gathernd_output_shape;

//---pads
const op_pads = @import("lib_shape_math/op_pads.zig");

pub const pads = op_pads.pads;
pub const pads_lean = op_pads.pads_lean;
pub const get_pads_output_shape = op_pads.get_pads_output_shape;
pub const PadMode = op_pads.PadMode;

//---pad (ONNX)
const op_pad = @import("op_pad.zig");

pub const pad = op_pad.pad;
pub const get_pad_output_shape = op_pad.get_pad_output_shape;

//---clip
const op_clip = @import("lib_elementWise_math/op_clip.zig");

pub const clip = op_clip.clip;
pub const clip_lean = op_clip.lean_clip;
pub const clip_quantized_lean = op_clip.clip_quantized_lean;
pub const lowerClip = op_clip.lowerClip;

//---floor
const op_floor = @import("lib_elementWise_math/op_floor.zig");

pub const floor = op_floor.floor;
pub const floor_lean = op_floor.floor_lean;
pub const get_floor_output_shape = op_floor.get_floor_output_shape;

//---unsqueeze
const op_unsqueeze = @import("lib_shape_math/op_unsqueeze.zig");

pub const unsqueeze = op_unsqueeze.unsqueeze;
pub const unsqueeze_lean = op_unsqueeze.unsqueeze_lean;
pub const get_unsqueeze_output_shape = op_unsqueeze.get_unsqueeze_output_shape;

//---concatenate
const op_concat = @import("lib_shape_math/op_concatenate.zig");

pub const concatenate = op_concat.concatenate;
pub const concatenate_lean = op_concat.concatenate_lean;
pub const get_concatenate_output_shape = op_concat.get_concatenate_output_shape;

//---identity
const op_identity = @import("lib_shape_math/op_identity.zig");

pub const identity = op_identity.identity;
pub const identity_lean = op_identity.identity_lean;
pub const get_identity_output_shape = op_identity.get_identity_shape_output;

//---transpose
const op_transp = @import("lib_shape_math/op_transpose.zig");

pub const transpose2D = op_transp.transpose2D;
pub const transposeDefault = op_transp.transposeDefault;
pub const transposeLastTwo = op_transp.transposeLastTwo;
pub const transpose_onnx = op_transp.transpose_onnx;
pub const transpose_onnx_lean = op_transp.transpose_onnx_lean;
pub const get_transpose_output_shape = op_transp.get_transpose_output_shape;

//---padding
const op_padding = @import("lib_shape_math/op_padding.zig");

pub const addPaddingAndDilation = op_padding.addPaddingAndDilation;

//---resize
const op_resize = @import("lib_shape_math/op_resize.zig");

pub const resize = op_resize.resize;
pub const get_resize_output_shape = op_resize.get_resize_output_shape;
pub const resize_lean = op_resize.rezise_lean;

//---split
const op_split = @import("lib_shape_math/op_split.zig");

pub const split = op_split.split;
pub const get_split_output_shapes = op_split.get_split_output_shapes;
pub const split_lean = op_split.split_lean;

//---neg
const op_neg = @import("lib_shape_math/op_neg.zig");

pub const neg = op_neg.neg;
pub const neg_lean = op_neg.neg_lean;
pub const get_neg_output_shape = op_neg.get_neg_output_shape;
pub const lowerNeg = op_neg.lowerNeg;
pub const flip = op_neg.flip_matrix;
pub const flip_lean = op_neg.flip_matrix_lean;

//---shape
const op_shape = @import("lib_shape_math/op_shape.zig");

pub const shape_onnx = op_shape.shape_onnx;
pub const shape_onnx_lean = op_shape.lean_shape_onnx;
pub const get_shape_output_shape = op_shape.get_shape_output_shape;

//---slice
const op_slice = @import("lib_shape_math/op_slice.zig");

pub const slice_onnx = op_slice.slice_onnx;
pub const slice_onnx_lean = op_slice.lean_slice_onnx;
pub const get_slice_output_shape = op_slice.get_slice_output_shape;

// ---------- importing standard element-wise math ----------

//---addition
const add = @import("lib_elementWise_math/op_addition.zig");

pub const add_bias = add.add_bias;
pub const sum_tensors = add.sum_tensors;
pub const sum_tensors_lean = add.lean_sum_tensors;
pub const sum_tensor_list = add.sum_tensor_list;
pub const sum_tensor_list_lean = add.lean_sum_tensor_list;

//---subtraction
const sub = @import("lib_elementWise_math/op_subtraction.zig");

pub const sub_tensors = sub.sub_tensors;
pub const sub_tensors_lean = sub.lean_sub_tensors;
pub const lean_sub_tensors_mixed = sub.lean_sub_tensors_mixed;

//---multiplication
const mult = @import("lib_elementWise_math/op_multiplication.zig");

pub const mul = mult.mul;
pub const mul_lean = mult.mul_lean;
pub const get_mul_output_shape = mult.get_mul_output_shape;

//---division
const division = @import("lib_elementWise_math/op_division.zig");

pub const div = division.div;
pub const div_lean = division.div_lean;
pub const lean_div_tensors_mixed = division.lean_div_tensors_mixed;

//---tanh
const tanhy = @import("lib_elementWise_math/op_tanh.zig");

pub const tanh = tanhy.tanh;
pub const tanh_lean = tanhy.tanh_lean;
pub const get_tanh_output_shape = tanhy.get_tanh_output_shape;

//---gelu
const Gelu = @import("lib_elementWise_math/op_gelu.zig");

pub const gelu = Gelu.gelu;
pub const gelu_lean = Gelu.gelu_lean;
pub const get_gelu_output_shape = Gelu.get_gelu_output_shape;

//---ceil
const Ceil = @import("lib_elementWise_math/op_ceil.zig");

pub const ceil = Ceil.ceil;
pub const ceil_lean = Ceil.ceil_lean;
pub const get_ceil_output_shape = Ceil.get_ceil_output_shape;

//---sqrt
const Sqrt = @import("lib_elementWise_math/op_sqrt.zig");

pub const sqrt = Sqrt.sqrt;
pub const sqrt_lean = Sqrt.sqrt_lean;
pub const get_sqrt_output_shape = Sqrt.get_sqrt_output_shape;

//---quantizeLinear
const QuantizeLinear = @import("lib_elementWise_math/op_quantizeLinear.zig");

pub const quantizeLinear = QuantizeLinear.quantizeLinear;
pub const quantizeLinear_lean = QuantizeLinear.quantizeLinear_lean;

//---dequantizeLinear
const DequantizeLinear = @import("lib_elementWise_math/op_dequantizeLinear.zig");

pub const dequantizeLinear = DequantizeLinear.dequantizeLinear;
pub const dequantizeLinear_lean = DequantizeLinear.dequantizeLinear_lean;

// ---------- importing standard matrix algebra methods ----------

//---matmul
const op_mat_mul = @import("op_mat_mul.zig");

pub const mat_mul = op_mat_mul.mat_mul;
pub const mat_mul_lean = op_mat_mul.lean_mat_mul;
pub const lean_matmul = op_mat_mul.lean_mat_mul;
pub const blocked_mat_mul = op_mat_mul.blocked_mat_mul;
pub const blocked_mat_mul_lean = op_mat_mul.lean_blocked_mat_mul;
pub const get_mat_mul_output_shape = op_mat_mul.get_mat_mul_output_shape;

//---gemm
const op_gemm = @import("op_gemm.zig");

pub const gemm = op_gemm.gemm;
pub const gemm_lean = op_gemm.lean_gemm;

// ---------- importing standard activation function methods ----------

//---elu
const op_elu = @import("op_elu.zig");

pub const elu = op_elu.elu;
pub const elu_lean = op_elu.elu_lean;
pub const get_elu_output_shape = op_elu.get_elu_output_shape;

//---relu
const op_relu = @import("lib_activation_function_math/op_reLU.zig");

pub const ReLU = op_relu.ReLU_standard;
pub const ReLU_lean = op_relu.lean_ReLU;

//---leaky_relu
const op_leaky_relu = @import("lib_activation_function_math/op_leaky_reLU.zig");

pub const leakyReLU = op_leaky_relu.leakyReLU;
pub const leakyReLU_lean = op_leaky_relu.lean_leakyReLU;
pub const leakyReLU_backward = op_leaky_relu.leakyReLU_backward;
pub const get_leaky_relu_output_shape = op_leaky_relu.get_leaky_relu_output_shape;

//---sigmoid
const op_sigmoid = @import("lib_activation_function_math/op_sigmoid.zig");

pub const sigmoid = op_sigmoid.sigmoid;
pub const sigmoid_lean = op_sigmoid.sigmoid_lean;
pub const sigmoid_backward = op_sigmoid.sigmoid_backward;
pub const get_sigmoid_output_shape = op_sigmoid.get_sigmoid_output_shape;
pub const lowerSigmoid = op_sigmoid.lowerSigmoid;

//---softmax
const op_softmax = @import("lib_activation_function_math/op_softmax.zig");

pub const softmax = op_softmax.softmax;
pub const softmax_lean = op_softmax.lean_softmax;
pub const softmax_backward = op_softmax.softmax_backward;
pub const get_longsoftmax_output_shape = op_softmax.get_longsoftmax_output_shape;

// ---------- importing standard reduction methods ----------

//---mean
const op_mean = @import("op_mean.zig");

pub const mean_standard = op_mean.mean_standard;
pub const mean_lean = op_mean.mean_lean;
pub const get_mean_output_shape = op_mean.get_mean_output_shape;

//---reduce_mean
const reduction_math_lib = @import("lib_reduction_math.zig");

pub const mean = reduction_math_lib.mean;
pub const reduce_mean = reduction_math_lib.reduce_mean;
pub const reduce_mean_lean = reduction_math_lib.lean_reduce_mean;
pub const get_reduce_mean_output_shape = reduction_math_lib.get_reduce_mean_output_shape;

// ---------- importing standard pooling methods ----------

//---onnx_maxpool
pub const op_maxPool = @import("op_maxPool.zig");
pub const onnx_maxpool = op_maxPool.onnx_maxpool;
pub const onnx_maxpool_lean = op_maxPool.lean_onnx_maxpool;
pub const get_onnx_maxpool_output_shape = op_maxPool.get_onnx_maxpool_output_shape;

//---onnx_averagepool
pub const op_averagePool = @import("op_averagePool.zig");
pub const onnx_averagepool = op_averagePool.onnx_averagepool;
pub const onnx_averagepool_lean = op_averagePool.lean_onnx_averagepool;
pub const get_onnx_averagepool_output_shape = op_averagePool.get_onnx_averagepool_output_shape;
pub const AutoPadType = op_averagePool.AutoPadType;

//---global average pooling
pub const op_globalAveragePool = @import("op_globalAveragePool.zig");
pub const globalAveragePool = op_globalAveragePool.globalAveragePool;
pub const globalAveragePool_lean = op_globalAveragePool.lean_globalAveragePool;
pub const get_global_average_pool_output_shape = op_globalAveragePool.get_global_average_pool_output_shape;

// ---------- importing standard convolution methods ----------

//---convolution
const convolution_math_lib = @import("op_convolution.zig");

pub const get_convolution_output_shape = convolution_math_lib.calculateOutputShape;
pub const conv = convolution_math_lib.conv;
pub const conv_lean = convolution_math_lib.conv_lean;
pub const conv_clip_lean = convolution_math_lib.conv_clip_lean;
pub const setLogFunctionC = convolution_math_lib.setLogFunctionC;

// ---convolution+ReLU
const conv_relu_math_lib = @import("op_conv_relu.zig");
pub const conv_relu = conv_relu_math_lib.conv_relu;
pub const conv_relu_lean = conv_relu_math_lib.conv_relu_lean;
pub const get_conv_relu_output_shape = conv_relu_math_lib.get_conv_relu_output_shape;

//---qlinearconv
const qlinearconv_math_lib = @import("op_qlinearconv.zig");
const qlinearconv_simd_lib = @import("op_qlinearconv_simd.zig");

pub const qlinearconv = qlinearconv_math_lib.qlinearconv;
pub const qlinearconv_lean = qlinearconv_math_lib.qlinearconv_lean;
pub const qlinearconv_embedded_lean = qlinearconv_math_lib.qlinearconv_embedded_lean;
pub const qlinearconv_cmsis_accelerated = qlinearconv_math_lib.qlinearconv_cmsis_accelerated;
pub const qlinearconv_dispatch = qlinearconv_math_lib.qlinearconv_dispatch;
pub const qlinearconv_simd_lean = qlinearconv_simd_lib.qlinearconv_simd_lean;
pub const qlinearconv_onnx_v10 = qlinearconv_simd_lib.qlinearconv_onnx_v10;

pub const get_qlinearconv_output_shape = qlinearconv_math_lib.get_qlinearconv_output_shape;

//---qlinearadd
const qlinearadd_math_lib = @import("op_qlinearadd.zig");

pub const qlinearadd = qlinearadd_math_lib.qlinearadd;
pub const qlinearadd_lean = qlinearadd_math_lib.lean_qlinearadd;
pub const get_qlinearadd_output_shape = qlinearadd_math_lib.get_qlinearadd_output_shape;

//---qlinearglobalaveragepool
const qlinearglobalaveragepool_math_lib = @import("op_qlinearglobalaveragepool.zig");

pub const qlinearglobalaveragepool = qlinearglobalaveragepool_math_lib.qlinearglobalaveragepool;
pub const qlinearglobalaveragepool_lean = qlinearglobalaveragepool_math_lib.lean_qlinearglobalaveragepool;
pub const get_qlinearglobalaveragepool_output_shape = qlinearglobalaveragepool_math_lib.get_qlinearglobalaveragepool_output_shape;

//---qlinearmatmul
const qlinearmatmul_math_lib = @import("op_qlinearmatmul.zig");

pub const qlinearmatmul = qlinearmatmul_math_lib.qlinearmatmul;
pub const qlinearmatmul_lean = qlinearmatmul_math_lib.lean_qlinearmatmul;
pub const qgemm_lean = qlinearmatmul_math_lib.qgemm_lean;
pub const get_qlinearmatmul_output_shape = qlinearmatmul_math_lib.get_qlinearmatmul_output_shape;

//---qlinearmul
const qlinearmul_math_lib = @import("op_qlinearmul.zig");

pub const qlinearmul = qlinearmul_math_lib.qlinearmul;
pub const qlinearmul_lean = qlinearmul_math_lib.lean_qlinearmul;
pub const get_qlinearmul_output_shape = qlinearmul_math_lib.get_qlinearmul_output_shape;

//---qlinearsoftmax
const qlinearsoftmax_math_lib = @import("op_qlinearsoftmax.zig");

pub const qlinearsoftmax = qlinearsoftmax_math_lib.qlinearsoftmax;
pub const qlinearsoftmax_lean = qlinearsoftmax_math_lib.lean_qlinearsoftmax;
pub const get_qlinearsoftmax_output_shape = qlinearsoftmax_math_lib.get_qlinearsoftmax_output_shape;

//---qlinearconcat
const qlinearconcat_math_lib = @import("op_qlinearconcat.zig");

pub const qlinearconcat = qlinearconcat_math_lib.qlinearconcat;
pub const lean_qlinearconcat = qlinearconcat_math_lib.lean_qlinearconcat;
pub const get_qlinearconcat_output_shape = qlinearconcat_math_lib.get_qlinearconcat_output_shape;

//---qlinearaveragepool
const qlinearaveragepool_math_lib = @import("op_qlinearaveragepool.zig");

pub const qlinearaveragepool = qlinearaveragepool_math_lib.qlinearaveragepool;
pub const lean_qlinearaveragepool = qlinearaveragepool_math_lib.lean_qlinearaveragepool;
pub const get_qlinearaveragepool_output_shape = qlinearaveragepool_math_lib.get_qlinearaveragepool_output_shape;

// ---------- importing standard normalization methods ----------

//---batch_normalization
const op_bachNorm = @import("op_batchNormalization.zig");

pub const batchNormalization = op_bachNorm.batchNormalization;
pub const batchNormalization_lean = op_bachNorm.batchNormalization_lean;
pub const get_batchNormalization_output_shape = op_bachNorm.get_batchNormalization_output_shape;

// ---------- importing standard quantization methods ----------

//---dynamic_quantize_linear
const op_DynamicQuantizeLinear = @import("op_DynamicQuantizeLinear.zig");

pub const dynamicQuantizeLinear = op_DynamicQuantizeLinear.dynamicQuantizeLinear;
pub const get_dynamicQuantizeLinear_output_shape = op_DynamicQuantizeLinear.get_dynamicQuantizeLinear_output_shape;
pub const dynamicQuantizeLinear_lean = op_DynamicQuantizeLinear.dynamicQuantizeLinear_lean;

//---convInteger
const quant_convolution_math_lib = @import("../QuantTensorMath/quant_op_convolution.zig");

pub const convInteger_lean = quant_convolution_math_lib.convInteger_lean;

// ---------- importing standard utility methods ----------

//---cast
const op_cast = @import("op_cast.zig");

pub const cast_lean = op_cast.cast_lean;

//---onehot
const op_oneHot = @import("op_oneHot.zig");

pub const oneHot = op_oneHot.onehot;
pub const oneHot_lean = op_oneHot.onehot_lean;
pub const get_oneHot_output_shape = op_oneHot.get_onehot_output_shape;

// ---------- importing standard logical methods ----------

//---exp
const op_exp = @import("op_exp.zig");

pub const exp = op_exp.exp;
pub const exp_lean = op_exp.exp_lean;
pub const get_exp_output_shape = op_exp.get_exp_output_shape;

//---nonmaxsuppression
const op_nonmaxsuppression = @import("op_nonmaxsuppression.zig");

pub const nonmaxsuppression = op_nonmaxsuppression.nonmaxsuppression;
pub const nonmaxsuppression_lean = op_nonmaxsuppression.nonmaxsuppression_lean;
pub const get_nonmaxsuppression_output_shape = op_nonmaxsuppression.get_nonmaxsuppression_output_shape;

//---topk
const op_topk = @import("op_topk.zig");

pub const topk = op_topk.topk;
pub const topk_lean = op_topk.topk_lean;
pub const get_topk_output_shape = op_topk.get_topk_output_shape;

//---min
const op_min = @import("op_min.zig");

pub const min = op_min.min;
pub const min_lean = op_min.min_lean;
pub const min_two = op_min.min_two;
pub const min_two_lean = op_min.min_two_lean;
pub const reduce_min = op_min.reduce_min;
pub const reduce_min_lean = op_min.reduce_min_lean;
pub const get_min_output_shape = op_min.get_min_output_shape;

//---logical
const logical_math_lib = @import("lib_logical_math.zig");

pub const isOneHot = logical_math_lib.isOneHot;
pub const isSafe = logical_math_lib.isSafe;
pub const equal = logical_math_lib.equal;

// --- pow
const op_pow = @import("op_pow.zig");
pub const pow = op_pow.pow;
pub const pow_lean = op_pow.pow_lean;
