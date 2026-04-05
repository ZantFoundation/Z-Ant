/// Centralized re-export of all operator math functions.
///
/// Each operator lives in its own folder as:
///   zant_opName.zig   — primary pair: op_name() + op_name_lean()
///   utils_opName.zig  — helpers, get_output_shape, backward, etc.
///   zant_variant.zig  — secondary standard/lean pairs (if any)
///
/// Naming: all public functions use snake_case.
// ---------------------------------------------------------------------------
// ---------------------------- importing methods ----------------------------
// ---------------------------------------------------------------------------

// ===================== structural methods =====================

//---reshape
const op_reshape = @import("op_reshape/zant_reshape.zig");
const op_reshape_f32 = @import("op_reshape/zant_reshape_f32.zig");
const op_reshape_utils = @import("op_reshape/utils_reshape.zig");

pub const reshape = op_reshape.reshape;
pub const reshape_lean = op_reshape.reshape_lean;
pub const reshape_f32 = op_reshape_f32.reshape_f32;
pub const reshape_lean_f32 = op_reshape_f32.reshape_lean_f32;
pub const reshape_lean_common = op_reshape_utils.reshape_lean_common;
pub const get_reshape_output_shape = op_reshape_utils.get_reshape_output_shape;

//---flatten
const op_flatten = @import("op_flatten/zant_flatten.zig");
const op_flatten_utils = @import("op_flatten/utils_flatten.zig");

pub const flatten = op_flatten.flatten;
pub const flatten_lean = op_flatten.flatten_lean;
pub const get_flatten_output_shape = op_flatten_utils.get_flatten_output_shape;

//---squeeze
const op_squeeze = @import("op_squeeze/zant_squeeze.zig");
const op_squeeze_utils = @import("op_squeeze/utils_squeeze.zig");

pub const squeeze = op_squeeze.squeeze;
pub const squeeze_lean = op_squeeze.squeeze_lean;
pub const get_squeeze_output_shape = op_squeeze_utils.get_squeeze_output_shape;

//---unsqueeze
const op_unsqueeze = @import("op_unsqueeze/zant_unsqueeze.zig");
const op_unsqueeze_utils = @import("op_unsqueeze/utils_unsqueeze.zig");

pub const unsqueeze = op_unsqueeze.unsqueeze;
pub const unsqueeze_lean = op_unsqueeze.unsqueeze_lean;
pub const get_unsqueeze_output_shape = op_unsqueeze_utils.get_unsqueeze_output_shape;

//---gather
const op_gather = @import("op_gather/zant_gather.zig");
const op_gather_utils = @import("op_gather/utils_gather.zig");

pub const gather = op_gather.gather;
pub const gather_lean = op_gather.gather_lean;
pub const get_gather_output_shape = op_gather_utils.get_gather_output_shape;

//---gathernd
const op_gathernd = @import("op_gathernd/zant_gathernd.zig");
const op_gathernd_utils = @import("op_gathernd/utils_gathernd.zig");

pub const gathernd = op_gathernd.gathernd;
pub const gathernd_lean = op_gathernd.gathernd_lean;
pub const get_gathernd_output_shape = op_gathernd_utils.get_gathernd_output_shape;

//---concat
const op_concat = @import("op_concat/zant_concat.zig");
const op_concat_utils = @import("op_concat/utils_concat.zig");

pub const concat = op_concat.concat;
pub const concat_lean = op_concat.concat_lean;
pub const get_concat_output_shape = op_concat_utils.get_concat_output_shape;
// backward compat
pub const concatenate = op_concat.concat;
pub const concatenate_lean = op_concat.concat_lean;
pub const get_concatenate_output_shape = op_concat_utils.get_concat_output_shape;

//---identity
const op_identity = @import("op_identity/zant_identity.zig");
const op_identity_utils = @import("op_identity/utils_identity.zig");

pub const identity = op_identity.identity;
pub const identity_lean = op_identity.identity_lean;
pub const get_identity_output_shape = op_identity_utils.get_identity_shape_output;

//---transpose
const op_transp = @import("op_transpose/zant_transpose.zig");
const op_transp_variants = @import("op_transpose/zant_transpose_variants.zig");
const op_transp_utils = @import("op_transpose/utils_transpose.zig");

pub const transpose = op_transp.transpose;
pub const transpose_lean = op_transp.transpose_lean;
pub const transpose2D = op_transp_variants.transpose2D;
pub const transposeDefault = op_transp_variants.transposeDefault;
pub const transposeLastTwo = op_transp_variants.transposeLastTwo;
pub const get_transpose_output_shape = op_transp_utils.get_transpose_output_shape;
// backward compat
pub const transpose_onnx = op_transp.transpose;
pub const transpose_onnx_lean = op_transp.transpose_lean;

//---resize
const op_resize = @import("op_resize/zant_resize.zig");
const op_resize_utils = @import("op_resize/utils_resize.zig");

pub const resize = op_resize.resize;
pub const resize_lean = op_resize.resize_lean;
pub const get_resize_output_shape = op_resize_utils.get_resize_output_shape;

//---split
const op_split = @import("op_split/zant_split.zig");
const op_split_utils = @import("op_split/utils_split.zig");

pub const split = op_split.split;
pub const split_lean = op_split.split_lean;
pub const get_split_output_shapes = op_split_utils.get_split_output_shapes;

//---neg
const op_neg = @import("op_neg/zant_neg.zig");
const op_neg_utils = @import("op_neg/utils_neg.zig");
const op_neg_flip = @import("op_neg/zant_neg_flip.zig");

pub const neg = op_neg.neg;
pub const neg_lean = op_neg.neg_lean;
pub const get_neg_output_shape = op_neg_utils.get_neg_output_shape;
pub const flip = op_neg_flip.flip_matrix;
pub const flip_lean = op_neg_flip.flip_matrix_lean;

//---shape
const op_shape = @import("op_shape/zant_shape.zig");
const op_shape_utils = @import("op_shape/utils_shape.zig");

pub const shape = op_shape.shape;
pub const shape_lean = op_shape.shape_lean;
pub const get_shape_output_shape = op_shape_utils.get_shape_output_shape;
// backward compat
pub const shape_onnx = op_shape.shape;
pub const shape_onnx_lean = op_shape.shape_lean;

//---slice
const op_slice = @import("op_slice/zant_slice.zig");
const op_slice_utils = @import("op_slice/utils_slice.zig");

pub const slice_op = op_slice.slice;
pub const slice_lean = op_slice.slice_lean;
pub const get_slice_output_shape = op_slice_utils.get_slice_output_shape;
// backward compat
pub const slice_onnx = op_slice.slice;
pub const slice_onnx_lean = op_slice.slice_lean;

//---pad (ONNX)
const op_pad = @import("op_pad/zant_pad.zig");
const op_pad_utils = @import("op_pad/utils_pad.zig");

pub const pad = op_pad.pad;
pub const get_pad_output_shape = op_pad_utils.get_pad_output_shape;

//---clip
const op_clip = @import("op_clip/zant_clip.zig");
const op_clip_utils = @import("op_clip/utils_clip.zig");

pub const clip = op_clip.clip;
pub const clip_lean = op_clip.clip_lean;
pub const clip_quantized_lean = op_clip_utils.clip_quantized_lean;
pub const lowerClip = op_clip_utils.lowerClip;
pub const get_clip_output_shape = op_clip_utils.get_clip_output_shape;

//---topk
const op_topk = @import("op_topk/zant_topk.zig");
const op_topk_utils = @import("op_topk/utils_topk.zig");

pub const topk = op_topk.topk;
pub const topk_lean = op_topk.topk_lean;
pub const get_topk_output_shape = op_topk_utils.get_topk_output_shape;

// ===================== element-wise math =====================

//---add
const op_add = @import("op_add/zant_add.zig");
const op_add_utils = @import("op_add/utils_add.zig");
const op_add_list = @import("op_add/zant_add_list.zig");

pub const add_op = op_add.add;
pub const add_lean = op_add.add_lean;
pub const add_bias = op_add_utils.add_bias;
pub const add_list = op_add_list.add_list;
pub const add_list_lean = op_add_list.add_list_lean;
// backward compat
pub const sum_tensors = op_add.add;
pub const sum_tensors_lean = op_add.add_lean;
pub const sum_tensor_list = op_add_list.add_list;
pub const sum_tensor_list_lean = op_add_list.add_list_lean;

//---sub
const op_sub = @import("op_sub/zant_sub.zig");
const op_sub_utils = @import("op_sub/utils_sub.zig");

pub const sub = op_sub.sub;
pub const sub_lean = op_sub.sub_lean;
pub const sub_lean_mixed = op_sub_utils.sub_lean_mixed;
// backward compat
pub const sub_tensors = op_sub.sub;
pub const sub_tensors_lean = op_sub.sub_lean;
pub const lean_sub_tensors_mixed = op_sub_utils.sub_lean_mixed;

//---mul
const op_mul = @import("op_mul/zant_mul.zig");
const op_mul_utils = @import("op_mul/utils_mul.zig");

pub const mul = op_mul.mul;
pub const mul_lean = op_mul.mul_lean;
pub const get_mul_output_shape = op_mul_utils.get_mul_output_shape;

//---div
const op_div = @import("op_div/zant_div.zig");

pub const div = op_div.div;
pub const div_lean = op_div.div_lean;

//---floor
const op_floor = @import("op_floor/zant_floor.zig");
const op_floor_utils = @import("op_floor/utils_floor.zig");

pub const floor = op_floor.floor;
pub const floor_lean = op_floor.floor_lean;
pub const get_floor_output_shape = op_floor_utils.get_floor_output_shape;

//---ceil
const op_ceil = @import("op_ceil/zant_ceil.zig");
const op_ceil_utils = @import("op_ceil/utils_ceil.zig");

pub const ceil = op_ceil.ceil;
pub const ceil_lean = op_ceil.ceil_lean;
pub const get_ceil_output_shape = op_ceil_utils.get_ceil_output_shape;

//---sqrt
const op_sqrt = @import("op_sqrt/zant_sqrt.zig");
const op_sqrt_utils = @import("op_sqrt/utils_sqrt.zig");

pub const sqrt = op_sqrt.sqrt;
pub const sqrt_lean = op_sqrt.sqrt_lean;
pub const get_sqrt_output_shape = op_sqrt_utils.get_sqrt_output_shape;

//---exp
const op_exp = @import("op_exp/zant_exp.zig");
const op_exp_utils = @import("op_exp/utils_exp.zig");

pub const exp = op_exp.exp;
pub const exp_lean = op_exp.exp_lean;
pub const get_exp_output_shape = op_exp_utils.get_exp_output_shape;

//---log
const op_log = @import("op_log/zant_log.zig");
const op_log_utils = @import("op_log/utils_log.zig");

pub const log = op_log.log;
pub const log_lean = op_log.log_lean;

//---pow
const op_pow = @import("op_pow/zant_pow.zig");

pub const pow = op_pow.pow;
pub const pow_lean = op_pow.pow_lean;

//---tanh
const op_tanh = @import("op_tanh/zant_tanh.zig");
const op_tanh_utils = @import("op_tanh/utils_tanh.zig");

pub const tanh = op_tanh.tanh;
pub const tanh_lean = op_tanh.tanh_lean;
pub const get_tanh_output_shape = op_tanh_utils.get_tanh_output_shape;

//---gelu
const op_gelu = @import("op_gelu/zant_gelu.zig");
const op_gelu_utils = @import("op_gelu/utils_gelu.zig");

pub const gelu = op_gelu.gelu;
pub const gelu_lean = op_gelu.gelu_lean;
pub const get_gelu_output_shape = op_gelu_utils.get_gelu_output_shape;

//---quantize_linear
const op_quantize_linear = @import("op_quantizeLinear/zant_quantizeLinear.zig");

pub const quantize_linear = op_quantize_linear.quantize_linear;
pub const quantize_linear_lean = op_quantize_linear.quantize_linear_lean;
// backward compat
pub const quantizeLinear = op_quantize_linear.quantize_linear;
pub const quantizeLinear_lean = op_quantize_linear.quantize_linear_lean;

//---dequantize_linear
const op_dequantize_linear = @import("op_dequantizeLinear/zant_dequantizeLinear.zig");

pub const dequantize_linear = op_dequantize_linear.dequantize_linear;
pub const dequantize_linear_lean = op_dequantize_linear.dequantize_linear_lean;
// backward compat
pub const dequantizeLinear = op_dequantize_linear.dequantize_linear;
pub const dequantizeLinear_lean = op_dequantize_linear.dequantize_linear_lean;

// ===================== matrix algebra =====================

//---mat_mul
const op_mat_mul = @import("op_matMul/zant_matMul.zig");
const op_blocked_mat_mul = @import("op_matMul/zant_blocked_mat_mul.zig");
const op_mat_mul_utils = @import("op_matMul/utils_matMul.zig");

pub const mat_mul = op_mat_mul.mat_mul;
pub const mat_mul_lean = op_mat_mul.mat_mul_lean;
pub const blocked_mat_mul = op_blocked_mat_mul.blocked_mat_mul;
pub const blocked_mat_mul_lean = op_blocked_mat_mul.blocked_mat_mul_lean;
pub const get_mat_mul_output_shape = op_mat_mul_utils.get_mat_mul_output_shape;
// backward compat
pub const lean_matmul = op_mat_mul.mat_mul_lean;

//---gemm
const op_gemm = @import("op_gemm/zant_gemm.zig");
const op_gemm_utils = @import("op_gemm/utils_gemm.zig");

pub const gemm = op_gemm.gemm;
pub const gemm_lean = op_gemm.gemm_lean;

// ===================== activation functions =====================

//---relu
const op_relu = @import("op_relu/zant_relu.zig");

pub const relu = op_relu.relu;
pub const relu_lean = op_relu.relu_lean;
// backward compat
pub const ReLU = op_relu.relu;
pub const ReLU_lean = op_relu.relu_lean;

//---elu
const op_elu = @import("op_elu/zant_elu.zig");
const op_elu_utils = @import("op_elu/utils_elu.zig");

pub const elu = op_elu.elu;
pub const elu_lean = op_elu.elu_lean;
pub const get_elu_output_shape = op_elu_utils.get_elu_output_shape;

//---leaky_relu
const op_leaky_relu = @import("op_leakyRelu/zant_leakyRelu.zig");
const op_leaky_relu_utils = @import("op_leakyRelu/utils_leakyRelu.zig");

pub const leaky_relu = op_leaky_relu.leaky_relu;
pub const leaky_relu_lean = op_leaky_relu.leaky_relu_lean;
pub const leaky_relu_backward = op_leaky_relu_utils.leaky_relu_backward;
pub const get_leaky_relu_output_shape = op_leaky_relu_utils.get_leaky_relu_output_shape;
// backward compat
pub const leakyReLU = op_leaky_relu.leaky_relu;
pub const leakyReLU_lean = op_leaky_relu.leaky_relu_lean;
pub const leakyReLU_backward = op_leaky_relu_utils.leaky_relu_backward;

//---sigmoid
const op_sigmoid = @import("op_sigmoid/zant_sigmoid.zig");
const op_sigmoid_utils = @import("op_sigmoid/utils_sigmoid.zig");

pub const sigmoid = op_sigmoid.sigmoid;
pub const sigmoid_lean = op_sigmoid.sigmoid_lean;
pub const sigmoid_backward = op_sigmoid_utils.sigmoid_backward;
pub const get_sigmoid_output_shape = op_sigmoid_utils.get_sigmoid_output_shape;

//---softmax
const op_softmax = @import("op_softmax/zant_softmax.zig");
const op_softmax_with_axis = @import("op_softmax/zant_softmax_with_axis.zig");

pub const softmax = op_softmax.softmax;
pub const softmax_lean = op_softmax.softmax_lean;
pub const softmax_with_axis = op_softmax_with_axis.softmax_with_axis;
pub const softmax_with_axis_lean = op_softmax_with_axis.softmax_with_axis_lean;

// ===================== reduction methods =====================

//---reduce_mean (from op_reduceMean)
const op_reduce_mean = @import("op_reduceMean/zant_reduceMean.zig");
const op_reduce_mean_utils = @import("op_reduceMean/utils_reduceMean.zig");

pub const reduce_mean_op = op_reduce_mean.reduce_mean;
pub const reduce_mean_op_lean = op_reduce_mean.reduce_mean_lean;
pub const get_reduce_mean_op_output_shape = op_reduce_mean_utils.get_reduce_mean_output_shape;
// backward compat
pub const mean_standard = op_reduce_mean.reduce_mean;
pub const mean_lean = op_reduce_mean.reduce_mean_lean;
pub const get_mean_output_shape = op_reduce_mean_utils.get_reduce_mean_output_shape;

//---min
const op_min = @import("op_min/zant_min.zig");
const op_min_utils = @import("op_min/utils_min.zig");
const op_min_two = @import("op_min/zant_min_two.zig");
const op_reduce_min = @import("op_min/zant_reduce_min.zig");

pub const min = op_min.min;
pub const min_lean = op_min.min_lean;
pub const min_two = op_min_two.min_two;
pub const min_two_lean = op_min_two.min_two_lean;
pub const reduce_min = op_reduce_min.reduce_min;
pub const reduce_min_lean = op_reduce_min.reduce_min_lean;
pub const get_min_output_shape = op_min_utils.get_min_output_shape;

// ===================== pooling methods =====================

//---max_pool
const op_max_pool = @import("op_maxPool/zant_maxPool.zig");
const op_max_pool_utils = @import("op_maxPool/utils_maxPool.zig");

pub const max_pool = op_max_pool.max_pool;
pub const max_pool_lean = op_max_pool.max_pool_lean;
pub const get_max_pool_output_shape = op_max_pool_utils.get_max_pool_output_shape;
// backward compat
pub const onnx_maxpool = op_max_pool.max_pool;
pub const onnx_maxpool_lean = op_max_pool.max_pool_lean;
pub const get_onnx_maxpool_output_shape = op_max_pool_utils.get_max_pool_output_shape;

//---average_pool
const op_average_pool = @import("op_averagePool/zant_averagePool.zig");
const op_average_pool_utils = @import("op_averagePool/utils_averagePool.zig");

pub const average_pool = op_average_pool.average_pool;
pub const average_pool_lean = op_average_pool.average_pool_lean;
pub const get_average_pool_output_shape = op_average_pool_utils.get_average_pool_output_shape;
pub const AutoPadType = op_average_pool.AutoPadType;
// backward compat
pub const onnx_averagepool = op_average_pool.average_pool;
pub const onnx_averagepool_lean = op_average_pool.average_pool_lean;
pub const get_onnx_averagepool_output_shape = op_average_pool_utils.get_average_pool_output_shape;

//---global_average_pool
const op_global_average_pool = @import("op_globalAveragePool/zant_globalAveragePool.zig");
const op_global_average_pool_utils = @import("op_globalAveragePool/utils_globalAveragePool.zig");

pub const global_average_pool = op_global_average_pool.global_average_pool;
pub const global_average_pool_lean = op_global_average_pool.global_average_pool_lean;
pub const get_global_average_pool_output_shape = op_global_average_pool_utils.get_global_average_pool_output_shape;
// backward compat
pub const globalAveragePool = op_global_average_pool.global_average_pool;
pub const globalAveragePool_lean = op_global_average_pool.global_average_pool_lean;

// ===================== convolution methods =====================

//---conv
const op_conv = @import("op_conv/zant_conv.zig");
const op_conv_utils = @import("op_conv/utils_conv.zig");

pub const conv = op_conv.conv;
pub const conv_lean = op_conv.conv_lean;
pub const get_conv_output_shape = op_conv_utils.get_conv_output_shape;
pub const conv_clip_lean = op_conv_utils.conv_clip_lean;
// backward compat
pub const get_convolution_output_shape = op_conv_utils.get_conv_output_shape;

//---qlinear_conv
const op_qlinear_conv = @import("op_qlinearconv/zant_qlinearconv.zig");
const op_qlinear_conv_utils = @import("op_qlinearconv/utils_qlinearconv.zig");

pub const qlinear_conv = op_qlinear_conv.qlinear_conv;
pub const qlinear_conv_lean = op_qlinear_conv.qlinear_conv_lean;
pub const qlinear_conv_embedded_lean = op_qlinear_conv_utils.qlinearconv_embedded_lean;
pub const qlinear_conv_dispatch = op_qlinear_conv_utils.qlinearconv_dispatch;
pub const get_qlinear_conv_output_shape = op_qlinear_conv_utils.get_qlinearconv_output_shape;
// backward compat
pub const qlinearconv = op_qlinear_conv.qlinear_conv;
pub const qlinearconv_lean = op_qlinear_conv.qlinear_conv_lean;
pub const qlinearconv_embedded_lean = op_qlinear_conv_utils.qlinearconv_embedded_lean;
pub const qlinearconv_dispatch = op_qlinear_conv_utils.qlinearconv_dispatch;
pub const get_qlinearconv_output_shape = op_qlinear_conv_utils.get_qlinearconv_output_shape;

//---qlinear_add
const op_qlinear_add = @import("op_qlinearadd/zant_qlinearadd.zig");
const op_qlinear_add_utils = @import("op_qlinearadd/utils_qlinearadd.zig");

pub const qlinear_add = op_qlinear_add.qlinear_add;
pub const qlinear_add_lean = op_qlinear_add.qlinear_add_lean;
pub const get_qlinear_add_output_shape = op_qlinear_add_utils.get_qlinearadd_output_shape;
// backward compat
pub const qlinearadd = op_qlinear_add.qlinear_add;
pub const qlinearadd_lean = op_qlinear_add.qlinear_add_lean;
pub const get_qlinearadd_output_shape = op_qlinear_add_utils.get_qlinearadd_output_shape;

//---qlinear_global_average_pool
const op_qlinear_gap = @import("op_qlinearglobalaveragepool/zant_qlinearglobalaveragepool.zig");
const op_qlinear_gap_utils = @import("op_qlinearglobalaveragepool/utils_qlinearglobalaveragepool.zig");

pub const qlinear_global_average_pool = op_qlinear_gap.qlinear_global_average_pool;
pub const qlinear_global_average_pool_lean = op_qlinear_gap.qlinear_global_average_pool_lean;
pub const get_qlinear_global_average_pool_output_shape = op_qlinear_gap_utils.get_qlinearglobalaveragepool_output_shape;
// backward compat
pub const qlinearglobalaveragepool = op_qlinear_gap.qlinear_global_average_pool;
pub const qlinearglobalaveragepool_lean = op_qlinear_gap.qlinear_global_average_pool_lean;
pub const get_qlinearglobalaveragepool_output_shape = op_qlinear_gap_utils.get_qlinearglobalaveragepool_output_shape;

//---qlinear_mat_mul
const op_qlinear_mat_mul = @import("op_qlinearmatmul/zant_qlinearmatmul.zig");
const op_qlinear_mat_mul_utils = @import("op_qlinearmatmul/utils_qlinearmatmul.zig");

pub const qlinear_mat_mul = op_qlinear_mat_mul.qlinear_mat_mul;
pub const qlinear_mat_mul_lean = op_qlinear_mat_mul.qlinear_mat_mul_lean;
pub const qgemm_lean = op_qlinear_mat_mul_utils.qgemm_lean;
pub const get_qlinear_mat_mul_output_shape = op_qlinear_mat_mul_utils.get_qlinearmatmul_output_shape;
// backward compat
pub const qlinearmatmul = op_qlinear_mat_mul.qlinear_mat_mul;
pub const qlinearmatmul_lean = op_qlinear_mat_mul.qlinear_mat_mul_lean;
pub const get_qlinearmatmul_output_shape = op_qlinear_mat_mul_utils.get_qlinearmatmul_output_shape;

//---qlinear_mul
const op_qlinear_mul = @import("op_qlinearMul/zant_qlinearMul.zig");
const op_qlinear_mul_utils = @import("op_qlinearMul/utils_qlinearMul.zig");

pub const qlinear_mul = op_qlinear_mul.qlinear_mul;
pub const qlinear_mul_lean = op_qlinear_mul.qlinear_mul_lean;
pub const get_qlinear_mul_output_shape = op_qlinear_mul_utils.get_qlinearmul_output_shape;
// backward compat
pub const qlinearmul = op_qlinear_mul.qlinear_mul;
pub const qlinearmul_lean = op_qlinear_mul.qlinear_mul_lean;
pub const get_qlinearmul_output_shape = op_qlinear_mul_utils.get_qlinearmul_output_shape;

//---qlinear_softmax
const op_qlinear_softmax = @import("op_qlinearSoftmax/zant_qlinearSoftmax.zig");
const op_qlinear_softmax_utils = @import("op_qlinearSoftmax/utils_qlinearSoftmax.zig");

pub const qlinear_softmax = op_qlinear_softmax.qlinear_softmax;
pub const qlinear_softmax_lean = op_qlinear_softmax.qlinear_softmax_lean;
pub const get_qlinear_softmax_output_shape = op_qlinear_softmax_utils.get_qlinearsoftmax_output_shape;
// backward compat
pub const qlinearsoftmax = op_qlinear_softmax.qlinear_softmax;
pub const qlinearsoftmax_lean = op_qlinear_softmax.qlinear_softmax_lean;
pub const get_qlinearsoftmax_output_shape = op_qlinear_softmax_utils.get_qlinearsoftmax_output_shape;

//---qlinear_concat
const op_qlinear_concat = @import("op_qlinearconcat/zant_qlinearconcat.zig");
const op_qlinear_concat_utils = @import("op_qlinearconcat/utils_qlinearconcat.zig");

pub const qlinear_concat = op_qlinear_concat.qlinear_concat;
pub const qlinear_concat_lean = op_qlinear_concat.qlinear_concat_lean;
pub const get_qlinear_concat_output_shape = op_qlinear_concat_utils.get_qlinearconcat_output_shape;
// backward compat
pub const lean_qlinearconcat = op_qlinear_concat.qlinear_concat_lean;
pub const get_qlinearconcat_output_shape = op_qlinear_concat_utils.get_qlinearconcat_output_shape;

//---qlinear_average_pool
const op_qlinear_avg_pool = @import("op_qlinearaveragepool/zant_qlinearaveragepool.zig");
const op_qlinear_avg_pool_utils = @import("op_qlinearaveragepool/utils_qlinearaveragepool.zig");

pub const qlinear_average_pool = op_qlinear_avg_pool.qlinear_average_pool;
pub const qlinear_average_pool_lean = op_qlinear_avg_pool.qlinear_average_pool_lean;
pub const get_qlinear_average_pool_output_shape = op_qlinear_avg_pool_utils.get_qlinearaveragepool_output_shape;
// backward compat
pub const qlinearaveragepool = op_qlinear_avg_pool.qlinear_average_pool;
pub const lean_qlinearaveragepool = op_qlinear_avg_pool.qlinear_average_pool_lean;
pub const get_qlinearaveragepool_output_shape = op_qlinear_avg_pool_utils.get_qlinearaveragepool_output_shape;

// ===================== normalization methods =====================

//---batch_normalization
const op_batch_norm = @import("op_batchNormalization/zant_batchNormalization.zig");
const op_batch_norm_utils = @import("op_batchNormalization/utils_batchNormalization.zig");

pub const batch_normalization = op_batch_norm.batch_normalization;
pub const batch_normalization_lean = op_batch_norm.batch_normalization_lean;
pub const get_batch_normalization_output_shape = op_batch_norm_utils.get_batch_normalization_output_shape;
// backward compat
pub const batchNormalization = op_batch_norm.batch_normalization;
pub const batchNormalization_lean = op_batch_norm.batch_normalization_lean;
pub const get_batchNormalization_output_shape = op_batch_norm_utils.get_batch_normalization_output_shape;

// ===================== quantization methods =====================

//---dynamic_quantize_linear
const op_dynamic_quantize = @import("op_dynamicQuantizeLinear/zant_dynamicQuantizeLinear.zig");
const op_dynamic_quantize_utils = @import("op_dynamicQuantizeLinear/utils_dynamicQuantizeLinear.zig");

pub const dynamic_quantize_linear = op_dynamic_quantize.dynamic_quantize_linear;
pub const dynamic_quantize_linear_lean = op_dynamic_quantize.dynamic_quantize_linear_lean;
pub const get_dynamic_quantize_linear_output_shape = op_dynamic_quantize_utils.get_dynamicQuantizeLinear_output_shape;
// backward compat
pub const dynamicQuantizeLinear = op_dynamic_quantize.dynamic_quantize_linear;
pub const dynamicQuantizeLinear_lean = op_dynamic_quantize.dynamic_quantize_linear_lean;
pub const get_dynamicQuantizeLinear_output_shape = op_dynamic_quantize_utils.get_dynamicQuantizeLinear_output_shape;

// ===================== utility methods =====================

//---cast
const op_cast = @import("op_cast/zant_cast.zig");

pub const cast_lean = op_cast.cast_lean;

//---one_hot
const op_one_hot = @import("op_oneHot/zant_oneHot.zig");
const op_one_hot_utils = @import("op_oneHot/utils_oneHot.zig");

pub const one_hot = op_one_hot.one_hot;
pub const one_hot_lean = op_one_hot.one_hot_lean;
pub const get_one_hot_output_shape = op_one_hot_utils.get_one_hot_output_shape;
// backward compat
pub const oneHot = op_one_hot.one_hot;
pub const oneHot_lean = op_one_hot.one_hot_lean;
pub const get_oneHot_output_shape = op_one_hot_utils.get_one_hot_output_shape;

// ===================== logical / misc methods =====================

//---non_max_suppression
const op_nms = @import("op_nonmaxsuppression/zant_nonmaxsuppression.zig");
const op_nms_utils = @import("op_nonmaxsuppression/utils_nonmaxsuppression.zig");

pub const non_max_suppression = op_nms.non_max_suppression;
pub const non_max_suppression_lean = op_nms.non_max_suppression_lean;
pub const get_non_max_suppression_output_shape = op_nms_utils.get_non_max_suppression_output_shape;
// backward compat
pub const nonmaxsuppression = op_nms.non_max_suppression;
pub const nonmaxsuppression_lean = op_nms.non_max_suppression_lean;
pub const get_nonmaxsuppression_output_shape = op_nms_utils.get_non_max_suppression_output_shape;

// ===================== from op_utils / op_pad =====================

//---padding
const op_padding = @import("op_pad/op_padding.zig");

pub const addPaddingAndDilation = op_padding.addPaddingAndDilation;

//---pads
const op_pads = @import("op_pad/op_pads.zig");

pub const pads = op_pads.pads;
pub const pads_lean = op_pads.pads_lean;
pub const get_pads_output_shape = op_pads.get_pads_output_shape;
pub const PadMode = op_pads.PadMode;

//---conv+relu (fused)
const conv_relu_math_lib = @import("op_utils/zant_conv_relu.zig");
pub const conv_relu = conv_relu_math_lib.conv_relu;
pub const conv_relu_lean = conv_relu_math_lib.conv_relu_lean;
pub const get_conv_relu_output_shape = conv_relu_math_lib.get_conv_relu_output_shape;

//---reduce_mean (from op_utils)
const reduction_math_lib = @import("op_utils/zant_reduction_math.zig");

pub const mean = reduction_math_lib.mean;
pub const reduce_mean = reduction_math_lib.reduce_mean;
pub const reduce_mean_lean = reduction_math_lib.lean_reduce_mean;
pub const get_reduce_mean_output_shape = reduction_math_lib.get_reduce_mean_output_shape;

//---logical
const logical_math_lib = @import("op_utils/zant_logical_math.zig");

pub const isOneHot = logical_math_lib.isOneHot;
pub const isSafe = logical_math_lib.isSafe;
pub const equal = logical_math_lib.equal;
