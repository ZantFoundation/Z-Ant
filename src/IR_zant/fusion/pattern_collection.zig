const std = @import("std");

// --- Zant_IR ---
const IR_zant = @import("../IR_zant.zig");
const GraphZant = IR_zant.GraphZant;
const NodeZant = IR_zant.NodeZant;

const fused_operators = IR_zant.fused_operators;
const Op_union = IR_zant.Op_union;

// -- fusion ---
const PatternConfig = @import("pattern_matcher.zig").PatternConfig;

pub const patterns = [_]PatternConfig{
    .{ // "DequantizeLinear" -> "Pad" -> "QuantizeLinear" -> "QLinearConv" into "QLinearConv"
        .pattern = &[_][]const u8{ "DequantizeLinear", "Clip", "QuantizeLinear" },
        .name = "DequantPadQuantQLinConv",
        .fn_pattern_detection = fused_operators.Fused_Dequant_Clip_Quant.fn_pattern_detection,
        .fn_pattern_fusion = fused_operators.Fused_Dequant_Clip_Quant.fn_pattern_fusion,
        .fn_pattern_sobstitution = fused_operators.Fused_Dequant_Clip_Quant.fn_pattern_sobstitution,
    },

    .{ //  DequantizeLinear->Clip->QuantizeLinear into "QLinearConv"
        .pattern = &[_][]const u8{ "DequantizeLinear", "Pad", "QuantizeLinear", "QLinearConv" },
        .name = "DequantPadQuantQLinConv",
        .fn_pattern_detection = fused_operators.Fused_Dequant_Pad_Quant_QLinConv.fn_pattern_detection,
        .fn_pattern_fusion = fused_operators.Fused_Dequant_Pad_Quant_QLinConv.fn_pattern_fusion,
        .fn_pattern_sobstitution = fused_operators.Fused_Dequant_Pad_Quant_QLinConv.fn_pattern_sobstitution,
    },

    .{ // "QuantizeLinear" -> "DequantizeLinear" into nothing
        .pattern = &[_][]const u8{ "QuantizeLinear", "DequantizeLinear" },
        .name = "QuantizeLinearDequantizeLinear",
        .fn_pattern_detection = fused_operators.Fused_Quant_Dequant.fn_pattern_detection,
        .fn_pattern_fusion = fused_operators.Fused_Quant_Dequant.fn_pattern_fusion,
        .fn_pattern_sobstitution = fused_operators.Fused_Quant_Dequant.fn_pattern_sobstitution,
    },

    .{
        .pattern = &[_][]const u8{ "Conv", "Clip" },
        .name = "fused_Conv_Clip", //used for more complex pattern like detect_qadd_pattern()
        .fn_pattern_detection = fused_operators.Fused_Conv_Clip.fn_pattern_detection, // pattern recognition
        .fn_pattern_fusion = fused_operators.Fused_Conv_Clip.fn_pattern_fusion, // fusion stategy
        .fn_pattern_sobstitution = fused_operators.Fused_Conv_Clip.fn_pattern_sobstitution, // sobstitution stategy
    },

    // .{ // "DequantizeLinear" -> "QuantizeLinear" into nothing
    //     .pattern = &[_][]const u8{ "DequantizeLinear", "QuantizeLinear" },
    //     .name = "DequantizeLinearQuantizeLinear",
    //     .fn_pattern_detection = fused_operators.Fused_Dequant_Quant.fn_pattern_detection,
    //     .fn_pattern_fusion = fused_operators.Fused_Dequant_Quant.fn_pattern_fusion,
    //     .fn_pattern_sobstitution = fused_operators.Fused_Dequant_Quant.fn_pattern_sobstitution,
    // },

    // .{
    //     .pattern = &[_][]const u8{ "Conv", "Relu" },
    //     .name = "fused_Conv_Relu", //used for more complex pattern like detect_qadd_pattern()
    //     .fn_pattern_detection = fused_operators.Fused_Conv_Relu.fn_pattern_detection,
    //     .fn_pattern_fusion = fused_operators.Fused_Conv_Relu.fn_pattern_fusion, // fusion stategy
    //     .fn_pattern_sobstitution = fused_operators.Fused_Conv_Relu.fn_pattern_sobstitution, // sobstitution stategy
    // },

    // // Quantized convolution with padding
    // .{
    //     .pattern = &[_][]const u8{ "DequantizeLinear", "Pad", "QuantizeLinear", "QLinearConv" },
    //     .fusion_fn = @import("fuse_DequantizeLinear_Pad_QuantizeLinear_QLinearConv.zig").fuse_dequant_pad_qant_qLinConv,
    //     .name = "QLinearConv",
    //     .pattern_fn = null,
    // },

    // // Standard Conv + BatchNorm + ReLU fusion
    // .{
    //     .pattern = &[_][]const u8{ "Conv", "BatchNormalization", "Relu" },
    //     .fusion_fn = @import("fuse_Conv_BatchNorm_Relu.zig").fuse_conv_batchNormlaization_relu,
    //     .name = "fused_conv_bn_relu",
    //     .pattern_fn = null,
    // },
};

// // TODO, just organized into different files
// pub const patterns_marco = [_]PatternConfig{
//     // DequantizeLinear -> Clip -> QuantizeLinear
//     .{
//         .pattern = &[_][]const u8{ "DequantizeLinear", " Clip ", " QuantizeLinear" },
//         .fusion_fn = @import("fuse_DequantizeLinear_Clip_QuantizeLinearu.zig").fuse_DequantizeLinear_Clip_QuantizeLinearu,
//         .name = "fused_DequantizeLinear_Clip_QuantizeLinearu",
//         .pattern_fn = null,
//     },

//     // DequantizeLinear ->
//     //                     |->  Add -> QuantizeLinear
//     // DequantizeLinear ->
//     .{
//         .pattern = &[_][]const u8{ "DequantizeLinear", "DequantizeLinear", " Add ", " QuantizeLinear" },
//         .fusion_fn = @import("fuse_custom_QAdd.zig").fuse_custom_QAdd.zig,
//         .name = "fused_DequantizeLinear_Clip_QuantizeLinearu",
//         .pattern_fn = @import("fuse_custom_QAdd.zig").detect_qadd_pattern,
//     },

//     // Transpose -> QuantizeLinear
//     .{
//         .pattern = &[_][]const u8{ "Transpose", "QuantizeLinear" },
//         .fusion_fn = @import("fuse_Transpose_QuantizeLinear.zig").fuse_DequantizeLinear_QuantizeLinear,
//         .name = "Transpose_QuantizeLinear",
//         .pattern_fn = null,
//     },

//     //TODO, similar to already existing
//     //  - Pad + QuantizeLinear + QLinearConv fusion pattern
//     //  - Pad + QLinearConv fusion pattern (direct)
// };

// pub const todo_patterns = [_]PatternConfig{
//     // Conv + ReLU (simpler activation fusion)
//     .{
//         .pattern = &[_][]const u8{ "Conv", "Relu" },
//         .fusion_fn = undefined, //TODO fuse_conv_relu,
//         .name = "conv_relu",
//         .pattern_fn = null,
//     },

//     // Linear/Dense layer with activation
//     .{
//         .pattern = &[_][]const u8{ "MatMul", "Add", "Relu" },
//         .fusion_fn = undefined, //TODO fuse_linear_relu,
//         .name = "linear_relu",
//         .pattern_fn = null,
//     },

//     // Quantized linear operations
//     .{
//         .pattern = &[_][]const u8{ "QLinearMatMul", "QLinearAdd" },
//         .fusion_fn = undefined, //TODO fuse_qlinear_matmul_add,
//         .name = "qlinear_dense",
//         .pattern_fn = null,
//     },

//     // GELU activation pattern (common in transformers)
//     .{
//         .pattern = &[_][]const u8{ "MatMul", "Add", "Gelu" },
//         .fusion_fn = undefined, //TODO  fuse_linear_gelu,
//         .name = "linear_gelu",
//         .pattern_fn = null,
//     },

//     // Convolution with bias and activation
//     .{
//         .pattern = &[_][]const u8{ "Conv", "Add", "Relu" },
//         .fusion_fn = undefined, //TODO fuse_conv_bias_relu,
//         .name = "conv_bias_relu",
//         .pattern_fn = null,
//     },

//     // Depthwise separable convolution pattern
//     .{
//         .pattern = &[_][]const u8{ "Conv", "BatchNormalization", "Relu", "Conv", "BatchNormalization" },
//         .fusion_fn = undefined, //TODO fuse_depthwise_separable,
//         .name = "depthwise_separable",
//         .pattern_fn = null,
//     },

//     // Average pooling with activation
//     .{
//         .pattern = &[_][]const u8{ "AveragePool", "Relu" },
//         .fusion_fn = undefined, //TODO fuse_avgpool_relu,
//         .name = "avgpool_relu",
//         .pattern_fn = null,
//     },

//     // Max pooling with activation
//     .{
//         .pattern = &[_][]const u8{ "MaxPool", "Relu" },
//         .fusion_fn = undefined, //TODO fuse_maxpool_relu,
//         .name = "maxpool_relu",
//         .pattern_fn = null,
//     },

//     // Swish/SiLU activation pattern (x * sigmoid(x))
//     .{
//         .pattern = &[_][]const u8{ "Sigmoid", "Mul" },
//         .fusion_fn = undefined, //TODO fuse_swish,
//         .name = "swish_activation",
//         .pattern_fn = null,
//     },

//     // Layer normalization pattern
//     .{
//         .pattern = &[_][]const u8{ "ReduceMean", "Sub", "Mul", "Add" },
//         .fusion_fn = undefined, //TODO fuse_layer_norm,
//         .name = "layer_normalization",
//         .pattern_fn = null,
//     },

//     // Residual connection pattern
//     .{
//         .pattern = &[_][]const u8{ "Conv", "BatchNormalization", "Add" },
//         .fusion_fn = undefined, //TODO fuse_conv_bn_residual,
//         .name = "conv_bn_residual",
//         .pattern_fn = null,
//     },

//     // Quantized activation
//     .{
//         .pattern = &[_][]const u8{ "DequantizeLinear", "Relu", "QuantizeLinear" },
//         .fusion_fn = undefined, //TODO fuse_dequant_relu_quant,
//         .name = "quantized_relu",
//         .pattern_fn = null,
//     },

//     // Attention pattern (simplified)
//     .{
//         .pattern = &[_][]const u8{ "MatMul", "Softmax", "MatMul" },
//         .fusion_fn = undefined, //TODO  fuse_attention,
//         .name = "attention_pattern",
//         .pattern_fn = null,
//     },

//     // Clip/clamp with activation
//     .{
//         .pattern = &[_][]const u8{ "Clip", "Relu" },
//         .fusion_fn = undefined, //TODO fuse_clip_relu,
//         .name = "clip_relu",
//         .pattern_fn = null,
//     },

//     // Leaky ReLU pattern (sometimes split into operations)
//     .{
//         .pattern = &[_][]const u8{ "Mul", "Max" },
//         .fusion_fn = undefined, //TODO  fuse_leaky_relu_pattern,
//         .name = "leaky_relu_pattern",
//         .pattern_fn = null,
//     },

//     // Batch norm inference (scale and shift only)
//     .{
//         .pattern = &[_][]const u8{ "Mul", "Add" },
//         .fusion_fn = undefined, //TODO  fuse_scale_shift,
//         .name = "scale_shift",
//         .pattern_fn = null,
//     },

//     // PReLU pattern
//     .{
//         .pattern = &[_][]const u8{ "Relu", "Neg", "Relu", "Mul", "Add" },
//         .fusion_fn = undefined, //TODO  fuse_prelu,
//         .name = "prelu_activation",
//         .pattern_fn = null,
//     },

//     // Instance normalization
//     .{
//         .pattern = &[_][]const u8{ "ReduceMean", "Sub", "Mul", "Sqrt", "Div", "Mul", "Add" },
//         .fusion_fn = undefined, //TODO fuse_instance_norm,
//         .name = "instance_normalization",
//         .pattern_fn = null,
//     },
// };
