// Conv -> Relu
pub const Fused_Conv_Relu = @import("fused_conv_relu/fused_Conv_Relu.zig").Fused_Conv_Relu;

// Pad -> Conv
pub const Fused_Pad_Conv = @import("fused_pad_conv/fused_Pad_Conv.zig").Fused_Pad_Conv;

// DequantizeLinear -> Pad -> QuantizeLinear -> QLinearConv
pub const Fused_Dequant_Pad_Quant_QLinConv = @import("fused_dequant_pad_quant_qlinconv/fused_Dequant_Pad_Quant_QLinConv.zig").Fused_Dequant_Pad_Quant_QLinConv;

// QuantizeLinear -> DequantizeLinear
pub const Fused_Quant_Dequant = @import("fused_quant_dequant/fused_Quant_Dequant.zig").Fused_Quant_Dequant;

// DequantizeLinear -> QuantizeLinear
pub const Fused_Dequant_Quant = @import("fused_dequant_quant/fused_Dequant_Quant.zig").Fused_Dequant_Quant;

// Conv -> Clip
pub const Fused_Conv_Clip = @import("fused_conv_clip/fused_Conv_Clip.zig").Fused_Conv_Clip;

// DequantizeLinear -> Clip -> QuantizeLinear
pub const Fused_Dequant_Clip_Quant = @import("fused_dequant_clip_quant/fused_Dequant_Clip_Quant.zig").Fused_Dequant_Clip_Quant;

// DequantizeLinear -->
//                     |--> Add -> QuantizeLinear
// DequantizeLinear -->
pub const Fused_2Dequant_Add_Quant = @import("fused_2dequant_add_quant/fused_2Dequant_Add_Quant.zig").Fused_2Dequant_Add_Quant;

//        --> Sigmoid -->
// Conv --|              |--> Mul
//        -------------->
// pub const Fused_Conv_Sigmoid_Mul = @import("fused_conv_sigmoid_mul/fused_Conv_Sigmoid_Mul.zig").Fused_Conv_Sigmoid_Mul;
