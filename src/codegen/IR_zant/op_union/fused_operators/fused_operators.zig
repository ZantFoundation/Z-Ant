// Conv -> Relu
pub const Fused_Conv_Relu = @import("fused_Conv_Relu.zig").Fused_Conv_Relu;

// Pad -> Conv
pub const Fused_Pad_Conv = @import("fused_Pad_Conv.zig").Fused_Pad_Conv;

// DequantizeLinear -> pad -> QuantizeLinear -> QLinearConv
pub const Fused_Dequant_Pad_Quant_QLinConv = @import("fused_Dequant_Pad_Quant_QLinConv.zig").Fused_Dequant_Pad_Quant_QLinConv;

// QuantizeLinear -> DequantizeLinear
pub const Fused_Quant_Dequant = @import("fused_Quant_Dequant.zig").Fused_Quant_Dequant;

// DequantizeLinear -> QuantizeLinear
pub const Fused_Dequant_Quant = @import("fused_Dequant_Quant.zig").Fused_Dequant_Quant;

// Conv -> Clip
pub const Fused_Conv_Clip = @import("fused_Conv_Clip.zig").Fused_Conv_Clip;

// DequantizeLinear -> Clip -> QuantizeLinear
pub const Fused_Dequant_Clip_Quant = @import("fused_Dequant_Clip_Quant.zig").Fused_Dequant_Clip_Quant;

// DequantizeLinear-->
//                    |--> Add->QuantizeLinear operation for better performance
// DequantizeLinear-->
pub const Fused_2Dequant_Add_Quant = @import("fused_2Dequant_Add_Quant.zig").Fused_2Dequant_Add_Quant;

//        --> Sigmoid -->
// Conv --|              |-->Mul
//        -------------->
// Open YOLOv11n with netron to see the pattern
// pub const Fused_Conv_Sigmoid_Mul = @import("fused_Conv_Sigmoid_Mul.zig").Fused_Conv_Sigmoid_Mul;
