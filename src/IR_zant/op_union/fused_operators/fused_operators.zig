// Conv -> Relu
pub const Fused_Conv_Relu = @import("fused_Conv_Relu.zig").Fused_Conv_Relu;

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
