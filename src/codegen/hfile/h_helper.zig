const std = @import("std");
const IR_zant = @import("IR_zant");
pub const codegen_options = @import("codegen_options");

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorType = tensorZant_lib.TensorType;
const NodeZant = IR_zant.NodeZant;
const IR_utils = IR_zant.utils;

// --- allocator
const allocator = @import("zant_utils").allocator.allocator;

pub const H_helper = struct {
    input_shape: []usize,
    input_type: []const u8,
    input_shape_size: usize,

    pub fn init() !H_helper {
        const input: TensorZant = try extract_input();

        return H_helper{
            .input_shape = input.shape,
            .input_type = try from_TensorType_to_C_string_type(input.ty),
            .input_shape_size = input.shape.len,
        };
    }
};

fn from_TensorType_to_C_string_type(tensorType: TensorType) ![]const u8 {
    return switch (tensorType) {
        .f32 => "float",
        .f64 => "double",
        .i8 => "int8_t",
        .i16 => "int16_t",
        .i32 => "int32_t",
        .i64 => "int64_t",
        .u8 => "uint8_t",
        .u16 => "uint16_t",
        .u32 => "uint32_t",
        .u64 => "uint64_t",
        else => error.typeNotSupported,
    };
}

fn extract_input() !TensorZant {
    const inputs: []TensorZant = try IR_utils.getInputs(&tensorZant_lib.tensorMap);

    if (inputs.len == 0) return error.noInputAvailable;

    // Finding the first non initializer input
    var primary_index: usize = std.math.maxInt(usize);
    for (inputs, 0..) |*tz, idx| {
        if (tz.tc != tensorZant_lib.TensorCategory.INITIALIZER) {
            primary_index = idx;
            break;
        }
    }

    return inputs[primary_index];
}
