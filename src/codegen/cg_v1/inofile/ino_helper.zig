const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorType = tensorZant_lib.TensorType;
const NodeZant = IR_zant.NodeZant;
const IR_utils = IR_zant.utils;

pub const Ino_helper = struct {
    input_shape: []usize,
    input_type: []u8,

    output_shape: []usize,
    output_type: []u8,

    pub fn init() !Ino_helper {
        const input: TensorZant = try extract_input();
        const output: TensorZant = try extract_output();

        return Ino_helper{
            .input_shape = input.get_shape(),
            .input_type = try from_TensorType_to_C_string_type(input.ty),
            .output_shape = output.getShape(),
            .output_type = try from_TensorType_to_C_string_type(output.ty),
        };
    }

    pub fn from_shape_to_size(input_shape: []usize) usize {
        var size: usize = 1;
        for (input_shape) |dim| {
            size *= dim;
        }
        return size;
    }
};

///////////////////////////////////////////
//-----------HELPER FUNCTIONS------------//
///////////////////////////////////////////

fn extract_input() !TensorZant {
    const inputs: []TensorZant = try IR_utils.getInputs(TensorZant.tensorMap);

    //TODO gestione corretta di no inputs?
    if (inputs.len == 0) return error.noInputAvailable;

    // Finding the first non initializer input
    //TODO if there are no inputs but only initializers?
    var primary_index: usize = std.math.maxInt(usize);
    for (inputs, 0..) |*tz, idx| {
        if (tz.tc != tensorZant_lib.TensorCategory.INITIALIZER) {
            primary_index = idx;
            break;
        }
    }

    // only initializer???
    // return inputs[0].getShape(); ??
    return inputs[primary_index];
}

fn extract_output() ![]usize {
    const outputs: []TensorZant = try IR_utils.getOutputs(TensorZant.tensorMap);
    return outputs[0];
}

fn from_TensorType_to_C_string_type(tensorType: TensorType) []const u8 {
    return switch (tensorType) {
        .f32 => "float",
        .f64 => "double",
        .i8 => "int8_t",
        .i16 => "int16_t",
        .i32 => "int32_t",
        .i64 => "i64",
        .u8 => "uint8_t",
        .u16 => "uint16_t",
        .u32 => "uint32_t",
        .u64 => "uint64_t",
        else => error.typeNotSupported,
    };
}
