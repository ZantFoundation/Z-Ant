const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

//https://onnx.ai/onnx/operators/onnx__Transpose.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
// ATTRIBUTES:
//      - perm : A list of integers

pub const Transpose = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    perm: []i64,

    pub fn init(nodeProto: *NodeProto) !Transpose {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        // Get the perm attribute if it exists
        var perm: []i64 = undefined;
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "perm")) {
                if (attr.type == onnx.AttributeType.INTS) {
                    perm = attr.ints;
                }
            }
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Transpose{
            .input_X = input_X,
            .output_Y = output_Y,
            .perm = perm,
        };
    }

    pub fn get_output_shape(self: Transpose) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Transpose) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_X);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Transpose) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn compute_output_shape(self: Transpose) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_transpose_output_shape(
            self.input_X.shape,
            try utils.i64SliceToUsizeSlice(self.perm),
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Transpose) void {
        std.debug.print("\n Transpose: {any}", .{self});
    }

    pub fn write_op(self: Transpose, writer: std.fs.File.Writer) !void {
        // --- Input tensor string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_X.name),
            });
        }

        // --- Perm string
        var perm_string: []const u8 = undefined;
        perm_string = try utils.i64SliceToUsizeArrayString(self.perm);

        // --- Write transpose op
        _ = try writer.print(
            \\    tensMath.transpose_onnx_lean(
            \\        {s}, //input type 
            \\        {s}, // input tensor
            \\        {s}, // perm array
            \\        &tensor_{s}, // output 
            \\        allocator,
            \\    ) catch return -1;
        , .{
            self.input_X.ty.toString(),
            tensor_X_string,
            perm_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn sobstitute_tensors(self: *Transpose, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
