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

// https://onnx.ai/onnx/operators/onnx__Flatten.html
// INPUTS:
//      - data (heterogeneous) - T: Input tensor of any shape.
// OUTPUTS:
//      - output (heterogeneous) - T: Output tensor with shape [outer_dim, inner_dim].
// ATTRIBUTES:
//      - axis - INT (default is '1'): Indicate up to which input dimension should be flattened.
pub const Flatten = struct {
    data: *TensorZant,
    output: *TensorZant,
    //attributes:
    axis: i64 = 1, // default = 1,

    pub fn init(nodeProto: *NodeProto) !Flatten {
        const data = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var axis: i64 = 1;
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != onnx.AttributeType.INT) {
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
            }
        }

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = data.ty;

        return Flatten{
            .data = data,
            .output = output,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Flatten) []usize {
        return self.output.getShape();
    }

    pub fn compute_output_shape(self: Flatten) []usize {
        var output_shape: []usize = undefined;
        const axis = @as(isize, @intCast(self.axis));
        output_shape = try tensorMath.get_flatten_output_shape(self.data.get_shape(), axis);
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn get_input_tensors(self: Flatten) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.data);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Flatten) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output);

        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Flatten, writer: std.fs.File.Writer) !void {
        // Input tensor string generation
        var input_string: []u8 = undefined;
        defer allocator.free(input_string);

        if (self.data.tc == tensorZant_lib.TensorCategory.INITIALIZER) {
            input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.data.name),
                ")",
            });
        } else {
            input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.data.name),
            });
        }

        const output_name = try utils.getSanitizedName(self.output.name);

        // Write the actual operation
        _ = try writer.print(
            \\
            \\
            \\    tensMath.flatten_lean({s}, {s}, &tensor_{s}) catch return -1;
        , .{
            self.data.ty.toString(),
            input_string,
            output_name,
        });
    }

    pub fn print(self: Flatten) void { //TODO
        std.debug.print("\n Flatten:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Flatten, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.data == old_tensor) {
            self.data = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
