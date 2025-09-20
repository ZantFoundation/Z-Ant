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

// https://onnx.ai/onnx/operators/onnx__OneHot.html
// INPUTS:
//      - indices (heterogeneous) - T1: Input tensor
//      - depth (heterogeneous) - T2: Scalar or Rank 1 tensor
//      - values (heterogeneous) - T3: Rank 1 tensor
// OUTPUTS:
//      - output (heterogeneous) - T3: Tensor of rank one greater than input tensor ‘indices’
// ATTRIBUTES:
//      - axis - INT (default is '-1'): (Optional) Axis

pub const OneHot = struct {
    indices: *TensorZant,
    depth: *TensorZant,
    values: *TensorZant,
    output: *TensorZant,
    axis: ?i64,

    pub fn init(nodeProto: *NodeProto) !OneHot {
        const indices = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const depth = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.depth_notFound;
        const values = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.values_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var axis: i64 = -1;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != onnx.AttributeType.INT) {
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
            }
        }

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = indices.ty;

        return OneHot{
            .indices = indices,
            .depth = depth,
            .values = values,
            .output = output,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: OneHot) []usize {
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: OneHot) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.indices);
        try inputs.append(self.depth);
        try inputs.append(self.values);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: OneHot) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output);

        return outputs.toOwnedSlice();
    }

    pub fn compute_output_shape(self: OneHot) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_oneHot_output_shape(
            self.indices.getShape(),
            self.depth.ptr.?.i64.data[0],
            self.axis.?,
        );
        return output_shape;
    }

    pub fn write_op(self: OneHot, writer: std.fs.File.Writer) !void {
        //----create indices string
        var indices_string: []u8 = undefined;
        defer allocator.free(indices_string);
        if (self.indices.tc == TensorCategory.INITIALIZER) {
            indices_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.indices.name),
                ")",
            });
        } else {
            indices_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.indices.name),
                ")",
            });
        }

        //----create depth string
        var depth_string: []u8 = undefined;
        defer allocator.free(depth_string);
        if (self.depth.tc == TensorCategory.INITIALIZER) {
            depth_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.depth.name),
                ")",
            });
        } else {
            depth_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.depth.name),
                ")",
            });
        }

        //----create values string
        var values_string: []u8 = undefined;
        defer allocator.free(values_string);
        if (self.values.tc == TensorCategory.INITIALIZER) {
            values_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.values.name),
                ")",
            });
        } else {
            values_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.values.name),
                ")",
            });
        }

        _ = try writer.print(
            \\    
            \\
            \\    tensMath.oneHot_lean(
            \\        {s}, // T
            \\        {s}, // indices
            \\        {s}.data[0], // depth (scalare)
            \\        {s}, // values
            \\        {?}, // axis
            \\        &tensor_{s}, // output
            \\    ) catch return -1;
        , .{
            self.values.ty.toString(), // T
            indices_string, // indices
            depth_string, // depth
            values_string, // values
            self.axis, // axis
            try utils.getSanitizedName(self.output.name), // output
        });
    }

    pub fn print(self: OneHot) void {
        std.debug.print("\n OneHot:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *OneHot, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.indices == old_tensor) {
            self.indices = new_tensor;
            return;
        }
        if (self.depth == old_tensor) {
            self.depth = new_tensor;
            return;
        }
        if (self.values == old_tensor) {
            self.values = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
