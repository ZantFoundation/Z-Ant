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

//https://onnx.ai/onnx/operators/onnx__Split.html
// INPUTS:
//      - input (heterogeneous) - T: Input tensor
//      - split (optional, heterogeneous) - tensor(int64):
// OUTPUTS:
//      - output (heterogeneous) - T: Output tensor
// ATTRIBUTES:
//      - axis - INT (default is '0'): Indicate up to which input dimension should be split.
//      - num_outputs - INT: Number of outputs
pub const Split = struct {
    input: *TensorZant,
    split: ?*TensorZant,
    outputs: []*TensorZant,
    //attributes:
    axis: i64 = 0, // default = 0,
    num_outputs: i64 = undefined,

    pub fn init(nodeProto: *NodeProto) !Split {
        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const splitTensor = if (nodeProto.input.len > 1) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.axes_notFound else null;

        // Get all outputs
        var outputs = try allocator.alloc(*TensorZant, nodeProto.output.len);
        for (nodeProto.output, 0..) |output_name, i| {
            outputs[i] = if (tensorZant_lib.tensorMap.getPtr(output_name)) |ptr| ptr else return error.output_notFound;
        }

        var axis: i64 = 0;
        var num_outputs: i64 = @intCast(nodeProto.output.len); // Default to actual number of outputs

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i;
            }
            if (std.mem.eql(u8, attr.name, "num_outputs")) {
                if (attr.type == onnx.AttributeType.INT) num_outputs = attr.i;
            }
        }

        //set the output types:
        for (outputs) |output| {
            if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input.ty;
        }

        return Split{
            .input = input,
            .split = splitTensor,
            .outputs = outputs,
            .axis = axis,
            .num_outputs = num_outputs,
        };
    }

    pub fn get_output_shape(self: Split) []usize {
        return self.outputs[0].shape;
    }

    pub fn get_input_tensors(self: Split) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input);
        if (self.split) |s| try inputs.append(s);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Split) ![]*TensorZant {
        return self.outputs;
    }

    pub fn compute_output_shape() ![]usize {} // TODO

    pub fn print(self: Split) void {
        std.debug.print("\n Split: {any}", .{self});
    }

    pub fn write_op(self: Split, writer: std.fs.File.Writer) !void {
        // --- Crea stringa per input
        var tensor_input_string: []u8 = undefined;
        defer allocator.free(tensor_input_string);

        if (self.input.tc == TensorCategory.INITIALIZER) {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input.name),
                ")",
            });
        } else {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input.name),
            });
        }

        // --- Crea stringa per split (se presente)
        var tensor_split_string: []const u8 = undefined;
        var tensor_split_allocated = false;
        defer if (tensor_split_allocated) allocator.free(tensor_split_string);

        if (self.split != null) {
            if (self.split.?.tc == TensorCategory.INITIALIZER) {
                tensor_split_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(self.split.?.name),
                    ")",
                });
            } else {
                tensor_split_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "&tensor_",
                    try utils.getSanitizedName(self.split.?.name),
                });
            }
            tensor_split_allocated = true;
        } else {
            tensor_split_string = "null";
        }

        // Creare l'array di output tensor
        var output_list = std.ArrayList(u8).init(allocator);
        defer output_list.deinit();

        try output_list.appendSlice("[_]Tensor(");
        try output_list.appendSlice(self.input.ty.toString());
        try output_list.appendSlice("){");

        for (self.outputs, 0..) |output, i| {
            if (i > 0) try output_list.appendSlice(", ");
            try output_list.appendSlice("tensor_");
            try output_list.appendSlice(try utils.getSanitizedName(output.name));
        }
        try output_list.appendSlice("}");

        // Chiamare split_lean con la signature corretta
        if (self.split != null) {
            // Con split sizes specifici
            const split_sizes_string = if (self.split.?.tc == TensorCategory.INITIALIZER)
                try std.mem.concat(allocator, u8, &[_][]const u8{ "param_lib.tensor_", try utils.getSanitizedName(self.split.?.name), ".data" })
            else
                try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", try utils.getSanitizedName(self.split.?.name), ".data" });

            _ = try writer.print(
                \\
                \\    // Perform split operation with {d} outputs (with split sizes)
                \\    {{
                \\        var split_outputs = {s};
                \\        var split_outputs_slice: []Tensor({s}) = split_outputs[0..];
                \\        tensMath.split_lean(
                \\            {s},
                \\            {s}, // input tensor
                \\            {d}, // axis
                \\            {s}, // split sizes
                \\            &split_outputs_slice // all outputs
                \\        ) catch return -1;
                \\    }}
            , .{
                self.num_outputs,
                output_list.items,
                self.input.ty.toString(),
                self.input.ty.toString(),
                tensor_input_string,
                self.axis,
                split_sizes_string,
            });
        } else {
            // Senza split sizes - split uniforme
            // Remove & from tensor_input_string for shape access
            const tensor_name = if (std.mem.startsWith(u8, tensor_input_string, "&"))
                tensor_input_string[1..]
            else
                tensor_input_string;
            const input_dim_axis = try std.fmt.allocPrint(allocator, "{s}.shape[{d}]", .{ tensor_name, self.axis });
            defer allocator.free(input_dim_axis);

            _ = try writer.print(
                \\
                \\    // Perform split operation with {d} outputs (uniform split)
                \\    {{
                \\        var split_outputs = {s};
                \\        var split_outputs_slice: []Tensor({s}) = split_outputs[0..];
                \\        const split_size = {s} / {d};
                \\        var uniform_sizes = [_]usize{{split_size}} ** {d};
                \\        tensMath.split_lean(
                \\            {s},
                \\            {s}, // input tensor
                \\            {d}, // axis
                \\            uniform_sizes[0..], // uniform split sizes
                \\            &split_outputs_slice // all outputs
                \\        ) catch return -1;
                \\    }}
            , .{
                self.num_outputs,
                output_list.items,
                self.input.ty.toString(),
                input_dim_axis,
                self.num_outputs,
                self.num_outputs,
                self.input.ty.toString(),
                tensor_input_string,
                self.axis,
            });
        }
    }

    pub fn sobstitute_tensors(self: *Split, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input == old_tensor) {
            self.input = new_tensor;
            return;
        }
        if (self.split != null and self.split.? == old_tensor) {
            self.split = new_tensor;
            return;
        }
        for (self.outputs, 0..) |tensor, i| {
            if (tensor == old_tensor) {
                self.outputs[i] = new_tensor;
                return;
            }
        }
        return error.TensorNotFound;
    }
};
