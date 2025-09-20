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

// https://onnx.ai/onnx/operators/onnx__Gather.html
// INPUTS:
//      - A (heterogeneous) - T:  data tensor.
//      - B (heterogeneous) - T:  indices tensor.
// OUTPUTS:
//      - C (heterogeneous) - T:  result tensor.
// ATTRIBUTES:
//      - axis - INT (default is '0'): Indicate up to which input dimension should be gathered.

pub const Gather = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,
    //attributes:
    axis: i64 = 0, // default = 0,

    pub fn init(nodeProto: *NodeProto) !Gather {
        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        var axis: i64 = 0;
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i;
            }
        }

        //set the output type:
        if (output_C.ty == tensorZant_lib.TensorType.undefined) output_C.ty = input_A.ty;

        return Gather{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Gather) []usize {
        return self.compute_output_shape() catch {
            // Fallback to a default shape in case of error
            std.log.warn("[GATHER DEBUG] Failed to compute output shape, using fallback", .{});
            const fallback_shape = allocator.alloc(usize, 1) catch unreachable;
            fallback_shape[0] = 1;
            return fallback_shape;
        };
    }

    pub fn compute_output_shape(self: Gather) ![]usize {
        const output_shape = try tensorMath.get_gather_output_shape(
            self.input_A.shape,
            self.input_B.shape,
            self.axis,
        );
        self.output_C.shape = output_shape;
        return output_shape;
    }

    pub fn get_input_tensors(self: Gather) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input_A);
        try inputs.append(self.input_B);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Gather) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output_C);

        return outputs.toOwnedSlice();
    }

    // pub fn compute_output_shape(self: Gather) []usize {
    //     var output_shape: []usize = undefined;

    //     output_shape = try utils.usizeSliceToI64Slice(try tensorMath.get_gather_output_shape(
    //         try utils.i64SliceToUsizeSlice(data_shape),
    //         try utils.i64SliceToUsizeSlice(indices_shape),
    //         axis,
    //     ));
    // }

    pub fn write_op(self: Gather, writer: std.fs.File.Writer) !void {
        // Input A (data)
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);
        if (self.input_A.tc == tensorZant_lib.TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_A.name),
                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_A.name),
            });
        }

        // Input B (indices)
        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);
        if (self.input_B.tc == tensorZant_lib.TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_B.name),
            });
        }

        _ = try writer.print(
            \\    
            \\
            \\    const array_usize_{s}_{s}= utils.sliceToUsizeSlice(allocator, tensor_{s}.data);
            \\    defer allocator.free(array_usize_{s}_{s});
        , .{
            try self.input_B.getNameSanitized(), //array_usize_{s}_
            try utils.getSanitizedName(self.output_C.name), //{s}
            try utils.getSanitizedName(self.input_B.name), //tensor_{s}.data
            try self.input_B.getNameSanitized(), //defer allocator.free(array_usize_{s}
            try utils.getSanitizedName(self.output_C.name), //{s});
        });

        _ = try writer.print(
            \\    
            \\
            \\    var tensor_usize_{s}_{s} = Tensor(usize).fromArray(&allocator, array_usize_{s}_{s}, tensor_{s}.shape) catch return -1;
            \\    defer tensor_usize_{s}_{s}.deinit();
        , .{
            try self.input_B.getNameSanitized(), //tensor_usize_{s}_
            try utils.getSanitizedName(self.output_C.name), //{s}
            try self.input_B.getNameSanitized(), //array_usize_{s}_
            try utils.getSanitizedName(self.output_C.name), //{s}
            try utils.getSanitizedName(self.input_B.name), //tensor_{s}.shape
            try self.input_B.getNameSanitized(), //defer tensor_usize_{s}_
            try utils.getSanitizedName(self.output_C.name), //{s}.deinit();
        });

        // Output C
        const output_name = try utils.getSanitizedName(self.output_C.name);

        _ = try writer.print(
            \\
            \\
            \\    tensMath.gather_lean(
            \\        {s}, // input type
            \\        {s}, // input tensor
            \\        &tensor_usize_{s}_{s}, 
            \\        {},
            \\        &tensor_{s},
            \\    ) catch return -1;
        , .{
            self.input_A.ty.toString(),
            tensor_A_string,
            try utils.getSanitizedName(self.input_B.name),
            try utils.getSanitizedName(self.output_C.name),
            self.axis,
            output_name,
        });
    }

    pub fn print(self: Gather) void {
        std.debug.print("\n Gather:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Gather, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_A == old_tensor) {
            self.input_A = new_tensor;
            return;
        }
        if (self.input_B == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.output_C == old_tensor) {
            self.output_C = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
