const std = @import("std");
const allocator = std.heap.page_allocator;
const IR_zant = @import("IR_zant");

// --- onnx ---
const onnx = IR_zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = IR_zant.core.math_standard;

const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__Div.html
// INPUTS:
//      - A (heterogeneous) - T: First operand.
//      - B (heterogeneous) - T: Second operand.
// OUTPUTS:
//      - C (heterogeneous) - T: Result, has same element type as two inputs.
pub const Div = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Div {
        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        //set the output type:
        if (output_C.ty == tensorZant_lib.TensorType.undefined) output_C.ty = input_B.ty;

        return Div{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
        };
    }

    pub fn get_output_shape(self: Div) []usize {
        return self.output_C.getShape();
    }

    pub fn get_input_tensors(self: Div) ![]*TensorZant {
        var inputs: std.ArrayList(*TensorZant) = .empty;
        defer inputs.deinit(allocator);

        try inputs.append(allocator, self.input_A);
        try inputs.append(allocator, self.input_B);

        return inputs.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: Div) ![]*TensorZant {
        var outputs: std.ArrayList(*TensorZant) = .empty;
        defer outputs.deinit(allocator);

        try outputs.append(allocator, self.output_C);
        return outputs.toOwnedSlice(allocator);
    }

    pub fn write_op(self: Div, writer: *std.Io.Writer) !void {
        //----create tensor_A_string
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);

        if (self.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",

                try utils.getSanitizedName(self.input_A.name),

                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_A.name), ")" });
        }

        //----create tensor_B_string
        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);
        if (self.input_B.tc == TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_B.name), ")" });
        }

        // Check if we need cast operations for mixed precision
        const target_type = self.output_C.ty.toString();
        const a_type = self.input_A.ty.toString();
        const b_type = self.input_B.ty.toString();
        const need_a_cast = !std.mem.eql(u8, a_type, target_type);
        const need_b_cast = !std.mem.eql(u8, b_type, target_type);

        var final_a_string: []const u8 = undefined;
        var final_b_string: []const u8 = undefined;
        var need_free_a = false;
        var need_free_b = false;
        defer if (need_free_a) allocator.free(@constCast(final_a_string));
        defer if (need_free_b) allocator.free(@constCast(final_b_string));

        if (need_a_cast) {
            // Generate cast for input A
            const a_name = try utils.getSanitizedName(self.input_A.name);
            const prefix = if (self.input_A.tc == TensorCategory.INITIALIZER) "param_lib." else "";
            _ = try writer.print(
                \\
                \\    // Cast input A from {s} to {s}
                \\    var tensor_{s}_A_casted = Tensor({s}).fromShape(&allocator, @constCast({s}tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_A_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&{s}tensor_{s}), &tensor_{s}_A_casted, IR_zant.onnx.DataType.FLOAT) catch return {d};
                \\
            , .{
                a_type,
                target_type,
                a_name,
                target_type,
                prefix,
                a_name,
                a_name,
                a_type,
                target_type,
                prefix,
                a_name,
                a_name,
                utils.getMathErrorReturn(), // Error code for math errors
            });
            final_a_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", a_name, "_A_casted)" });
            need_free_a = true;
        } else {
            final_a_string = tensor_A_string;
        }

        if (need_b_cast) {
            // Generate cast for input B
            const b_name = try utils.getSanitizedName(self.input_B.name);
            const prefix = if (self.input_B.tc == TensorCategory.INITIALIZER) "param_lib." else "";
            _ = try writer.print(
                \\
                \\    // Cast input B from {s} to {s}
                \\    var tensor_{s}_B_casted = Tensor({s}).fromShape(&allocator, @constCast({s}tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_B_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&{s}tensor_{s}), &tensor_{s}_B_casted, IR_zant.onnx.DataType.FLOAT) catch return {d};
                \\
            , .{
                b_type,
                target_type,
                b_name,
                target_type,
                prefix,
                b_name,
                b_name,
                b_type,
                target_type,
                prefix,
                b_name,
                b_name,
                utils.getMathErrorReturn(), // Error code for math errors
            });
            final_b_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", b_name, "_B_casted)" });
            need_free_b = true;
        } else {
            final_b_string = tensor_B_string;
        }

        _ = try writer.print(
            \\
            \\    tensMath.div_lean({s}, {s}, ({s}), &tensor_{s}) catch return {d};
        , .{
            target_type,
            final_a_string, // Input tensor A (possibly casted)
            final_b_string, // Input tensor B (possibly casted)
            try utils.getSanitizedName(self.output_C.name), // Output tensor C
            utils.getMathErrorReturn(), // Error code for math errors
        });
    }

    pub fn compute_output_shape(self: Div) []usize {
        var output_shape: []usize = undefined;
        output_shape = try utils.broadcastShapes(allocator, self.input_A.shape, self.input_B.shape);
        self.output_C.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Div) void {
        std.debug.print("\n Div:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Div, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
