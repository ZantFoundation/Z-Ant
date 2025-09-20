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

// https://onnx.ai/onnx/operators/onnx__Gemm.html
// INPUTS:
//      - Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
//      - Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
//      - Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
//OUTPUTS:
//      - Output tensor of shape (M, N).
// ATTRIBUTES:
//      - alpha. FLOAT (default is '1.0'): Scalar multiplier for the product of input tensors A * B.
//      - beta - FLOAT (default is '1.0'): Scalar multiplier for input tensor C.
//      - transA - INT (default is '0'): Whether A should be transposed
//      - transB - INT (default is '0'): Whether B should be transposed

pub const Gemm = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    input_C: ?*TensorZant,
    output: *TensorZant,
    //attributes:
    alpha: f32, // = 1.0;
    beta: f32, // = 1.0;
    transA: bool, // = false;
    transB: bool, // = false;

    pub fn init(nodeProto: *NodeProto) !Gemm {
        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const input_C = if (nodeProto.input.len > 2) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_C_notFound else null;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var alpha: f32 = 1.0;
        var beta: f32 = 1.0;
        var transA: bool = false;
        var transB: bool = false;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "alpha")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) alpha = attr.f else return error.GemmAphaNotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "beta")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) beta = attr.f else return error.GemmBetaNotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "transA")) |_| {
                if (attr.type == onnx.AttributeType.INT) transA = if (attr.i != 0) true else false else return error.GemmTransANotINT;
            } else if (std.mem.indexOf(u8, attr.name, "transB")) |_| {
                if (attr.type == onnx.AttributeType.INT) transB = if (attr.i != 0) true else false else return error.GemmTransBNotINT;
            }
        }

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input_A.ty;

        return Gemm{
            .input_A = input_A,
            .input_B = input_B,
            .input_C = input_C,
            .output = output,
            .alpha = alpha,
            .beta = beta,
            .transA = transA,
            .transB = transB,
        };
    }

    pub fn get_output_shape(self: Gemm) []usize {
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: Gemm) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input_A);
        try inputs.append(self.input_B);
        if (self.input_C) |bias| {
            try inputs.append(bias);
        }

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Gemm) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output);

        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Gemm, writer: std.fs.File.Writer) !void {
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
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_A.name) });
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
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_B.name) });
        }

        // Input Tensor C is optional! verify the presence
        var tensor_C_string: []u8 = undefined;
        if (self.input_C) |in_C| {
            const sanitized_tensor_C = try utils.getSanitizedName(in_C.name);
            tensor_C_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (in_C.tc == TensorCategory.INITIALIZER) "param_lib." else "",
                "tensor_",
                sanitized_tensor_C,
                ")",
            });
        } else {
            tensor_C_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
        }

        // Check if we need cast operations for mixed precision
        const target_type = self.output.ty.toString();
        const a_type = self.input_A.ty.toString();
        const b_type = self.input_B.ty.toString();
        const c_type = if (self.input_C) |c| c.ty.toString() else target_type;
        const need_a_cast = !std.mem.eql(u8, a_type, target_type);
        const need_b_cast = !std.mem.eql(u8, b_type, target_type);
        const need_c_cast = if (self.input_C != null) !std.mem.eql(u8, c_type, target_type) else false;

        var final_a_string: []const u8 = undefined;
        var final_b_string: []const u8 = undefined;
        var final_c_string: []const u8 = undefined;
        var need_free_a = false;
        var need_free_b = false;
        var need_free_c = false;
        defer if (need_free_a) allocator.free(@constCast(final_a_string));
        defer if (need_free_b) allocator.free(@constCast(final_b_string));
        defer if (need_free_c) allocator.free(@constCast(final_c_string));

        if (need_a_cast) {
            // Generate cast for input A
            const a_name = try utils.getSanitizedName(self.input_A.name);
            const output_name = try utils.getSanitizedName(self.output.name);
            const prefix = if (self.input_A.tc == TensorCategory.INITIALIZER) "param_lib." else "";
            _ = try writer.print(
                \\
                \\    // Cast input A from {s} to {s}
                \\    var tensor_{s}_A_casted_{s} = Tensor({s}).fromShape(&allocator, @constCast({s}tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_A_casted_{s}.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&{s}tensor_{s}), &tensor_{s}_A_casted_{s}, zant.onnx.DataType.FLOAT) catch return -1;
                \\
            , .{
                a_type,
                target_type,
                a_name,
                output_name,
                target_type,
                prefix,
                a_name,
                a_name,
                output_name,
                a_type,
                target_type,
                prefix,
                a_name,
                a_name,
                output_name,
            });
            final_a_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", a_name, "_A_casted_", output_name, ")" });
            need_free_a = true;
        } else {
            final_a_string = tensor_A_string;
        }

        if (need_b_cast) {
            // Generate cast for input B
            const b_name = try utils.getSanitizedName(self.input_B.name);
            const output_name = try utils.getSanitizedName(self.output.name);
            const prefix = if (self.input_B.tc == TensorCategory.INITIALIZER) "param_lib." else "";
            _ = try writer.print(
                \\
                \\    // Cast input B from {s} to {s}
                \\    var tensor_{s}_B_casted_{s} = Tensor({s}).fromShape(&allocator, @constCast({s}tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_B_casted_{s}.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&{s}tensor_{s}), &tensor_{s}_B_casted_{s}, zant.onnx.DataType.FLOAT) catch return -1;
                \\
            , .{
                b_type,
                target_type,
                b_name,
                output_name,
                target_type,
                prefix,
                b_name,
                b_name,
                output_name,
                b_type,
                target_type,
                prefix,
                b_name,
                b_name,
                output_name,
            });
            final_b_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", b_name, "_B_casted_", output_name, ")" });
            need_free_b = true;
        } else {
            final_b_string = tensor_B_string;
        }

        if (need_c_cast and self.input_C != null) {
            // Generate cast for input C
            const c_name = try utils.getSanitizedName(self.input_C.?.name);
            const output_name = try utils.getSanitizedName(self.output.name);
            const prefix = if (self.input_C.?.tc == TensorCategory.INITIALIZER) "param_lib." else "";
            _ = try writer.print(
                \\
                \\    // Cast input C from {s} to {s}
                \\    var tensor_{s}_C_casted_{s} = Tensor({s}).fromShape(&allocator, @constCast({s}tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_C_casted_{s}.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&{s}tensor_{s}), &tensor_{s}_C_casted_{s}, zant.onnx.DataType.FLOAT) catch return -1;
                \\
            , .{
                c_type,
                target_type,
                c_name,
                output_name,
                target_type,
                prefix,
                c_name,
                c_name,
                output_name,
                c_type,
                target_type,
                prefix,
                c_name,
                c_name,
                output_name,
            });
            final_c_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", c_name, "_C_casted_", output_name, ")" });
            need_free_c = true;
        } else {
            final_c_string = tensor_C_string;
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.gemm_lean({s}, {s}, {s}, {s}, {}, {}, {s}, {s}, &tensor_{s} ) catch return -1;
        , .{
            target_type, // T
            final_a_string, // Input tensor A (possibly casted)
            final_b_string, // Input tensor B (possibly casted)
            final_c_string, // Input tensor C (possibly casted)
            self.alpha,
            self.beta,
            if (self.transA) "true" else "false",
            if (self.transB) "true" else "false",
            try utils.getSanitizedName(self.output.name), // Output
        });
    }

    pub fn compute_output_shape() []usize {} // TODO

    pub fn print(self: Gemm) void { // TODO
        std.debug.print("\n Gemm:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Gemm, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_A == old_tensor) {
            self.input_A = new_tensor;
            return;
        }
        if (self.input_B == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.input_C != null and self.input_C.? == old_tensor) {
            self.input_C = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
