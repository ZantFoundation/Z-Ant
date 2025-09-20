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

// --- uops ---
const cg_v2 = @import("codegen").codegen_v2;
const Uops = cg_v2.uops;
const UOpBuilder = cg_v2.builder;
const DType = Uops.DType;
const Any = Uops.Any;

//https://onnx.ai/onnx/operators/onnx__Sub.html
// INPUTS:
//      - A (heterogeneous) - T: First input tensor
//      - B (heterogeneous) - T: Second input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Sub = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Sub {
        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_A.ty;

        return Sub{
            .input_A = input_A,
            .input_B = input_B,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Sub) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Sub) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_A);
        try inputs.append(self.input_B);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Sub) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn compute_output_shape(self: Sub) []usize {
        var output_shape: []usize = undefined;
        output_shape = try utils.broadcastShapes(allocator, self.input_A.shape, self.input_B.shape);
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Sub) void {
        std.debug.print("\n SUB: {any}", .{self});
    }

    pub fn write_op(self: Sub, writer: std.fs.File.Writer) !void {
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);

        if (self.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_A.name),
                ")",
            });
        } else if (self.input_A.tc == TensorCategory.INPUT) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.input_A.name),
                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_A.name),
            });
        }

        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);

        if (self.input_B.tc == TensorCategory.INITIALIZER or self.input_B.tc == TensorCategory.CONSTANT) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (self.input_B.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        } else if (self.input_B.tc == TensorCategory.INPUT) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_B.name),
            });
        }

        // Check if we need cast operations for mixed precision
        const target_type = self.output_Y.ty.toString();
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
                \\    tensMath.cast_lean({s}, {s}, @constCast(&{s}tensor_{s}), &tensor_{s}_A_casted, zant.onnx.DataType.FLOAT) catch return -1;
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
                \\    tensMath.cast_lean({s}, {s}, @constCast(&{s}tensor_{s}), &tensor_{s}_B_casted, zant.onnx.DataType.FLOAT) catch return -1;
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
            });
            final_b_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", b_name, "_B_casted)" });
            need_free_b = true;
        } else {
            final_b_string = tensor_B_string;
        }

        _ = try writer.print(
            \\    tensMath.sub_tensors_lean(
            \\        {s}, // input type
            \\        {s}, // output type
            \\        {s}, // input A
            \\        {s}, // input B
            \\        &tensor_{s} // output Y
            \\    ) catch return -1;
        , .{
            target_type,
            target_type,
            final_a_string,
            final_b_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn sobstitute_tensors(self: *Sub, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_A == old_tensor) {
            self.input_A = new_tensor;
            return;
        }
        if (self.input_B == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn render_lower(self: Sub, builder: *UOpBuilder) !void {
        const A_id = self.input_A.get_tensorZantID();
        const B_id = self.input_B.get_tensorZantID();
        const out_id = self.output_Y.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const strideA = self.input_A.stride;
        const strideB = self.input_B.stride;
        const out_dtype = utils.tensorTypeToDtype(self.output_Y.ty);

        lowerSub(
            builder,
            A_id,
            B_id,
            out_id,
            out_shape,
            strideA,
            strideB,
            out_dtype,
        );
    }

    /// https://onnx.ai/onnx/operators/onnx__Sub.html
    pub fn lowerSub(
        b: *UOpBuilder,
        A_id: usize, // input-tensor SSA ids
        B_id: usize,
        out_id: usize,
        out_shape: []const usize, // broadcasted shape
        strideA: []const usize, // per-dim strides (0 ⇒ broadcast)
        strideB: []const usize,
        out_dtype: DType, // promoted element type
    ) void { // returns id of result buffer

        // // ── Set-up phase ────────────────────────────────────────────────────
        // _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)
        // _ = b.push(.SHAPE, .i32, &.{B_id}, null); // b_shape  (dbg only)

        const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideA } });

        const id_viewB = b.push(.VIEW, out_dtype, &.{B_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideB } });

        // ── Flat element loop ───────────────────────────────────────────────
        var nelem: usize = 1;
        for (out_shape) |d| nelem *= d;

        const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

        const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

        const id_gepB = b.push(.GEP, out_dtype, &.{ id_viewB, id_range }, Any{ .mem_info = .{ .base = id_viewB, .offset = 0, .stride = 1 } });

        const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);
        const id_loadB = b.push(.LOAD, out_dtype, &.{id_gepB}, null);

        const id_sub = b.push(.SUB, out_dtype, &.{ id_loadA, id_loadB }, null);

        const id_gepO = b.push(.GEP, out_dtype, &.{ out_id, id_range }, Any{ .mem_info = .{ .base = out_id, .offset = 0, .stride = 1 } });

        _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_sub }, null);

        _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);
    }
};
