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
const mathHandler_log = std.log.scoped(.mathHandler);

const utils = IR_zant.utils;

// --- uops ---
const cg_v2 = @import("codegen").codegen_v2;
const Uops = cg_v2.uops;
const UOpBuilder = cg_v2.builder;
const DType = Uops.DType;
const Any = Uops.Any;

// https://onnx.ai/onnx/operators/onnx__MatMul.html#l-onnx-doc-matmul
// INPUTS:
//      - A (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - C (heterogeneous) - T:  output tensor.

pub const MatMul = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !MatMul {
        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        //set the output type:
        if (output_C.ty == tensorZant_lib.TensorType.undefined) output_C.ty = input_A.ty;

        return MatMul{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
        };
    }

    pub fn get_output_shape(self: MatMul) []usize {
        return self.output_C.getShape();
    }

    pub fn get_input_tensors(self: MatMul) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input_A);
        try inputs.append(self.input_B);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: MatMul) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output_C);

        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: MatMul, writer: std.fs.File.Writer) !void {
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

        var element_size_bytes: usize = 4; // Default to f32 size as fallback

        // Determine size from DataType enum
        element_size_bytes = switch (self.input_B.ty) {
            .f32 => @sizeOf(f32),
            .f16 => @sizeOf(f16),
            .i64 => @sizeOf(i64),
            .i32 => @sizeOf(i32),
            .i8 => @sizeOf(i8),
            .u8 => @sizeOf(u8),
            // Add other supported types as needed
            else => blk: {
                mathHandler_log.warn("Warning: Unsupported DataType '{any}' for MatMul input B '{s}'. Assuming f32 size.\n", .{ self.input_B.ty, tensor_B_string });
                break :blk 4;
            },
        };

        const b_dims = self.input_B.getShape().len;
        if (b_dims == 0) {
            mathHandler_log.warn("Error: MatMul input B '{s}' has zero dimensions.\n", .{tensor_B_string});
            return error.InvalidShape; // Avoid panic on empty shape
        }

        const b_width_elements: usize = self.input_B.shape[b_dims - 1];
        const b_width_bytes: usize = b_width_elements * element_size_bytes;

        if (b_width_bytes >= std.atomic.cache_line) { //B is large enough for the new mat mul to work;
            _ = try writer.print(
                \\
                \\    tensMath.blocked_mat_mul_lean(T, {s}, {s}, &tensor_{s}) catch return -1;
            , .{
                tensor_A_string, // Input tensor A
                tensor_B_string, // Input tensor B
                try utils.getSanitizedName(self.output_C.name), // Output tensor C
            });
        } else { //B is not large enough, so we keep the old but improved mat_mul
            _ = try writer.print(
                \\
                \\    tensMath.mat_mul_lean({s}, {s}, {s}, &tensor_{s}) catch return -1;
            , .{
                self.output_C.ty.toString(),
                tensor_A_string, // Input tensor A
                tensor_B_string, // Input tensor B
                try utils.getSanitizedName(self.output_C.name), // Output tensor C
            });
        }
    }

    pub fn compute_output_shape(self: MatMul) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_mat_mul_output_shape(self.input_A.shape, self.input_B.shape);
        self.output_C.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: MatMul) void {
        std.debug.print("\n MatMul:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *MatMul, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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

    pub fn render_lower(self: MatMul, builder: *UOpBuilder) !void {
        const A_id = self.input_A.get_tensorZantID();
        const B_id = self.input_B.get_tensorZantID();
        const out_id = self.output_C.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const strideA = self.input_A.stride;
        const strideB = self.input_B.stride;
        const out_dtype = utils.tensorTypeToDtype(self.output_C.ty);

        lowerMatMul(
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

    /// https://onnx.ai/onnx/operators/onnx__MatMul.html
    pub fn lowerMatMul(
        b: *UOpBuilder,
        A_id: usize, // SSA id of input matrix A
        B_id: usize, // SSA id of input matrix B
        out_id: usize,
        a_shape: []const usize, // A: shape vec (len 2)
        b_shape: []const usize, // B: shape vec (len 2)
        out_shape: []const usize, // [M, N] output shape
        out_dtype: DType,
    ) void {

        // ── Tiny helpers to reduce boilerplate ────────────────────────────
        const r = struct {
            fn rng(bi: *UOpBuilder, end: usize) usize { // RANGE 0..end-1
                return bi.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = end } });
            }
            fn kconst(bi: *UOpBuilder, v: usize) usize { // CONST <v>
                return bi.push(.CONST, .i32, &.{}, Any{ .int = v });
            }
        };

        // ── 1. Logical views for A and B (no data copies) -----------------
        // Calculate default row-major strides
        const a_strides = &[_]usize{ @intCast(a_shape[1]), 1 }; // Strides for [M, K] are [K, 1]
        const b_strides = &[_]usize{ @intCast(b_shape[1]), 1 }; // Strides for [K, N] are [N, 1]

        const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = a_shape, .strides = a_strides } });

        const id_viewB = b.push(.VIEW, out_dtype, &.{B_id}, Any{ .view_meta = .{ .shape = b_shape, .strides = b_strides } });

        // Output buffer

        // ── 2. Outer loops for output dimensions M x N ------------------
        const c_rows = r.rng(b, out_shape[0]); // rows of output // M
        const c_cols = r.rng(b, out_shape[1]); // columns of output // N

        // ── 3. Accumulator register (one per output element) ---------------
        const id_acc = b.push(.DEFINE_ACC, out_dtype, &.{}, null);

        // ── 4. Inner reduction loop over K dimension ----------------------
        const a_cols = r.rng(b, a_shape[1]); // inner dimension for reduction (K = A's columns = B's rows) // K

        // ── 5. GEPs for current A and B elements ------------------------
        const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, c_rows, a_cols }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

        const id_gepB = b.push(.GEP, out_dtype, &.{ id_viewB, a_cols, c_cols }, Any{ .mem_info = .{ .base = id_viewB, .offset = 0, .stride = 1 } });

        // ── 6. Multiply & accumulate  acc += A*B ------------------------
        const id_a = b.push(.LOAD, out_dtype, &.{id_gepA}, null);
        const id_b = b.push(.LOAD, out_dtype, &.{id_gepB}, null);
        _ = b.push(.MULACC, out_dtype, &.{ id_acc, id_a, id_b }, null);

        // close reduction loop
        _ = b.push(.ENDRANGE, .bool, &.{a_cols}, null);

        // ── 7. Write output element ------------------------------------------
        const id_gepC = b.push(.GEP, out_dtype, &.{ out_id, c_rows, c_cols }, Any{ .mem_info = .{ .base = out_id, .offset = 0, .stride = 1 } });

        _ = b.push(.STORE, out_dtype, &.{ id_gepC, id_acc }, null);

        // close outer loops (reverse order)
        _ = b.push(.ENDRANGE, .bool, &.{c_cols}, null);
        _ = b.push(.ENDRANGE, .bool, &.{c_rows}, null);
    }
};
