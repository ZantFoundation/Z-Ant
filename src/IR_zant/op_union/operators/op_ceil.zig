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

// https://onnx.ai/onnx/operators/onnx__Ceil.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor with ceiling of input elements
pub const Ceil = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Ceil {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Ceil{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Ceil) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Ceil) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        // Append the single input tensor X
        try input_tensors.append(self.input_X);

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Ceil) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        // Append the single output tensor Y
        try output_tensors.append(self.output_Y);

        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: Ceil, writer: std.fs.File.Writer) !void {
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.ceil_lean({s}, {s}, &tensor_{s}) catch return -1;
        , .{
            self.input_X.ty.toString(),
            input_tensor_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn compute_output_shape(self: Ceil) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_ceil_output_shape(self.input_X.get_shape());
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Ceil) void { // TODO
        std.debug.print("\n Ceil:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Ceil, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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

    pub fn render_lower(self: Ceil, builder: *UOpBuilder) !void {
        const X_id = self.input_X.get_tensorZantID();
        const out_id = self.output_Y.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const out_dtype = utils.tensorTypeToDtype(self.output_Y.ty);

        lowerCeil(
            builder,
            X_id,
            out_id,
            out_shape,
            out_dtype,
        );
    }

    pub fn lowerCeil(
        b: *UOpBuilder,
        A_id: usize, // input-tensor SSA ids
        out_id: usize,
        out_shape: []const usize,
        out_dtype: DType, // promoted element type
    ) void { // returns id of result buffer

        // ── Set-up phase ────────────────────────────────────────────────────
        _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

        const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = &.{1} } });

        // ── Flat element loop ───────────────────────────────────────────────
        var nelem: usize = 1;
        for (out_shape) |d| nelem *= d;

        const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

        const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

        const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);

        const id_ceil = b.push(.CLIP, out_dtype, &.{id_loadA}, null);

        const id_gepO = b.push(.GEP, out_dtype, &.{ out_id, id_range }, Any{ .mem_info = .{ .base = out_id, .offset = 0, .stride = 1 } });

        _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_ceil }, null);

        _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);
    }
};
