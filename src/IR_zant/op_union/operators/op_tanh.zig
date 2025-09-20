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

//https://onnx.ai/onnx/operators/onnx__Tanh.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Tanh = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Tanh {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Tanh{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Tanh) []usize {
        return self.output_Y.shape;
    }

    pub fn get_input_tensors(self: Tanh) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_X);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Tanh) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn print(self: Tanh) void {
        std.debug.print("\n Tanh: {any}", .{self});
    }

    pub fn compute_output_shape(self: Tanh) []usize {
        const output_shape: []usize = undefined;
        const input_shape = self.input_X.shape;
        output_shape = try tensorMath.get_tanh_output_shape(input_shape);
        return output_shape;
    }

    pub fn write_op(self: Tanh, writer: std.fs.File.Writer) !void {
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

        // --- Write the Tanh op
        _ = try writer.print(
            \\    tensMath.tanh_lean(
            \\        {s},
            \\        {s}, // input tensor
            \\        &tensor_{s} // output tensor
            \\    ) catch return -1;
        , .{
            self.input_X.ty.toString(),
            tensor_X_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn sobstitute_tensors(self: *Tanh, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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

    pub fn render_lower(self: Tanh, builder: *UOpBuilder) !void {
        const X_id = self.input_X.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const out_dtype = utils.tensorTypeToDtype(self.output_Y.ty);

        const out_buf_id = try lowerTanh(
            builder,
            X_id,
            out_shape,
            out_dtype,
        );
        _ = out_buf_id;
    }

    /// https://onnx.ai/onnx/operators/onnx__Tanh.html
    pub fn lowerTanh(
        b: *UOpBuilder,
        A_id: usize, // input-tensor SSA ids
        out_shape: []const usize,
        out_dtype: DType, // promoted element type
    ) usize { // returns id of result buffer

        // ── Set-up phase ────────────────────────────────────────────────────
        _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

        const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = &.{1} } });

        const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

        // ── Flat element loop ───────────────────────────────────────────────
        var nelem: usize = 1;
        for (out_shape) |d| nelem *= d;

        const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

        const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

        const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);

        const id_tanh = b.push(.TANH, out_dtype, &.{id_loadA}, null);

        const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

        _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_tanh }, null);

        _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

        return id_outBuf; // SSA id of the output tensor
    }
};
