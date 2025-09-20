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

// https://onnx.ai/onnx/operators/onnx__Identity.html#l-onnx-doc-identity
// INPUTS:
//      - input (heterogeneous) - V:  input tensor.
// OUTPUTS:
//      - output (heterogeneous) - V:  output tensor.

pub const Identity = struct {
    input: *TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Identity {
        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input.ty;

        return Identity{
            .input = input,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Identity) []usize { // TODO
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: Identity) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Identity) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output);

        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Identity, writer: std.fs.File.Writer) !void {
        // Create input tensor string
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);

        if (self.input.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input.name) });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.identity_lean({s}, {s}, &tensor_{s}) catch return -1;
        , .{
            self.input.ty.toString(),
            input_tensor_string,
            try utils.getSanitizedName(self.output.name),
        });
    }

    pub fn compute_output_shape(self: Identity) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_identity_output_shape(self.input.ptr.?.get_shape());
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Identity) void {
        std.debug.print("\n Identity:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Identity, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input == old_tensor) {
            self.input = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn render_lower(self: Identity, builder: *UOpBuilder) !void {
        const A_id = self.input.get_tensorZantID();
        const StrideA = self.input.stride;
        const out_shape = self.get_output_shape();
        const out_dtype = utils.tensorTypeToDtype(self.output.ty);

        const out_buf_id = lowerIdentity(
            &builder,
            A_id,
            StrideA,
            out_shape,
            out_dtype,
        );
        _ = out_buf_id;
    }

    // https://onnx.ai/onnx/operators/onnx__Identity.html
    pub fn lowerIdentity(
        b: *UOpBuilder,
        A_id: usize, // input-tensor SSA ids
        strideA: []const usize,
        out_shape: []const usize,
        out_dtype: DType, // promoted element type
    ) usize { // returns id of result buffer

        // ── Set-up phase ────────────────────────────────────────────────────
        _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

        const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = strideA } });

        const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

        // ── Copy of the data ───────────────────────────────────────────────

        const copy_id = b.push(.COPY, out_dtype, &.{id_viewA}, null);

        _ = b.push(.STORE, out_dtype, &.{ id_outBuf, copy_id }, null);

        return id_outBuf;
    }
};
