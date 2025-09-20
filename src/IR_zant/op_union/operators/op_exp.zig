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

//https://onnx.ai/onnx/operators/onnx__Exp.html
// INPUTS:
//      - input (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - output (heterogeneous) - T: Output tensor
pub const Exp = struct {
    input: *TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Exp {
        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input.ty;

        return Exp{
            .input = input,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Exp) []usize {
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: Exp) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Exp) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Exp, writer: std.fs.File.Writer) !void {
        //----create tensor_input_string
        var tensor_input_string: []u8 = undefined;
        defer allocator.free(tensor_input_string);
        if (self.input.tc == TensorCategory.INITIALIZER) {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input.name),
                ")",
            });
        } else {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input.name) });
        }

        _ = try writer.print(
            \\
            \\    tensMath.exp_lean(
            \\      {s},
            \\      {s},
            \\      &tensor_{s},
            \\    ) catch return -1;
        ,
            .{
                self.input.ty.toString(),
                tensor_input_string,
                try utils.getSanitizedName(self.output.name),
            },
        );
    }

    pub fn compute_output_shape(self: Exp) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_exp_output_shape(self.input.shape);
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Exp) void {
        std.debug.print("\n Exp: {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Exp, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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

    pub fn render_lower(self: Exp, builder: *UOpBuilder) !void {
        const X_id = self.input.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const out_dtype = utils.tensorTypeToDtype(self.output.ty);

        const out_buf_id = lowerExp(
            builder,
            X_id,
            out_shape,
            out_dtype,
        );
        _ = out_buf_id;
    }

    /// https://onnx.ai/onnx/operators/onnx__Exp.html
    pub fn lowerExp(
        b: *UOpBuilder,
        X_id: usize, // SSA id of input tensor X
        x_shape: []const usize, // Shape of input tensor X
        out_dtype: DType,
    ) usize {
        // ── 1. Create a logical view for the input tensor ─────────────────
        const id_viewX = b.push(.VIEW, out_dtype, &.{X_id}, Any{ .view_meta = .{ .shape = x_shape, .strides = &.{1} } });

        // ── 2. Create output tensor with the same shape as input ──────────
        const id_Y = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = x_shape });

        // ── 3. Create nested loops for each dimension of the tensor ───────
        var nelem: usize = 1;
        for (x_shape) |d| nelem *= d;

        const id_range = b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

        // ── 4. Create GEP operation for current element ───────────────────
        const id_gepX = b.push(.GEP, out_dtype, &.{ id_viewX, id_range }, Any{ .mem_info = .{ .base = id_viewX, .offset = 0, .stride = 1 } });

        // ── 5. Load the input value ─────────────────────────────────────────
        const id_x = b.push(.LOAD, out_dtype, &.{id_gepX}, null);

        // ── 6. Implement Exp: f(x) = exp(x) ────────────────────────────────
        // Since we already have EXP2 in UOps, we'll use that with conversion: exp(x) = 2^(x * log2(e))
        const id_log2_e = b.push(.CONST, out_dtype, &.{}, Any{ .float = 1.4426950408889634 }); // log2(e)
        const id_scaled_x = b.push(.MUL, out_dtype, &.{ id_x, id_log2_e }, null);
        const id_result = b.push(.EXP2, out_dtype, &.{id_scaled_x}, null);

        // ── 7. Store the result to the output tensor ───────────────────────
        const id_gepY = b.push(.GEP, out_dtype, &.{ id_Y, id_range }, Any{ .mem_info = .{ .base = id_Y, .offset = 0, .stride = 1 } });
        _ = b.push(.STORE, out_dtype, &.{ id_gepY, id_result }, null);

        // ── 8. Close all the nested loops ───────────────────────────────────
        _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

        return id_Y; // SSA id of the produced output tensor Y
    }
};
