const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

pub fn identity(comptime T: type, input: *const Tensor(T)) !Tensor(T) {
    var output = try Tensor(T).fromShape(&pkg_allocator, input.shape);
    errdefer output.deinit();

    try identity_lean(T, input, &output);

    return output;
}

pub fn identity_lean(comptime T: anytype, input: *const Tensor(T), output: *const Tensor(T)) !void {
    @memcpy(output.data, input.data);
}

pub fn get_identity_shape_output(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

// https://onnx.ai/onnx/operators/onnx__Identity.html
pub fn lowerIdentity(
    b: *UOpBuilder,
    A_id: usize, // input-tensor SSA ids
    out_shape: []const usize,
    out_dtype: DType, // promoted element type
) usize { // returns id of result buffer

    // ── Set-up phase ────────────────────────────────────────────────────
    _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

    const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = 1 } });

    const id_outBuf = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

    // ── Flat element loop ───────────────────────────────────────────────

    var nelem: usize = 1;
    for (out_shape) |dim| nelem *= dim;

    const id_range = b.push(.RANGE, .u16, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    const id_gepA = b.push(.GEP, out_dtype, &.{ id_viewA, id_range }, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

    const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);

    const id_gepO = b.push(.GEP, out_dtype, &.{ id_outBuf, id_range }, Any{ .mem_info = .{ .base = id_outBuf, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_loadA }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_outBuf;
}
