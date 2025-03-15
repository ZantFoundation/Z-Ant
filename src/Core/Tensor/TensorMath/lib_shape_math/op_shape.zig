const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

/// Implements the ONNX Shape operator (https://onnx.ai/onnx/operators/onnx__Shape.html)
/// Takes a tensor as input and outputs a 1D int64 tensor containing the shape of the input tensor.
/// Optional start and end parameters can be used to compute a slice of the input tensor's shape.
pub fn shape_onnx(comptime T: type, input: *const Tensor(T), start: ?i64, end: ?i64) !Tensor(i64) {
    const rank = input.shape.len;

    // Handle start parameter
    var start_axis: i64 = start orelse 0;
    if (start_axis < 0) start_axis += @as(i64, @intCast(rank));
    start_axis = @max(0, @min(start_axis, @as(i64, @intCast(rank - 1))));

    // Handle end parameter
    var end_axis: i64 = end orelse @as(i64, @intCast(rank));
    if (end_axis < 0) end_axis += @as(i64, @intCast(rank));
    end_axis = @max(start_axis, @min(end_axis, @as(i64, @intCast(rank))));

    // Calculate output size and create output tensor
    const output_size = @max(0, end_axis - start_axis);
    var shape = [_]usize{@intCast(output_size)};
    const initial_data = try pkg_allocator.alloc(i64, output_size);
    defer pkg_allocator.free(initial_data);
    @memset(initial_data, 0);
    var output = try Tensor(i64).fromArray(&pkg_allocator, initial_data, shape[0..]);
    errdefer output.deinit();

    // Copy shape values to output tensor
    var i: usize = 0;
    while (i < output_size) : (i += 1) {
        const idx = @as(usize, @intCast(start_axis)) + i;
        output.data[i] = @intCast(input.shape[idx]);
    }

    return output;
}

/// Lean version of shape_onnx that operates on an existing output tensor
pub fn lean_shape_onnx(comptime InputT: type, comptime OutputT: type, input: *const Tensor(InputT), start: ?i64, end: ?i64, output: *Tensor(OutputT)) !void {
    const rank = input.shape.len;

    // Handle start parameter
    var start_axis: i64 = start orelse 0;
    if (start_axis < 0) start_axis += @as(i64, @intCast(rank));
    start_axis = @max(0, @min(start_axis, @as(i64, @intCast(rank - 1))));

    // Handle end parameter
    var end_axis: i64 = end orelse @as(i64, @intCast(rank));
    if (end_axis < 0) end_axis += @as(i64, @intCast(rank));
    end_axis = @max(start_axis, @min(end_axis, @as(i64, @intCast(rank))));

    // Calculate output size and validate output tensor shape
    const output_size = @max(0, end_axis - start_axis);
    if (output.shape.len != 1 or output.shape[0] != output_size) {
        return TensorError.ShapeMismatch;
    }

    // Copy shape values to output tensor
    var i: usize = 0;
    while (i < output_size) : (i += 1) {
        const idx = @as(usize, @intCast(start_axis)) + i;
        switch (OutputT) {
            f16 => output.data[1] = @floatFromInt(input.shape[idx]),
            f32 => output.data[i] = @floatFromInt(input.shape[idx]),
            f64 => output.data[i] = @floatFromInt(input.shape[idx]),
            i32, i64, u32, u64 => output.data[i] = @intCast(input.shape[idx]),
            else => @compileError("Unsupported output type for shape_onnx"),
        }
    }
}

/// Calculate the output shape for an ONNX Shape operation without performing the operation
pub fn get_shape_output_shape(input_shape: []const usize, start: ?i64, end: ?i64) ![]usize {
    const rank = input_shape.len;

    // Alloca l'output_shape (sempre un tensore 1D)
    var output_shape = try pkg_allocator.alloc(usize, 1);
    errdefer pkg_allocator.free(output_shape);

    // Caso speciale per rank 0 (tensore scalare)
    if (rank == 0) {
        output_shape[0] = 0; // Nessuna dimensione da rappresentare
        return output_shape;
    }

    // Gestione del parametro start
    var start_axis: i64 = start orelse 0;
    if (start_axis < 0) start_axis += @as(i64, @intCast(rank));
    start_axis = @max(0, @min(start_axis, @as(i64, @intCast(rank))));

    // Gestione del parametro end
    var end_axis: i64 = end orelse @as(i64, @intCast(rank));
    if (end_axis < 0) end_axis += @as(i64, @intCast(rank));
    end_axis = @max(start_axis, @min(end_axis, @as(i64, @intCast(rank))));

    // Calcolo della dimensione dell'output
    const output_size = @max(0, end_axis - start_axis);
    output_shape[0] = @intCast(output_size);

    return output_shape;
}
