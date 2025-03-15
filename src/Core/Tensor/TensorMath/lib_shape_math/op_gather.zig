const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

/// Implements https://onnx.ai/onnx/operators/onnx__Gather.html
/// NOTE: (IMPORTANT FOR CODE GEN) according to onnx standard, values in indices tensor can be negative and if so they are converted to positive values by adding the size of the axis pointed dimension of the data tensor. For performance and code clarity reasons (check + double casting) we support only positive indices instead, remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
/// Gather elements from the data tensor along the specified axis using the provided indices.
/// The axis parameter specifies the axis along which the elements will be gathered.
/// The shape of the output tensor is the same as the shape of the data tensor, with the axis dimension replaced with the shape of the indices tensor.
/// The output tensor is created by copying elements from the input tensor using the indices tensor.
pub fn gather(comptime T: anytype, data: *Tensor(T), indices: *Tensor(usize), selected_axis: isize) !Tensor(T) {
    // Scalar data tensor is not allowed
    if (data.shape.len == 0) {
        return TensorError.InvalidRank;
    }

    // Validate that the axis is within the tensor's dimensions
    const number_dimensions: isize = @intCast(data.shape.len);
    if (selected_axis >= number_dimensions or selected_axis < -1 * number_dimensions) {
        return TensorError.InvalidAxis;
    }

    // If axis is negative, convert it to a positive index
    const axis: usize = @intCast(if (selected_axis < 0) number_dimensions + selected_axis else selected_axis);

    // All index values must be within bounds [0, s-1] where s is the length of the chosen axis
    for (0..indices.size) |i| {
        if (indices.data[i] >= data.shape[axis] or indices.data[i] < 0) {
            return TensorError.IndexOutOfBounds;
        }
    }

    // Calculate the shape of the output tensor:
    // [data.shape[0..axis], indices.shape..., data.shape[axis+1..]]
    const output_shape_len = data.shape.len + indices.shape.len - 1;
    const output_shape = try pkg_allocator.alloc(usize, output_shape_len);
    defer pkg_allocator.free(output_shape);
    errdefer pkg_allocator.free(output_shape);

    // Copy the dimensions before the axis
    for (0..axis) |i| {
        output_shape[i] = data.shape[i];
    }

    // Copy indices shape
    var indices_idx: usize = 0;
    while (indices_idx < indices.shape.len) : (indices_idx += 1) {
        output_shape[axis + indices_idx] = indices.shape[indices_idx];
    }

    // Copy the dimensions after the axis
    for (axis + 1..data.shape.len) |i| {
        output_shape[axis + indices.shape.len + (i - axis - 1)] = data.shape[i];
    }

    // Calculate total size
    var total_size: usize = 1;
    for (output_shape) |dim| {
        total_size *= dim;
    }

    // Create output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    try lean_gather(T, data, indices, selected_axis, &output);

    return output;
}

/// Lean version of gather
/// NOTE: (IMPORTANT FOR CODE GEN) according to onnx standard, values in indices tensor can be negative and if so they are converted to positive values by adding the size of the axis pointed dimension of the data tensor. For performance and code clarity reasons (check + double casting) we support only positive indices instead, remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
pub fn lean_gather(comptime T: anytype, data: *Tensor(T), indices: *Tensor(usize), selected_axis: isize, output: *Tensor(T)) !void {
    //std.debug.print("\n[GATHER] Input shape: {any}", .{data.shape});
    //std.debug.print("\n[GATHER] Input data: {any}", .{data.data});
    //std.debug.print("\n[GATHER] Indices shape: {any}", .{indices.shape});
    //std.debug.print("\n[GATHER] Indices data: {any}", .{indices.data});
    //std.debug.print("\n[GATHER] Selected axis: {d}", .{selected_axis});

    //If axis is negative, convert it to a positive index
    const number_dimensions: isize = @intCast(data.shape.len);
    const axis: usize = @intCast(if (selected_axis < 0) number_dimensions + selected_axis else selected_axis);

    // Compute the total number of elements in each segment
    var outer_size: usize = 1;
    for (0..axis) |i| outer_size *= data.shape[i];

    const indices_size: usize = indices.size;

    var inner_size: usize = 1;
    for ((axis + 1)..data.shape.len) |i| {
        inner_size *= data.shape[i];
    }

    //std.debug.print("[GATHER DEBUG] Outer size: {d}, Inner size: {d}, Indices size: {d}\n", .{ outer_size, inner_size, indices_size });

    // Iterate over each "outer" segment
    for (0..outer_size) |outer_idx| {
        // Iterate over each index in the indices tensor
        for (0..indices_size) |idx| {
            // Retrieve the gather index from the indices tensor
            const gather_idx = try indices.get(idx);

            // Calculate the correct data_offset
            const data_offset = (outer_idx * data.shape[axis] + gather_idx) * inner_size;

            // Calculate the starting offset in the output tensor
            const output_offset = (outer_idx * indices_size + idx) * inner_size;

            //std.debug.print("[GATHER DEBUG] Outer idx: {d}, Gather idx: {d}, Data offset: {d}, Output offset: {d}\n", .{ outer_idx, gather_idx, data_offset, output_offset });

            // Perform the data copy using std.mem.copy
            @memcpy(output.data[output_offset .. output_offset + inner_size], data.data[data_offset .. data_offset + inner_size]);

            //std.debug.print("[GATHER DEBUG] Copied data: ", .{});
            //for (data.data[data_offset .. data_offset + inner_size]) |val| {
            //std.debug.print("{d} ", .{val});
            // }
            //std.debug.print("\n", .{});
        }
    }
    //std.debug.print("\n[GATHER] Output shape: {any}", .{output.shape});
    //std.debug.print("\n[GATHER] Output data: {any}\n", .{output.data});
}

pub fn get_gather_output_shape(input_shape: []const usize, indices_shape: []const usize, selected_axis: isize) ![]usize {
    // Scalar data tensor is not allowed
    if (input_shape.len == 0) {
        return TensorError.InvalidRank;
    }

    // Validate that the axis is within the tensor's dimensions
    const number_dimensions: isize = @intCast(input_shape.len);
    if (selected_axis >= number_dimensions or selected_axis < -1 * number_dimensions) {
        return TensorError.InvalidAxis;
    }

    // If axis is negative, convert it to a positive index
    const axis: usize = @intCast(if (selected_axis < 0) number_dimensions + selected_axis else selected_axis);

    // Calculate the shape of the output tensor:
    // [input_shape[0..axis], indices_shape..., input_shape[axis+1..]]
    const output_shape_len = input_shape.len + indices_shape.len - 1;
    const output_shape = try pkg_allocator.alloc(usize, output_shape_len);
    errdefer pkg_allocator.free(output_shape);

    // Copy the dimensions before the axis
    for (0..axis) |i| {
        output_shape[i] = input_shape[i];
    }

    // Copy indices shape
    var indices_idx: usize = 0;
    while (indices_idx < indices_shape.len) : (indices_idx += 1) {
        output_shape[axis + indices_idx] = indices_shape[indices_idx];
    }

    // Copy the dimensions after the axis
    for (axis + 1..input_shape.len) |i| {
        output_shape[axis + indices_shape.len + (i - axis - 1)] = input_shape[i];
    }

    return output_shape;
}
