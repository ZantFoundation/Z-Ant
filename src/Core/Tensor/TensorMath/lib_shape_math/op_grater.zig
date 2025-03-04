const std = @import("std");
const Tensor = @import("tensor").Tensor;
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorError = @import("errorHandler").TensorError;
const TensorMathError = @import("errorHandler").TensorMathError;

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

    // All index values must be within bounds [0, s-1] where s is the lenght of the chosen axis.
    // See note above for details on indices bound values
    for (0..indices.size) |i| {
        if (indices.data[i] >= data.shape[axis] or indices.data[i] < 0) {
            return TensorError.IndexOutOfBounds;
        }
    }

    // Calculate the shape of the output tensor:
    // [data.shape[0..axis], indices.shape..., data.shape[axis+1..]]
    const output_shape_len = data.shape.len - 1 + indices.shape.len;
    const output_shape = try data.allocator.alloc(usize, output_shape_len);
    defer data.allocator.free(output_shape);

    // Copy the dimensions before the axis
    for (0..axis) |i| {
        output_shape[i] = data.shape[i];
    }

    // Copy the indices tensor's shape
    for (0..indices.shape.len) |i| {
        output_shape[axis + i] = indices.shape[i];
    }

    // Copy the dimensions after the axis
    for (0..(data.shape.len - axis - 1)) |i| {
        output_shape[axis + indices.shape.len + i] = data.shape[axis + 1 + i];
    }

    // Create output tensor
    var output = try Tensor(T).fromShape(data.allocator, output_shape);
    errdefer output.deinit();

    try lean_gather(T, data, indices, selected_axis, &output);

    return output;
}

/// Lean version of gather
/// NOTE: (IMPORTANT FOR CODE GEN) according to onnx standard, values in indices tensor can be negative and if so they are converted to positive values by adding the size of the axis pointed dimension of the data tensor. For performance and code clarity reasons (check + double casting) we support only positive indices instead, remove this note and edit "discrepancies from the standard onnx" if this is changed in the future.
pub fn lean_gather(comptime T: anytype, data: *Tensor(T), indices: *Tensor(usize), selected_axis: isize, output: *Tensor(T)) !void {

    //If axis is negative, convert it to a positive index
    const number_dimensions: isize = @intCast(data.shape.len);
    const axis: usize = @intCast(if (selected_axis < 0) number_dimensions + selected_axis else selected_axis);

    // Compute the total number of elements in each segment
    var outer_size: usize = 1;
    for (0..axis) |i| outer_size *= data.shape[i];

    const indices_size: usize = indices.size;

    var inner_size: usize = 1;
    for ((axis + 1)..data.shape.len) |i| {
        std.debug.print("\ninner_size = {d}\ndata.shape[{d}] = {d}", .{ inner_size, i, data.shape[i] });
        inner_size *= data.shape[i];
    }

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

            // Debug Prints (optional, can be commented out after debugging)
            // std.debug.print("Outer Index: {}, Gather Index: {}, Data Offset: {}, Output Offset: {}\n", .{ outer_idx, gather_idx, data_offset, output_offset });
            // std.debug.print("Copying from input data[{}] = {}\n", .{ data_offset, data.data[data_offset] });

            // Perform the data copy using std.mem.copy
            @memcpy(output.data[output_offset .. output_offset + inner_size], data.data[data_offset .. data_offset + inner_size]);

            // std.debug.print("Copied to output data[{}] = {}\n", .{ output_offset, output_data[output_offset] });
        }
    }
}
