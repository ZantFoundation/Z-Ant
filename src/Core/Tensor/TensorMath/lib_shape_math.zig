//! These operations change the structure or organization of a tensor.
//!
//!    Reshape: Change the shape without changing the data.
//!    Expand/Squeeze: like Padding and dilatation.
//!    Transpose/Permute: Change the order of dimensions.
//!    Flatten: Convert a multi-dimensional tensor into 1D.
//!    Concatenation: Combine tensors along a specified dimension.
//!    Split: Divide a tensor into smaller tensors.
//!    Neg: used to flip the kernel in some convolution operations.
//!    Identity: returns the input tensor

const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

/// Resize the input tensor using interpolation.
/// Supports 'nearest', 'linear', and 'cubic' interpolation modes.
pub fn resize(comptime T: type, t: *Tensor(T), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8) !Tensor(T) {
    //check if mode exixts:
    if (!(std.mem.eql(u8, mode, "nearest") or std.mem.eql(u8, mode, "linear") or std.mem.eql(u8, mode, "cubic"))) {
        return TensorError.UnsupportedMode;
    }

    //check args: there should be one and only one between scales and sizes
    if (scales == null and sizes == null) {
        return TensorError.InvalidInput;
    }
    if (scales != null and sizes != null) {
        return TensorError.InvalidInput;
    }

    // Create output tensor
    var output = try Tensor(T).init(t.allocator);

    //call rezise_lean
    if (scales) |s| {
        if (s.len != t.shape.len) {
            return TensorError.InvalidInput;
        } else {
            try rezise_lean(T, t, mode, scales, null, coordinate_transformation_mode, &output);
        }
    } else if (sizes) |sz| {
        if (sz.len != t.shape.len) {
            return TensorError.InvalidInput;
        } else {
            try rezise_lean(T, t, mode, null, sizes, coordinate_transformation_mode, &output);
        }
    }

    return output;
}

//resize lean
pub fn rezise_lean(comptime T: type, t: *Tensor(T), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8, output_tensor: *Tensor(T)) !void {

    // Calculate output dimensions
    var output_shape = try t.allocator.alloc(usize, t.shape.len);
    errdefer t.allocator.free(output_shape);

    if (scales) |s| {
        if (s.len != t.shape.len) {
            return TensorError.InvalidInput;
        }
        for (0..t.shape.len) |i| {
            output_shape[i] = @intFromFloat(@floor(@as(f32, @floatFromInt(t.shape[i])) * s[i]));
        }
    } else if (sizes) |sz| {
        if (sz.len != t.shape.len) {
            return TensorError.InvalidInput;
        }
        @memcpy(output_shape, sz);
    }

    // Calculate total size of output tensor
    var total_size: usize = 1;
    for (output_shape) |dim| {
        total_size *= dim;
    }

    // Allocate memory for output data
    const output_data = try t.allocator.alloc(T, total_size);
    errdefer t.allocator.free(output_data);

    // Perform interpolation based on mode
    if (std.mem.eql(u8, mode, "nearest")) {
        try nearest_interpolation(T, t, output_data, output_shape, coordinate_transformation_mode);
    } else if (std.mem.eql(u8, mode, "linear")) {
        try linear_interpolation(T, t, output_data, output_shape, coordinate_transformation_mode);
    } else { //cubic interpolation
        try cubic_interpolation(T, t, output_data, output_shape, coordinate_transformation_mode);
    }

    //fill output tensor
    output_tensor.data = output_data;
    output_tensor.shape = output_shape;
    output_tensor.size = total_size;
    output_tensor.allocator = t.allocator;
}

pub fn get_resize_output_shape(input_shape: []const usize, scales: ?[]const f32, sizes: ?[]const usize) ![]usize {
    if (scales == null and sizes == null) {
        return TensorError.InvalidInput;
    }
    if (scales != null and sizes != null) {
        return TensorError.InvalidInput;
    }

    var output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    if (scales) |s| {
        if (s.len != input_shape.len) {
            return TensorError.InvalidInput;
        }
        for (0..input_shape.len) |i| {
            output_shape[i] = @intFromFloat(@floor(@as(f32, @floatFromInt(input_shape[i])) * s[i]));
        }
    } else if (sizes) |sz| {
        if (sz.len != input_shape.len) {
            return TensorError.InvalidInput;
        }
        @memcpy(output_shape, sz);
    }

    return output_shape;
}

fn nearest_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    const input_strides = try self.getStrides();
    defer self.allocator.free(input_strides);
    const output_strides = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_strides);

    // Calculate output strides
    var stride: usize = 1;
    var idx: usize = output_shape.len;
    while (idx > 0) {
        idx -= 1;
        output_strides[idx] = stride;
        stride *= output_shape[idx];
    }

    var output_indices = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_indices);
    @memset(output_indices, 0);

    var done = false;
    while (!done) {
        var output_idx: usize = 0;
        var input_idx: usize = 0;

        for (0..output_shape.len) |i| {
            const scale = @as(f32, @floatFromInt(output_shape[i])) / @as(f32, @floatFromInt(self.shape[i]));
            var input_pos: f32 = undefined;

            if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                input_pos = (@as(f32, @floatFromInt(output_indices[i])) + 0.5) / scale - 0.5;
            } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                input_pos = @as(f32, @floatFromInt(output_indices[i])) * @as(f32, @floatFromInt(self.shape[i] - 1)) / @as(f32, @floatFromInt(output_shape[i] - 1));
            } else { // asymmetric
                input_pos = @as(f32, @floatFromInt(output_indices[i])) / scale;
            }

            const input_idx_i = @as(i32, @intFromFloat(@round(input_pos)));
            const clamped_idx = @min(@max(input_idx_i, 0), @as(i32, @intCast(self.shape[i] - 1)));
            input_idx += @as(usize, @intCast(clamped_idx)) * input_strides[i];
            output_idx += output_indices[i] * output_strides[i];
        }

        output_data[output_idx] = self.data[input_idx];

        // Increment indices
        done = true;
        for (0..output_shape.len) |i| {
            output_indices[output_shape.len - 1 - i] += 1;
            if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                done = false;
                break;
            }
            output_indices[output_shape.len - 1 - i] = 0;
        }
    }
}

fn linear_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    // For now, implement only for 1D and 2D tensors
    if (self.shape.len > 2) return TensorError.UnsupportedDimension;

    const input_strides = try self.getStrides();
    defer self.allocator.free(input_strides);

    var output_indices = try self.allocator.alloc(usize, output_shape.len);
    defer self.allocator.free(output_indices);
    @memset(output_indices, 0);

    var done = false;
    while (!done) {
        var output_idx: usize = 0;
        if (output_shape.len == 1) {
            output_idx = output_indices[0];
        } else {
            output_idx = output_indices[0] * output_shape[1] + output_indices[1];
        }

        // Calculate interpolation coordinates
        var x: f32 = undefined;
        if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
            x = (@as(f32, @floatFromInt(output_indices[0])) + 0.5) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0])) - 0.5;
        } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
            x = @as(f32, @floatFromInt(output_indices[0])) * @as(f32, @floatFromInt(self.shape[0] - 1)) / @as(f32, @floatFromInt(output_shape[0] - 1));
        } else { // asymmetric
            x = @as(f32, @floatFromInt(output_indices[0])) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0]));
        }

        const x_floor = @floor(x);
        const x0 = @as(usize, @intFromFloat(@max(0, x_floor)));
        const x1 = @min(x0 + 1, self.shape[0] - 1);
        const dx = x - x_floor;

        if (self.shape.len == 1) {
            const v0 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0]))));
            const v1 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1]))));
            const interpolated = v0 * (1 - dx) + v1 * dx;
            output_data[output_idx] = @as(T, @intFromFloat(@round(interpolated)));
        } else {
            var y: f32 = undefined;
            if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                y = (@as(f32, @floatFromInt(output_indices[1])) + 0.5) * @as(f32, @floatFromInt(self.shape[1])) / @as(f32, @floatFromInt(output_shape[1])) - 0.5;
            } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                y = @as(f32, @floatFromInt(output_indices[1])) * @as(f32, @floatFromInt(self.shape[1] - 1)) / @as(f32, @floatFromInt(output_shape[1] - 1));
            } else { // asymmetric
                y = @as(f32, @floatFromInt(output_indices[1])) * @as(f32, @floatFromInt(self.shape[1])) / @as(f32, @floatFromInt(output_shape[1]));
            }

            const y_floor = @floor(y);
            const y0 = @as(usize, @intFromFloat(@max(0, y_floor)));
            const y1 = @min(y0 + 1, self.shape[1] - 1);
            const dy = y - y_floor;

            const v00 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0 * self.shape[1] + y0]))));
            const v01 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0 * self.shape[1] + y1]))));
            const v10 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1 * self.shape[1] + y0]))));
            const v11 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1 * self.shape[1] + y1]))));

            const tmp1 = v00 * (1 - dx) * (1 - dy);
            const tmp2 = v01 * (1 - dx) * dy;
            const tmp3 = v10 * dx * (1 - dy);
            const tmp4 = v11 * dx * dy;

            const interpolated = tmp1 + tmp2 + tmp3 + tmp4;
            output_data[output_idx] = @as(T, @intFromFloat(@round(interpolated)));
        }

        // Increment indices
        done = true;
        for (0..output_shape.len) |i| {
            output_indices[output_shape.len - 1 - i] += 1;
            if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                done = false;
                break;
            }
            output_indices[output_shape.len - 1 - i] = 0;
        }
    }
}

fn cubic_interpolation(comptime T: type, self: *Tensor(T), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
    // For simplicity, implement only for 1D tensors initially
    if (self.shape.len != 1) return TensorError.UnsupportedDimension;

    var output_idx: usize = 0;
    while (output_idx < output_shape[0]) : (output_idx += 1) {
        var x: f32 = undefined;
        if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
            x = (@as(f32, @floatFromInt(output_idx)) + 0.5) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0])) - 0.5;
        } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
            x = @as(f32, @floatFromInt(output_idx)) * @as(f32, @floatFromInt(self.shape[0] - 1)) / @as(f32, @floatFromInt(output_shape[0] - 1));
        } else { // asymmetric
            x = @as(f32, @floatFromInt(output_idx)) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0]));
        }

        const x0 = @as(i32, @intFromFloat(@floor(x)));
        const dx = x - @as(f32, @floatFromInt(x0));

        var sum: f32 = 0;
        var weight_sum: f32 = 0;

        var i: i32 = -1;
        while (i < 3) : (i += 1) {
            const idx = x0 + i;
            if (idx >= 0 and idx < @as(i32, @intCast(self.shape[0]))) {
                const w = cubic_weight(dx - @as(f32, @floatFromInt(i)));
                sum += @as(f32, @floatFromInt(@as(i32, @intCast(self.data[@as(usize, @intCast(idx))])))) * w;
                weight_sum += w;
            }
        }

        output_data[output_idx] = @as(T, @intFromFloat(@round(sum / weight_sum)));
    }
}

fn cubic_weight(x: f32) f32 {
    const a = -0.75;
    const abs_x = @abs(x);
    if (abs_x <= 1) {
        return ((a + 2) * abs_x - (a + 3)) * abs_x * abs_x + 1;
    } else if (abs_x < 2) {
        return ((a * abs_x - 5 * a) * abs_x + 8 * a) * abs_x - 4 * a;
    }
    return 0;
}

/// Performs the core concatenation operation without validation checks.
/// This is a lean version of the concatenate function that assumes all inputs have been validated.
///
/// Parameters:
///     allocator - The memory allocator to use for the new tensor.
///     tensors - An array of tensors to concatenate.
///     axis - The axis along which to concatenate.
///     output - The pre-allocated output tensor to store the result.
///
/// Returns:
///     void
///
/// Errors:
///     - TensorError.IndexOutOfBounds
///     - TensorError.MismatchedRank
///     - TensorError.MismatchedShape
pub fn lean_concatenate(comptime T: type, allocator: *const std.mem.Allocator, tensors: []const Tensor(T), axis: isize, output: *Tensor(T)) !void {
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].shape.len;

    // Find the maximum rank among all tensors
    var max_rank: usize = rank;
    var need_reshape = false;

    for (tensors) |tensor| {
        if (tensor.shape.len != rank) {
            need_reshape = true;
            max_rank = @max(max_rank, tensor.shape.len);
        }
    }

    // Create a working copy of the tensors that we might modify
    var modified_tensors = try allocator.alloc(Tensor(T), tensors.len);
    defer {
        // Clean up any reshaped tensors we created
        if (need_reshape) {
            for (modified_tensors) |*tensor| {
                if (tensor.owns_memory) {
                    tensor.deinit();
                }
            }
        }
        allocator.free(modified_tensors);
    }

    // Initially, just copy the references
    for (tensors, 0..) |tensor, i| {
        modified_tensors[i] = tensor;
    }

    // Handle reshaping if needed
    if (need_reshape) {
        // Reshape tensors with lower rank to match the maximum rank
        for (tensors, 0..) |tensor, i| {
            if (tensor.shape.len < max_rank) {
                // Create a new shape with added dimensions
                var new_shape = try allocator.alloc(usize, max_rank);
                defer allocator.free(new_shape);

                // Fill with 1s first
                @memset(new_shape, 1);

                // Copy original dimensions
                const offset = max_rank - tensor.shape.len;
                for (tensor.shape, 0..) |dim, j| {
                    new_shape[offset + j] = dim;
                }

                // Create a new tensor with the reshaped dimensions
                var reshaped_tensor = try Tensor(T).init(allocator);
                errdefer reshaped_tensor.deinit();
                try reshaped_tensor.fill(tensor.data, new_shape);
                reshaped_tensor.owns_memory = true;

                // Replace the original tensor in our working copy
                modified_tensors[i] = reshaped_tensor;
            }
        }
    }

    // Update working rank to the potentially new maximum rank
    const working_rank = max_rank;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(working_rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(working_rank))) {
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Validate that all tensors have matching shapes except along the concatenation axis
    for (modified_tensors) |tensor| {
        for (0..working_rank) |d| {
            if (d != concat_axis_usize and tensor.shape[d] != modified_tensors[0].shape[d]) {
                return TensorError.MismatchedShape;
            }
        }
    }

    // Calculate the number of slices based on the concatenation axis
    var num_slices: usize = 1;
    for (0..concat_axis_usize) |d| {
        num_slices *= output.shape[d];
    }

    // Calculate the slice size (number of elements to copy per concatenation dimension)
    var slice_size: usize = 1;
    if (concat_axis_usize + 1 < working_rank) {
        for ((concat_axis_usize + 1)..working_rank) |d| {
            slice_size *= output.shape[d];
        }
    } else {
        slice_size = 1;
    }

    // Initialize the offset for copying data into output
    var offset: usize = 0;

    // Iterate over each slice
    for (0..num_slices) |slice_idx| {
        for (modified_tensors) |tensor| {
            const concat_dim = tensor.shape[concat_axis_usize];
            const copy_size = concat_dim * slice_size;

            // Calculate the start and end indices in the source tensor
            const src_start = slice_idx * concat_dim * slice_size;
            const src_end = src_start + copy_size;

            // Check bounds for the source tensor's data
            if (src_end > tensor.data.len) {
                return TensorError.IndexOutOfBounds;
            }

            // Calculate the destination indices in output data
            const dest_start = offset;
            const dest_end = offset + copy_size;

            // Check bounds for the output buffer
            if (dest_end > output.data.len) {
                return TensorError.IndexOutOfBounds;
            }

            @memcpy(output.data[dest_start..dest_end], tensor.data[src_start .. src_start + copy_size]);

            // Update the offset for the next copy
            offset += copy_size;
        }
    }
}

/// Concatenates a list of tensors into a single tensor along the specified axis.
/// All input tensors must have the same shape, except for the size of the concatenation axis.
///
/// Parameters:
///     allocator - The memory allocator to use for the new tensor.
///     tensors - An array of tensors to concatenate.
///     axis - The axis along which to concatenate. Negative values count dimensions from the back.
///
/// Returns:
///     A new tensor resulting from concatenation.
///
/// Errors:
///     - TensorError.EmptyTensorList
///     - TensorError.AxisOutOfBounds
///     - TensorError.MismatchedRank
///     - TensorError.MismatchedShape
pub fn concatenate(comptime T: type, allocator: *const std.mem.Allocator, tensors: []const Tensor(T), axis: isize) !Tensor(T) {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].shape.len;

    // Find the maximum rank among all tensors
    var max_rank: usize = rank;
    var need_reshape = false;

    for (tensors) |tensor| {
        if (tensor.shape.len != rank) {
            need_reshape = true;
            max_rank = @max(max_rank, tensor.shape.len);
        }
    }

    // Update working rank to the potentially new maximum rank
    const working_rank = max_rank;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(working_rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(working_rank))) {
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Calculate the new shape after concatenation
    var new_shape: []usize = undefined;
    var output_data: []T = undefined;
    var shape_allocated = false;
    var data_allocated = false;

    // Allocate the shape
    new_shape = allocator.alloc(usize, working_rank) catch |err| {
        return err;
    };
    shape_allocated = true;
    errdefer {
        if (shape_allocated) {
            allocator.free(new_shape);
        }
    }

    // Initialize with the shape of the first tensor (potentially reshaped)
    if (tensors[0].shape.len < working_rank) {
        // Fill with 1s first
        @memset(new_shape, 1);

        // Copy original dimensions
        const offset = working_rank - tensors[0].shape.len;
        for (tensors[0].shape, 0..) |dim, j| {
            new_shape[offset + j] = dim;
        }
    } else {
        for (0..working_rank) |d| {
            new_shape[d] = tensors[0].shape[d];
        }
    }

    // Calculate the sum along the concatenation axis
    var sum: usize = 0;
    for (tensors) |tensor| {
        if (tensor.shape.len < working_rank) {
            // For tensors with lower rank, we need to calculate the effective dimension
            const offset = working_rank - tensor.shape.len;
            if (concat_axis_usize >= offset) {
                sum += tensor.shape[concat_axis_usize - offset];
            } else {
                sum += 1; // Implicit dimension size is 1
            }
        } else {
            sum += tensor.shape[concat_axis_usize];
        }
    }
    new_shape[concat_axis_usize] = sum;

    // Calculate the total number of elements in the new tensor
    var total_size: usize = 1;
    for (new_shape) |dim| {
        total_size *= dim;
    }

    // Allocate memory for the output tensor's data
    output_data = allocator.alloc(T, total_size) catch |err| {
        allocator.free(new_shape);
        return err;
    };
    data_allocated = true;
    errdefer {
        if (data_allocated) {
            allocator.free(output_data);
        }
    }

    // Create the output tensor
    var output_tensor = Tensor(T){
        .data = output_data,
        .size = total_size,
        .shape = new_shape,
        .allocator = allocator,
        .owns_memory = true,
    };

    // Use the lean version to perform the actual concatenation
    lean_concatenate(T, allocator, tensors, axis, &output_tensor) catch |err| {
        // Since output_tensor owns the memory, we don't need to manually free it
        // The caller will handle the error and the memory will be properly freed
        return err;
    };

    return output_tensor;
}

pub fn get_concatenate_output_shape(tensors: []const []const usize, axis: isize) ![]usize {
    // Ensure there is at least one tensor to concatenate
    if (tensors.len == 0) return TensorMathError.EmptyTensorList;

    // Determine the rank (number of dimensions) from the first tensor
    const rank = tensors[0].len;

    var concat_axis = axis;
    if (concat_axis < 0) {
        concat_axis += @as(isize, @intCast(rank));
    }

    if (concat_axis < 0 or concat_axis >= @as(isize, @intCast(rank))) {
        return TensorError.AxisOutOfBounds;
    }

    const concat_axis_usize = @as(usize, @intCast(concat_axis));

    // Validate that all tensors have the same rank and matching shapes except along the concatenation axis
    for (tensors) |tensor_shape| {
        if (tensor_shape.len != rank) {
            return TensorError.MismatchedRank;
        }
        for (0..rank) |d| {
            if (d != concat_axis_usize and tensor_shape[d] != tensors[0][d]) {
                return TensorError.MismatchedShape;
            }
        }
    }

    // Calculate the new shape after concatenation
    var new_shape = try pkg_allocator.alloc(usize, rank);
    errdefer pkg_allocator.free(new_shape);

    for (0..rank) |d| {
        if (d == concat_axis_usize) {
            var sum: usize = 0;
            for (tensors) |tensor_shape| {
                sum += tensor_shape[d];
            }
            new_shape[d] = sum;
        } else {
            new_shape[d] = tensors[0][d];
        }
    }

    return new_shape;
}

/// Returns a Tensor self transposed. Does not modify self.
/// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
pub fn transpose2D(comptime T: type, t: *Tensor(T)) !Tensor(T) {
    if (t.shape.len != 2) {
        return error.InvalidDimension; // For simplicity, let's focus on 2D for now
    }

    const allocator = t.allocator;

    // Shape of the transposed tensor
    const transposed_shape: [2]usize = [_]usize{ t.shape[1], t.shape[0] };
    const tensorShape = try allocator.alloc(usize, t.shape.len);
    @memcpy(tensorShape, &transposed_shape);

    // Allocate space for transposed data
    const transposed_data = try allocator.alloc(T, t.size);

    // Perform the transposition
    for (0..t.shape[0]) |i| {
        for (0..t.shape[1]) |j| {
            // For 2D tensor, flatten the index and swap row/column positions
            const old_idx = i * t.shape[1] + j;
            const new_idx = j * t.shape[0] + i;
            transposed_data[new_idx] = t.data[old_idx];
        }
    }

    return Tensor(T){
        .data = transposed_data,
        .size = t.size,
        .shape = tensorShape,
        .allocator = allocator,
        .owns_memory = true,
    };
}

/// Returns a Tensor self transposed.
/// OSS! Does not modify self.data!! it only changes the shape! so it is necessary to acces it trough get_flatten_index()
/// By default, it transposes the tensor to the reverse shape.
pub fn transposeDefault(comptime T: type, t: *Tensor(T)) !Tensor(T) {
    // Reverse the shape of the tensor
    const tensorShape = try t.allocator.alloc(usize, t.shape.len);
    for (0..t.shape.len) |i| {
        tensorShape[i] = t.shape.len - 1 - i;
    }

    return transpose(T, t, tensorShape);
}

/// Returns a Tensor self transposed. Does not modify self.
fn transpose(comptime T: type, t: *Tensor(T), perms: []usize) !Tensor(T) {
    defer t.allocator.free(perms);
    const num_dims = t.shape.len;
    if (perms.len != num_dims) {
        return error.InvalidDimension;
    }

    // Check that the permutation is valid
    var bitmap = try t.allocator.alloc(bool, perms.len);
    defer t.allocator.free(bitmap);

    for (perms) |perm| {
        if (perm >= perms.len) {
            return error.InvalidPermutation;
        }
        if (bitmap[perm] == true) {
            return error.InvalidPermutation;
        }
        bitmap[perm] = true;
    }

    // Allocate space for the new shape
    const new_shape = try t.allocator.alloc(usize, num_dims);
    for (0..num_dims) |i| {
        new_shape[i] = t.shape[perms[i]];
    }
    defer t.allocator.free(new_shape);

    // Create the new tensor
    const new_tensor = try Tensor(T).fromShape(t.allocator, new_shape);

    // Copy data to the new tensor
    for (0..t.size) |i| {
        new_tensor.data[i] = t.data[i];
    }

    return new_tensor;
}

/// Given a 4D tensor it returns the tensor with the last 2 dimensions transposed. Operates on both data and shape, does not modify self, used by gemm.
pub fn transposeLastTwo(comptime T: anytype, tensor: *const Tensor(T)) !Tensor(T) {
    //std.debug.print("\n[DEBUG] transposeLastTwo:", .{});
    //std.debug.print("\n  Input tensor shape: ", .{});
    //for (tensor.shape) |s| std.debug.print("{d} ", .{s});

    // Verifying correct shape
    if (tensor.shape.len != 2 and tensor.shape.len != 4) {
        //std.debug.print("\n  Error: Expected 2D or 4D tensor, got {d}D", .{tensor.shape.len});
        return TensorMathError.InputTensorsWrongShape;
    }

    var rows: usize = undefined;
    var cols: usize = undefined;
    var total: usize = undefined;
    var newShape: []usize = undefined;

    if (tensor.shape.len == 2) {
        rows = tensor.shape[0];
        cols = tensor.shape[1];
        total = rows * cols;
        newShape = try pkg_allocator.alloc(usize, 2);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = cols;
        newShape[1] = rows;
    } else { // 4D case
        const batch = tensor.shape[0];
        const channel = tensor.shape[1];
        rows = tensor.shape[2];
        cols = tensor.shape[3];
        total = batch * channel * rows * cols;
        newShape = try pkg_allocator.alloc(usize, 4);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = batch;
        newShape[1] = channel;
        newShape[2] = cols;
        newShape[3] = rows;
    }

    //std.debug.print("\n  Rows: {d}, Cols: {d}, Total: {d}", .{ rows, cols, total });
    //std.debug.print("\n  New shape: ", .{});
    //for (newShape) |s| std.debug.print("{d} ", .{s});

    // Create a non-const copy of the input data using pkg_allocator
    const outData = try pkg_allocator.alloc(T, total);
    errdefer pkg_allocator.free(outData);

    //std.debug.print("\n  Transposing data...", .{});

    if (tensor.shape.len == 2) {
        // Simple 2D transpose - Fixed indexing
        for (0..rows) |i| {
            for (0..cols) |j| {
                outData[j * rows + i] = tensor.data[i * cols + j];
            }
        }
    } else {
        // 4D transpose of last two dimensions
        const batch = tensor.shape[0];
        const channel = tensor.shape[1];
        for (0..batch) |b| {
            for (0..channel) |c| {
                for (0..rows) |i| {
                    for (0..cols) |j| {
                        const index_in = (((b * channel) + c) * rows + i) * cols + j;
                        const index_out = (((b * channel) + c) * cols + j) * rows + i;
                        outData[index_out] = tensor.data[index_in];
                    }
                }
            }
        }
    }

    //std.debug.print("\n  Transpose complete", .{});

    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = &pkg_allocator,
        .owns_memory = true,
    };
}

/// Method to add a top&bottom padding and a left&right padding.
/// At the moment the function only supports 2 padding params, but the method
/// is already set to have different left, right, top and bottom padding values.
pub fn addPaddingAndDilation(
    comptime T: type,
    t: *Tensor(T),
    upDownPadding: usize,
    leftRightPadding: usize,
    verticalDil: usize,
    horizontalDil: usize,
) !void {

    //checks on padding dim (usize is alway >= 0)
    if (t.shape.len < 2) return TensorError.TooSmallToPadding;

    const upPadding = upDownPadding;
    const downPadding = upDownPadding;
    const leftPadding = leftRightPadding;
    const rightPadding = leftRightPadding;
    const dim = t.shape.len;

    const new_row_numb = t.shape[dim - 2] + upPadding + downPadding + verticalDil * (t.shape[dim - 2] - 1);
    const new_col_numb = t.shape[dim - 1] + leftPadding + rightPadding + horizontalDil * (t.shape[dim - 1] - 1);
    //std.debug.print("\n new_row_numb: {} new_col_numb:{}", .{ new_row_numb, new_col_numb });

    //compute new shape
    const new_shape = try t.allocator.alloc(usize, dim);
    @memcpy(new_shape, t.shape);
    new_shape[dim - 1] = new_col_numb;
    new_shape[dim - 2] = new_row_numb;

    //compute new size
    var new_total_size: usize = 1;
    for (new_shape) |size_i| {
        new_total_size *= size_i;
    }

    //alloc new tensor.data memory space to all zero
    const new_data = try t.allocator.alloc(T, new_total_size);
    @memset(new_data, 0);

    const new_matrix_dim = new_row_numb * new_col_numb;
    const total_number_2DMatrices = new_total_size / new_matrix_dim;
    const old_matrix_dim = t.shape[dim - 2] * t.shape[dim - 1];
    const old_total_number_2DMatrices = t.size / old_matrix_dim; //just for check assertion
    std.debug.assert(total_number_2DMatrices == old_total_number_2DMatrices);

    for (0..total_number_2DMatrices) |matix_i| {
        const num_elem_prec_new_matr = matix_i * new_matrix_dim;
        const num_elem_prec_old_matr = matix_i * old_matrix_dim;
        var i = upPadding;
        var old_row: usize = 0;
        while (i < new_row_numb - downPadding) : (i += (1 + verticalDil)) {
            var j = leftPadding;
            var old_col: usize = 0;
            while (j < new_col_numb - rightPadding) : (j += (1 + horizontalDil)) {
                const idx_new_matr = num_elem_prec_new_matr + i * new_col_numb + j;
                const idx_old_matr = num_elem_prec_old_matr + old_row * (t.shape[dim - 1]) + old_col;
                new_data[idx_new_matr] = t.data[idx_old_matr];
                old_col += 1;
            }
            old_row += 1;
        }
    }

    //free all old attributes and setting new ones
    t.allocator.free(t.data);
    t.allocator.free(t.shape);

    t.shape = new_shape;
    t.data = new_data;
    t.size = new_total_size;
}

/// Helper function to flip (rotate 180 degrees horizontaly and vertically) the kernel in convolution or any other matix 2D
/// ex:
///  neg( [[a, b], [c, d], [e, f]] ) = [[f, e], [d, c], [b, a]]
pub fn neg(comptime T: type, kernel: *Tensor(T)) !Tensor(T) {
    //create and initialize the new shape
    const flipped_shape = try kernel.allocator.alloc(usize, kernel.shape.len);
    defer kernel.allocator.free(flipped_shape);
    @memcpy(flipped_shape, kernel.shape);

    var flipped_kernel = try Tensor(T).fromShape(kernel.allocator, flipped_shape);

    try neg_lean(T, kernel, &flipped_kernel);
    return flipped_kernel;
}

pub fn neg_lean(comptime T: type, kernel: *Tensor(T), output_kernel: *Tensor(T)) !void {
    const kernel_dim = kernel.shape.len;
    const kernel_row = kernel.shape[kernel_dim - 2];
    const kernel_cols = kernel.shape[kernel_dim - 1];
    const matrix_dim = kernel_cols * kernel_row;

    const total_number_2DMatrices = output_kernel.size / matrix_dim;

    for (0..total_number_2DMatrices) |matix_i| {
        for (0..kernel_row) |i| {
            for (0..kernel_cols) |j| {
                output_kernel.data[(matix_i + 1) * matrix_dim - (i * kernel_cols + j + 1)] = kernel.data[matix_i * matrix_dim + i * kernel_cols + j];
            }
        }
    }
}

pub fn get_neg_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    @memcpy(output_shape, input_shape);
    return output_shape;
}

/// Split a tensor into multiple tensors along a specified axis.
/// If split_sizes is null, the tensor is split into equal parts.
/// If split_sizes is provided, it specifies the size of each split.
/// Negative axis values count from the back (-1 means last axis).
/// Returns an array of tensors that must be freed by the caller.
pub fn split(comptime T: anytype, t: *Tensor(T), axis: i64, split_sizes: ?[]const usize) ![]Tensor(T) {
    // Handle negative axis
    const positive_axis = @as(usize, @intCast(if (axis < 0) @as(i64, @intCast(t.shape.len)) + axis else axis));
    if (positive_axis >= t.shape.len) return TensorError.InvalidAxis;

    // Calculate split sizes
    const dim_size = t.shape[positive_axis];
    var sizes = std.ArrayList(usize).init(t.allocator.*);
    defer sizes.deinit();

    if (split_sizes) |s| {
        // Validate and use provided split sizes
        var total_size: usize = 0;
        for (s) |size| {
            try sizes.append(size);
            total_size += size;
        }
        if (total_size != dim_size) return TensorError.InvalidSplitSize;
    } else {
        // Split into equal parts
        if (dim_size == 0) return TensorError.InvalidSplitSize;
        const split_size = dim_size;
        try sizes.append(split_size);
    }

    // Create output tensors
    var output_tensors = try t.allocator.alloc(Tensor(T), sizes.items.len);
    errdefer {
        for (output_tensors) |*tensor| {
            tensor.deinit();
        }
        t.allocator.free(output_tensors);
    }

    const split_size = sizes.items;
    try split_lean(T, t, axis, split_size, &output_tensors);

    return output_tensors;
}

//lean split
//inputs:
//split_sizes can't be null
pub fn split_lean(comptime T: anytype, t: *Tensor(T), axis: i64, split_sizes: ?[]const usize, output_tensors: *[]Tensor(T)) !void {
    const positive_axis = @as(usize, @intCast(if (axis < 0) @as(i64, @intCast(t.shape.len)) + axis else axis));
    const dim_size = t.shape[positive_axis];
    const sizes = split_sizes.?;

    var offset: usize = 0;
    for (sizes, 0..) |split_size, i| {
        // Create shape for the split tensor
        var new_shape = try t.allocator.alloc(usize, t.shape.len);
        errdefer t.allocator.free(new_shape);
        @memcpy(new_shape, t.shape);
        new_shape[positive_axis] = split_size;

        // Calculate total size for the split tensor
        var total_size: usize = 1;
        for (new_shape) |dim| {
            total_size *= dim;
        }

        // Allocate memory for the split tensor's data
        var new_data = try t.allocator.alloc(T, total_size);
        errdefer t.allocator.free(new_data);

        // Calculate strides
        var stride: usize = 1;
        for (positive_axis + 1..t.shape.len) |j| {
            stride *= t.shape[j];
        }

        // Copy data to the split tensor
        const block_size = split_size * stride;
        const num_blocks = total_size / block_size;

        var block_idx: usize = 0;
        while (block_idx < num_blocks) : (block_idx += 1) {
            const src_start = offset + block_idx * dim_size * stride;
            const dst_start = block_idx * split_size * stride;
            const copy_size = split_size * stride;
            @memcpy(new_data[dst_start .. dst_start + copy_size], t.data[src_start .. src_start + copy_size]);
        }

        // Create the split tensor
        output_tensors.*[i].data = new_data;
        output_tensors.*[i].size = total_size;
        output_tensors.*[i].shape = new_shape;
        output_tensors.*[i].allocator = t.allocator;
        output_tensors.*[i].owns_memory = true;

        offset += split_size * stride;
    }
}

pub fn get_split_output_shapes(input_shape: []const usize, axis: i64, split_sizes: ?[]const usize) ![][]usize {
    // Handle negative axis
    var positive_axis: usize = undefined;
    if (axis < 0) {
        const adjusted = @as(i64, @intCast(input_shape.len)) + axis;
        if (adjusted < 0) return TensorError.InvalidAxis;
        positive_axis = @intCast(adjusted);
    } else {
        positive_axis = @intCast(axis);
    }

    if (positive_axis >= input_shape.len) return TensorError.InvalidAxis;

    // Rest of the function remains the same...
    const dim_size = input_shape[positive_axis];
    var sizes = std.ArrayList(usize).init(pkg_allocator);
    defer sizes.deinit();

    if (split_sizes) |s| {
        // Validate and use provided split sizes
        var total_size: usize = 0;
        for (s) |size| {
            try sizes.append(size);
            total_size += size;
        }
        if (total_size != dim_size) return TensorError.InvalidSplitSize;
    } else {
        // Split into equal parts
        if (dim_size == 0) return TensorError.InvalidSplitSize;
        const split_size = dim_size;
        try sizes.append(split_size);
    }

    // Create output shapes
    var output_shapes = try pkg_allocator.alloc([]usize, sizes.items.len);
    errdefer {
        for (output_shapes) |shape| {
            pkg_allocator.free(shape);
        }
        pkg_allocator.free(output_shapes);
    }

    // Fill output shapes
    for (sizes.items, 0..) |split_size, i| {
        output_shapes[i] = try pkg_allocator.alloc(usize, input_shape.len);
        errdefer pkg_allocator.free(output_shapes[i]);

        @memcpy(output_shapes[i], input_shape);
        output_shapes[i][positive_axis] = split_size;
    }

    return output_shapes;
}

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// The lean version of this method follows the onnx standard.
/// https://onnx.ai/onnx/operators/onnx__Reshape.html
/// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
/// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
pub fn reshape_f32(comptime T: anytype, input: *Tensor(T), newShape: []f32, allowZero: ?bool) !Tensor(T) {
    // Create output tensor with the same size as input but with new shape length
    //std.debug.print("\n[RESHAPE] Input shape: {any}", .{input.shape});
    var temp_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(temp_shape);

    // Initialize first dimension with total size, rest with 1
    temp_shape[0] = input.size;
    for (1..newShape.len) |i| {
        temp_shape[i] = 1;
    }

    var output = try Tensor(T).fromShape(&pkg_allocator, temp_shape);
    errdefer output.deinit();

    // Let reshape_lean handle the actual reshaping logic
    try reshape_lean_f32(T, input, newShape, allowZero, &output);

    return output;
}

/// Given and input tensor and the new shape, returns a new tensor with the same data of the input, in the same order, but a different shape.
/// This version accepts a slice of usize for the new shape.
/// At most one dimension of the new shape can be -1. In this case, the value is inferred from the size of the tensor and the remaining dimensions.
/// A dimension could also be 0, in which case the actual dimension value is unchanged (i.e. taken from the input tensor).
pub fn reshape(comptime T: anytype, input: *Tensor(T), newShape: []const usize, allowZero: ?bool) !Tensor(T) {
    // Create output tensor with the same size as input but with new shape length
    var temp_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(temp_shape);

    // Initialize first dimension with total size, rest with 1
    temp_shape[0] = input.size;
    for (1..newShape.len) |i| {
        temp_shape[i] = 1;
    }

    var output = try Tensor(T).fromShape(&pkg_allocator, temp_shape);
    errdefer output.deinit();

    // Let reshape_lean handle the actual reshaping logic
    try reshape_lean(T, input, newShape, allowZero, &output);

    return output;
}

/// lean version of the reshape function for f32 shape arrays
pub fn reshape_lean_f32(comptime T: anytype, input: *Tensor(T), newShape: []f32, allowZero: ?bool, output: *Tensor(T)) !void {
    _ = allowZero;

    // Create a copy of newShape that we can modify
    var modified_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(modified_shape);

    // Track if we have a -1 dimension and its position
    var neg_one_index: ?usize = null;

    // Calculate product of all non-negative and non-zero dimensions
    var known_dims_product: usize = 1;

    // First pass: identify -1 and 0 dimensions
    for (newShape, 0..) |dim, i| {
        if (dim == 0) {
            if (i >= input.shape.len) {
                return TensorError.InvalidInput;
            }
            modified_shape[i] = input.shape[i];
            known_dims_product *= input.shape[i];
        } else if (dim < 0) {
            if (neg_one_index != null) {
                return TensorError.InvalidInput;
            }
            neg_one_index = i;
            modified_shape[i] = 1; // Temporary value, will be updated later
        } else {
            modified_shape[i] = @as(usize, @intFromFloat(dim));
            known_dims_product *= modified_shape[i];
        }
    }

    try reshape_lean_common(T, input, modified_shape, neg_one_index, known_dims_product, output);
}

/// lean version of the reshape function for usize shape arrays
pub fn reshape_lean(comptime T: anytype, input: *Tensor(T), newShape: []const usize, allowZero: ?bool, output: *Tensor(T)) !void {
    _ = allowZero;

    // Create a copy of newShape that we can modify
    var modified_shape = try pkg_allocator.alloc(usize, newShape.len);
    defer pkg_allocator.free(modified_shape);

    // Track if we have a -1 dimension and its position
    var neg_one_index: ?usize = null;

    // Calculate product of all non-negative and non-zero dimensions
    var known_dims_product: usize = 1;

    // First pass: identify -1 and 0 dimensions
    for (newShape, 0..) |dim, i| {
        if (dim == 0) {
            if (i >= input.shape.len) {
                return TensorError.InvalidInput;
            }
            modified_shape[i] = input.shape[i];
            known_dims_product *= input.shape[i];
        } else if (dim == std.math.maxInt(usize)) { // Use maxInt as a sentinel for -1
            if (neg_one_index != null) {
                return TensorError.InvalidInput;
            }
            neg_one_index = i;
            modified_shape[i] = 1; // Temporary value, will be updated later
        } else {
            modified_shape[i] = dim;
            known_dims_product *= modified_shape[i];
        }
    }

    try reshape_lean_common(T, input, modified_shape, neg_one_index, known_dims_product, output);
}

/// Common implementation for reshape_lean functions
fn reshape_lean_common(comptime T: anytype, input: *Tensor(T), modified_shape: []usize, neg_one_index: ?usize, known_dims_product: usize, output: *Tensor(T)) !void {
    // If we have a -1 dimension, calculate its size
    if (neg_one_index) |idx| {
        if (known_dims_product == 0) {
            return TensorError.InvalidInput;
        }

        if (input.size % known_dims_product != 0) {
            return TensorError.InputArrayWrongSize;
        }

        modified_shape[idx] = input.size / known_dims_product;
    }

    // Calculate total size of modified shape
    var total_size: usize = 1;
    for (modified_shape) |dim| {
        total_size *= dim;
    }

    // Verify sizes match
    if (total_size != input.size) {
        return TensorError.InputArrayWrongSize;
    }

    // Update output shape
    for (modified_shape, 0..) |dim, i| {
        output.shape[i] = dim;
    }
    output.size = total_size;

    // Copy data from input to output
    @memcpy(output.data, input.data);
}

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

/// Implements the ONNX slice operator (https://onnx.ai/onnx/operators/onnx__Slice.html)
/// Takes a tensor and extracts a slice along multiple axes.
/// starts: Starting indices for each axis
/// ends: Ending indices for each axis (exclusive)
/// axes: Which axes to slice (if null, assumes [0,1,2,...])
/// steps: Step sizes for each axis (if null, assumes all 1s)
pub fn slice_onnx(comptime T: type, input: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !Tensor(T) {
    // Create output tensor
    var output = try Tensor(T).fromShape(&pkg_allocator, input.shape);
    errdefer output.deinit();

    try lean_slice_onnx(T, input, starts, ends, axes, steps, &output);
    return output;
}
/// Implements https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
/// Insert single-dimensional entries into the shape of the data tensor.
pub fn unsqueeze(comptime T: type, data: *Tensor(T), axes: *Tensor(i64)) !Tensor(T) {

    // Output rank
    const out_rank = data.shape.len + axes.size;
    const conv_out_rank: i64 = @intCast(out_rank);

    for (0..axes.data.len) |i| {

        // Check if axes are within bounds
        if (axes.data[i] < -conv_out_rank or axes.data[i] >= out_rank) {
            return TensorError.AxisOutOfBounds;
        }

        // Check for duplicates
        for (0..i) |j| {
            if (axes.data[i] == axes.data[j]) {
                return TensorError.DuplicateAxis;
            }
        }
    }

    // Create, fill and return output tensor
    var output = try Tensor(T).init(data.allocator);

    try unsqueeze_lean(T, data, axes, &output);

    return output;
}

/// Lean version of slice_onnx that operates on an existing output tensor
pub fn lean_slice_onnx(comptime T: type, input: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64, output: *Tensor(T)) !void {
    // Validate input lengths
    if (starts.len != ends.len) return TensorError.InvalidSliceIndices;
    if (axes) |a| {
        if (a.len != starts.len) return TensorError.InvalidSliceIndices;
    }
    if (steps) |s| {
        if (s.len != starts.len) return TensorError.InvalidSliceIndices;
    }

    // Create arrays to store the actual indices and steps for each dimension
    var actual_starts = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(actual_starts);
    var actual_ends = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(actual_ends);
    var actual_steps = try pkg_allocator.alloc(i64, input.shape.len);
    defer pkg_allocator.free(actual_steps);

    // Initialize with defaults (full range, step 1)
    for (0..input.shape.len) |i| {
        actual_starts[i] = 0;
        actual_ends[i] = @intCast(input.shape[i]);
        actual_steps[i] = 1;
    }

    // Update with provided values
    for (starts, 0..) |start, i| {
        const axis = if (axes) |a| a[i] else @as(i64, @intCast(i));
        const axis_usize = if (axis < 0) @as(usize, @intCast(axis + @as(i64, @intCast(input.shape.len)))) else @as(usize, @intCast(axis));
        if (axis_usize >= input.shape.len) return TensorError.InvalidSliceIndices;

        const dim_size = @as(i64, @intCast(input.shape[axis_usize]));

        // Handle negative indices and clamp to valid range
        var actual_start = if (start < 0) start + dim_size else start;
        actual_start = @max(0, @min(actual_start, dim_size));
        actual_starts[axis_usize] = actual_start;

        var actual_end = if (ends[i] < 0) ends[i] + dim_size else ends[i];
        if (steps) |s| {
            if (s[i] < 0) {
                // For negative steps, if end is negative, we want to include 0
                actual_end = if (ends[i] < 0) -1 else actual_end;
            } else {
                actual_end = @max(0, @min(actual_end, dim_size));
            }
        } else {
            actual_end = @max(0, @min(actual_end, dim_size));
        }
        actual_ends[axis_usize] = actual_end;

        if (steps) |s| {
            if (s[i] == 0) return TensorError.InvalidSliceStep;
            actual_steps[axis_usize] = s[i];
        }
    }

    // Calculate output shape
    var total_elements: usize = 1;
    for (0..input.shape.len) |i| {
        const start = actual_starts[i];
        const end = actual_ends[i];
        const step = actual_steps[i];

        var dim_size: usize = 0;
        if (step > 0) {
            if (end > start) {
                dim_size = @intCast(@divTrunc((@as(i64, @intCast(end - start)) + step - 1), step));
            }
        } else {
            if (start > end) {
                // For negative steps, we need to handle the range differently
                // Add 1 to end because end is exclusive
                const range = start - (end + 1);
                const abs_step = -step;
                dim_size = @intCast(@divTrunc(range + abs_step - 1, abs_step));
            }
        }
        output.shape[i] = dim_size;
        total_elements *= dim_size;
    }

    // Resize output data if needed
    if (output.data.len != total_elements) {
        if (output.data.len > 0) pkg_allocator.free(output.data);
        output.data = try pkg_allocator.alloc(T, total_elements);
    }
    output.size = total_elements;

    // Helper function to convert flat index to coordinates
    var input_coords = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(input_coords);
    var output_coords = try pkg_allocator.alloc(usize, input.shape.len);
    defer pkg_allocator.free(output_coords);

    // Copy data
    var output_idx: usize = 0;
    while (output_idx < total_elements) : (output_idx += 1) {
        // Convert output_idx to coordinates
        var temp = output_idx;
        for (0..input.shape.len) |i| {
            const dim_i = input.shape.len - 1 - i;
            output_coords[dim_i] = temp % output.shape[dim_i];
            temp /= output.shape[dim_i];
        }

        // Calculate input coordinates
        for (0..input.shape.len) |i| {
            const coord = @as(i64, @intCast(output_coords[i]));
            input_coords[i] = @intCast(actual_starts[i] + coord * actual_steps[i]);
        }

        // Get input value
        const input_idx = try input.flatten_index(input_coords);
        output.data[output_idx] = input.data[input_idx];
    }
}

/// Calculate the output shape of a slice operation without performing the slice
pub fn get_slice_output_shape(input_shape: []const usize, starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) ![]usize {
    std.debug.print("\n[DEBUG] get_slice_output_shape input:", .{});
    std.debug.print("\n  input_shape: {any}", .{input_shape});
    std.debug.print("\n  starts: {any}", .{starts});
    std.debug.print("\n  ends: {any}", .{ends});
    std.debug.print("\n  axes: {any}", .{axes});
    std.debug.print("\n  steps: {any}", .{steps});

    // Validate input lengths
    if (starts.len != ends.len) return TensorError.InvalidSliceIndices;
    if (axes) |a| {
        if (a.len != starts.len) return TensorError.InvalidSliceIndices;
    }
    if (steps) |s| {
        if (s.len != starts.len) return TensorError.InvalidSliceIndices;
    }

    // Create arrays to store the actual indices and steps for each dimension
    var actual_starts = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(actual_starts);
    var actual_ends = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(actual_ends);
    var actual_steps = try pkg_allocator.alloc(i64, input_shape.len);
    defer pkg_allocator.free(actual_steps);

    // Initialize with defaults (full range, step 1)
    for (0..input_shape.len) |i| {
        actual_starts[i] = 0;
        actual_ends[i] = @intCast(input_shape[i]);
        actual_steps[i] = 1;
    }

    std.debug.print("\n[DEBUG] Initial values:", .{});
    std.debug.print("\n  actual_starts: {any}", .{actual_starts});
    std.debug.print("\n  actual_ends: {any}", .{actual_ends});
    std.debug.print("\n  actual_steps: {any}", .{actual_steps});

    // Update with provided values
    for (starts, 0..) |start, i| {
        const axis = if (axes) |a| a[i] else @as(i64, @intCast(i));
        const axis_usize = if (axis < 0) @as(usize, @intCast(axis + @as(i64, @intCast(input_shape.len)))) else @as(usize, @intCast(axis));
        if (axis_usize >= input_shape.len) return TensorError.InvalidSliceIndices;

        const dim_size = @as(i64, @intCast(input_shape[axis_usize]));

        // Handle negative indices and clamp to valid range
        var actual_start = if (start < 0) start + dim_size else start;
        actual_start = @max(0, @min(actual_start, dim_size));
        actual_starts[axis_usize] = actual_start;

        var actual_end = if (ends[i] < 0) ends[i] + dim_size else ends[i];
        if (steps) |s| {
            if (s[i] < 0) {
                // For negative steps, if end is negative, we want to include 0
                actual_end = if (ends[i] < 0) -1 else actual_end;
            } else {
                actual_end = @max(0, @min(actual_end, dim_size));
            }
        } else {
            actual_end = @max(0, @min(actual_end, dim_size));
        }
        actual_ends[axis_usize] = actual_end;

        if (steps) |s| {
            if (s[i] == 0) return TensorError.InvalidSliceStep;
            actual_steps[axis_usize] = s[i];
        }

        std.debug.print("\n[DEBUG] After processing axis {d}:", .{axis});
        std.debug.print("\n  dim_size: {d}", .{dim_size});
        std.debug.print("\n  actual_start: {d}", .{actual_start});
        std.debug.print("\n  actual_end: {d}", .{actual_end});
        std.debug.print("\n  actual_step: {d}", .{actual_steps[axis_usize]});
    }

    std.debug.print("\n[DEBUG] Final values before shape calculation:", .{});
    std.debug.print("\n  actual_starts: {any}", .{actual_starts});
    std.debug.print("\n  actual_ends: {any}", .{actual_ends});
    std.debug.print("\n  actual_steps: {any}", .{actual_steps});

    // Calculate output shape
    // Calculate output shape
    var output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    for (0..input_shape.len) |i| {
        const start = actual_starts[i];
        const end = actual_ends[i];
        const step = actual_steps[i];

        var dim_size: usize = 0;
        if (step > 0) {
            if (end > start) {
                dim_size = @intCast(@divTrunc((@as(i64, @intCast(end - start)) + step - 1), step));
                std.debug.print("\n[DEBUG] Positive step calculation for dim {d}:", .{i});
                std.debug.print("\n  end ({d}) - start ({d}) = {d}", .{ end, start, end - start });
                std.debug.print("\n  (end-start) + step({d}) - 1 = {d}", .{ step, (end - start) + step - 1 });
                std.debug.print("\n  final dim_size = {d}", .{dim_size});
            }
        } else {
            if (start > end) {
                // For negative steps, treat end as inclusive.
                const range = start - end;
                dim_size = @intCast((@divTrunc(range, -step)) + 1);
                std.debug.print("\n[DEBUG] Negative step calculation for dim {d}:", .{i});
                std.debug.print("\n  start ({d}) - end ({d}) = range ({d})", .{ start, end, range });
                std.debug.print("\n  (range) / abs(step) + 1 = {d}", .{dim_size});
            }
        }
        output_shape[i] = dim_size;
    }

    std.debug.print("\n[DEBUG] Final output_shape: {any}\n", .{output_shape});
    return output_shape;
}

/// Implements the ONNX transpose operator (version 21)
/// Transposes the input tensor similar to numpy.transpose.
/// If perm is not provided, reverses the dimensions.
/// If perm is provided, permutes the axes according to the values given.
pub fn transpose_onnx(comptime T: type, input: *Tensor(T), perm: ?[]const usize) !Tensor(T) {
    var output = try Tensor(T).fromShape(&pkg_allocator, input.shape);
    errdefer output.deinit();

    try transpose_onnx_lean(T, input, perm, &output);

    return output;
}

/// Lean version of transpose_onnx that operates on an existing output tensor
//TODO SHAPETRACKER we'll gonna love you
pub fn transpose_onnx_lean(
    comptime T: type,
    input: *Tensor(T),
    perm: ?[]const usize,
    output: *Tensor(T),
) !void {
    // Validate rank
    const rank = input.shape.len;
    if (output.shape.len != rank) {
        return error.InvalidRank; // or however you handle shape mismatch
    }

    // -----------------------------
    // 1) Build the actual perm array
    // -----------------------------
    var real_perm = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(real_perm);

    if (perm) |p| {
        // Validate length
        if (p.len != rank) return error.InvalidPermutation;

        // Validate that p is a valid permutation of [0..rank)
        var used = try pkg_allocator.alloc(bool, rank);
        defer pkg_allocator.free(used);
        @memset(used, false);

        for (p) |idx| {
            if (idx >= rank) return error.InvalidPermutation;
            if (used[idx]) return error.InvalidPermutation;
            used[idx] = true;
        }
        // Copy into real_perm
        for (0..rank) |i| {
            real_perm[i] = p[i];
        }
    } else {
        // If no perm given, ONNX says reverse the dimension order
        for (0..rank) |i| {
            real_perm[i] = rank - 1 - i;
        }
    }

    // -----------------------------
    // 2) Compute input strides
    // -----------------------------
    var input_strides = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(input_strides);

    var stride: usize = 1;
    var i: usize = rank;
    while (i > 0) {
        i -= 1;
        input_strides[i] = stride;
        stride *= input.shape[i];
    }

    // -----------------------------
    // 3) Compute the output shape
    //    and output strides (row-major)
    // -----------------------------
    var output_shape = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(output_shape);

    // shape comes from perm
    for (0..rank) |j| {
        output_shape[j] = input.shape[real_perm[j]];
    }

    var output_strides = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(output_strides);

    stride = 1;
    i = rank;
    while (i > 0) {
        i -= 1;
        output_strides[i] = stride;
        stride *= output_shape[i];
    }

    // -----------------------------
    // 4) Allocate output data if needed
    // -----------------------------
    const total_size = stride; // product of all dims
    if (output.data.len != total_size) {
        // free old data if needed
        if (output.data.len > 0) pkg_allocator.free(output.data);
        output.data = try pkg_allocator.alloc(T, total_size);
    }
    output.size = total_size;

    // Copy shape into output
    @memcpy(output.shape, output_shape);

    // -----------------------------
    // 5) Fill output by iterating over all output coords
    // -----------------------------
    // We'll do a simple nested‐index iteration by flattening the output coordinate.
    // Then we un‐flatten to get [o0, o1, ..., o_{rank-1}].
    const out_data = output.data;
    const in_data = input.data;

    for (0..total_size) |flat_out_idx| {
        // Convert flat_out_idx -> array of indices in [o0, o1, ...]
        var tmp = flat_out_idx;
        var out_coord = try pkg_allocator.alloc(usize, rank);
        defer pkg_allocator.free(out_coord);

        // Unflatten in row-major order
        for (0..rank) |d| {
            out_coord[d] = tmp / output_strides[d];
            tmp %= output_strides[d];
        }

        // Now map output coords back to input coords via perm
        var in_idx: usize = 0;
        for (0..rank) |d| {
            const input_dim_index = real_perm[d]; // which input dimension
            in_idx += out_coord[d] * input_strides[input_dim_index];
        }

        // Do the copy
        out_data[flat_out_idx] = in_data[in_idx];
    }
}

/// Calculate the output shape for an ONNX transpose operation without performing the transpose
pub fn get_transpose_output_shape(input_shape: []const usize, perm: ?[]const usize) ![]usize {
    const rank = input_shape.len;

    // Validate perm if provided
    if (perm) |p| {
        if (p.len != rank) return error.InvalidPermutation;

        // Validate that p is a valid permutation of [0..rank)
        var used = try pkg_allocator.alloc(bool, rank);
        defer pkg_allocator.free(used);
        @memset(used, false);

        for (p) |idx| {
            if (idx >= rank) return error.InvalidPermutation;
            if (used[idx]) return error.InvalidPermutation;
            used[idx] = true;
        }
    }

    // Allocate output shape array
    var output_shape = try pkg_allocator.alloc(usize, rank);
    errdefer pkg_allocator.free(output_shape);

    // Fill output shape based on permutation
    if (perm) |p| {
        for (0..rank) |i| {
            output_shape[i] = input_shape[p[i]];
        }
    } else {
        // If no perm given, reverse the dimension order
        for (0..rank) |i| {
            output_shape[i] = input_shape[rank - 1 - i];
        }
    }

    return output_shape;
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
/// Lean version of unsqueeze, note that previous information stored in output tensor is lost
pub fn unsqueeze_lean(comptime T: type, data: *Tensor(T), axes: *Tensor(i64), output: *Tensor(T)) !void {
    //std.debug.print("\n[UNSQUEEZE] Input shape: {any}", .{data.shape});
    //std.debug.print("\n[UNSQUEEZE] Axes: {any}", .{axes.data});

    // Output rank
    const out_rank = data.shape.len + axes.size;

    // Convert negative axis
    var actual_axes = try data.allocator.alloc(usize, axes.size);
    defer data.allocator.free(actual_axes);

    for (0..axes.size) |i| {
        var conv: i64 = axes.data[i];
        if (conv < 0) {
            conv += @intCast(out_rank);
        }
        const new_axis: usize = @intCast(conv);
        actual_axes[i] = new_axis;
    }

    // Preparing the output shape
    var new_shape = try data.allocator.alloc(usize, out_rank);
    var is_unsqueezed = try data.allocator.alloc(bool, out_rank);
    defer data.allocator.free(new_shape);
    defer data.allocator.free(is_unsqueezed);

    // Initialize support array
    @memset(is_unsqueezed, false);

    // Adding new mono dimentions and setting support array.
    for (0..actual_axes.len) |i| {
        new_shape[actual_axes[i]] = 1;
        is_unsqueezed[actual_axes[i]] = true;
    }

    // Setting positions not marked using data shape
    var data_index: usize = 0;
    for (0..out_rank) |i| {
        if (!is_unsqueezed[i]) {
            new_shape[i] = data.shape[data_index];
            data_index += 1;
        }
    }

    // Modify output tensor
    try output.fill(data.data, new_shape);
    //std.debug.print("\n[UNSQUEEZE] Output shape: {any}\n", .{new_shape});
}

/// Calculate the output shape for an ONNX Unsqueeze operation without performing the operation
pub fn get_unsqueeze_output_shape(input_shape: []const usize, axes: []const i64) ![]usize {
    // Output rank
    const out_rank = input_shape.len + axes.len;

    // Convert negative axes to positive
    var actual_axes = try pkg_allocator.alloc(usize, axes.len);
    defer pkg_allocator.free(actual_axes);

    for (axes, 0..) |axis, i| {
        var conv: i64 = axis;
        if (conv < 0) {
            conv += @intCast(out_rank);
        }
        if (conv < 0 or conv >= out_rank) {
            return TensorError.AxisOutOfBounds;
        }
        const new_axis: usize = @intCast(conv);

        // Check for duplicates
        for (0..i) |j| {
            if (actual_axes[j] == new_axis) {
                return TensorError.DuplicateAxis;
            }
        }
        actual_axes[i] = new_axis;
    }

    // Create output shape array
    var output_shape = try pkg_allocator.alloc(usize, out_rank);
    errdefer pkg_allocator.free(output_shape);

    // Create and initialize support array to track unsqueezed dimensions
    var is_unsqueezed = try pkg_allocator.alloc(bool, out_rank);
    defer pkg_allocator.free(is_unsqueezed);
    @memset(is_unsqueezed, false);

    // Mark unsqueezed dimensions and set them to 1
    for (actual_axes) |axis| {
        output_shape[axis] = 1;
        is_unsqueezed[axis] = true;
    }

    // Fill remaining dimensions with input shape values
    var input_idx: usize = 0;
    for (0..out_rank) |i| {
        if (!is_unsqueezed[i]) {
            output_shape[i] = input_shape[input_idx];
            input_idx += 1;
        }
    }

    return output_shape;
}

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
