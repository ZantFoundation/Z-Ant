const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;

pub fn transposeLastTwo(comptime T: anytype, tensor: *const Tensor(T)) !Tensor(T) {
    //std.log.debug("\n[DEBUG] transposeLastTwo:", .{});
    //std.log.debug("\n  Input tensor shape: ", .{});
    //for (tensor.shape) |s| std.log.debug("{d} ", .{s});

    // Special case for 1D tensors
    if (tensor.shape.len == 1) {
        // For 1D tensors, we just return a copy since transpose doesn't change anything
        var newShape = try pkg_allocator.alloc(usize, 1);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = tensor.shape[0];

        // Create a copy of the input data
        const outData = try pkg_allocator.alloc(T, tensor.size);
        errdefer pkg_allocator.free(outData);

        // Copy the data
        @memcpy(outData, tensor.data);

        return Tensor(T){
            .data = outData,
            .size = tensor.size,
            .shape = newShape,
            .allocator = &pkg_allocator,
        };
    }

    // Verifying correct shape for 2D and 4D tensors
    if (tensor.shape.len != 2 and tensor.shape.len != 4) {
        //std.log.debug("\n  Error: Expected 2D or 4D tensor, got {d}D", .{tensor.shape.len});
        return error.InputTensorsWrongShape;
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

    //std.log.debug("\n  Rows: {d}, Cols: {d}, Total: {d}", .{ rows, cols, total });
    //std.log.debug("\n  New shape: ", .{});
    //for (newShape) |s| std.log.debug("{d} ", .{s});

    // Create a non-const copy of the input data using pkg_allocator
    const outData = try pkg_allocator.alloc(T, total);
    errdefer pkg_allocator.free(outData);

    //std.log.debug("\n  Transposing data...", .{});

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

    //std.log.debug("\n  Transpose complete", .{});

    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = &pkg_allocator,
    };
}
