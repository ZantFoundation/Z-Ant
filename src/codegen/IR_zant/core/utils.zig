//! Tensor utility functions.
//!
//! Contains helpers, debug printing, layout conversion, and index computation
//! functions that operate on `Tensor(T)` but are not part of its minimal core.

const std = @import("std");
const tensor_module = @import("tensor.zig");
const Tensor = tensor_module.Tensor;

// ---------------------------------------------------------------------------
// Layout conversion (NCHW <-> NHWC)
// ---------------------------------------------------------------------------

pub fn from_NCHW_to_NHWC(comptime T: type, alloc: *const std.mem.Allocator, tensor_nchw: *Tensor(T)) !*Tensor(T) {
    if (tensor_nchw.shape.len != 4) return error.InvalidShape;
    if (tensor_nchw.data.len == 0) return error.EmptyTensor;

    const N = tensor_nchw.shape[0];
    const C = tensor_nchw.shape[1];
    const H = tensor_nchw.shape[2];
    const W = tensor_nchw.shape[3];

    var new_shape = [_]usize{ N, H, W, C };

    const result = try alloc.create(Tensor(T));
    errdefer alloc.destroy(result);

    result.* = try Tensor(T).fromShape(alloc, &new_shape);
    errdefer result.deinit();

    for (0..N) |n| {
        for (0..H) |h| {
            for (0..W) |w| {
                for (0..C) |c| {
                    const old_idx = ((n * C + c) * H + h) * W + w;
                    const new_idx = ((n * H + h) * W + w) * C + c;
                    result.data[new_idx] = tensor_nchw.data[old_idx];
                }
            }
        }
    }

    return result;
}

pub fn from_NHWC_to_NCHW(comptime T: type, alloc: *const std.mem.Allocator, tensor_nhwc: *Tensor(T)) !*Tensor(T) {
    if (tensor_nhwc.shape.len != 4) return error.InvalidShape;
    if (tensor_nhwc.data.len == 0) return error.EmptyTensor;

    const N = tensor_nhwc.shape[0];
    const H = tensor_nhwc.shape[1];
    const W = tensor_nhwc.shape[2];
    const C = tensor_nhwc.shape[3];

    var new_shape = [_]usize{ N, C, H, W };

    const result = try alloc.create(Tensor(T));
    errdefer alloc.destroy(result);

    result.* = try Tensor(T).fromShape(alloc, &new_shape);
    errdefer result.deinit();

    for (0..N) |n| {
        for (0..C) |c| {
            for (0..H) |h| {
                for (0..W) |w| {
                    const old_idx = ((n * H + h) * W + w) * C + c;
                    const new_idx = ((n * C + c) * H + h) * W + w;
                    result.data[new_idx] = tensor_nhwc.data[old_idx];
                }
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Array conversion helpers
// ---------------------------------------------------------------------------

/// Given a Tensor returns the equivalent multidimensional array.
/// NOTE: memory ownership will be transferred to return value.
pub fn toArray(
    comptime T: type,
    self: Tensor(T),
    comptime n_dimensions: usize,
) !MagicalReturnType(T, n_dimensions) {
    if (n_dimensions == 1) return self.data;

    return constructMultidimensionalArray(
        self.allocator,
        T,
        self.data,
        self.shape,
        0,
        n_dimensions,
    );
}

pub fn constructMultidimensionalArray(
    allocator: *const std.mem.Allocator,
    comptime ElementType: type,
    data: []ElementType,
    shape: []usize,
    comptime depth: usize,
    comptime dimension: usize,
) !MagicalReturnType(ElementType, dimension - depth) {
    if (depth == dimension - 1) {
        return data;
    }

    const current_dim = shape[depth];
    var result = try allocator.alloc(
        MagicalReturnType(ElementType, dimension - depth - 1),
        current_dim,
    );

    var offset: usize = 0;
    const sub_array_size = calculateProduct(shape[(depth + 1)..]);

    for (0..current_dim) |i| {
        result[i] = try constructMultidimensionalArray(
            allocator,
            ElementType,
            data[offset .. offset + sub_array_size],
            shape,
            depth + 1,
            dimension,
        );
        offset += sub_array_size;
    }

    return result;
}

pub fn MagicalReturnType(comptime DataType: type, comptime dim_count: usize) type {
    return if (dim_count == 1) []DataType else []MagicalReturnType(DataType, dim_count - 1);
}

// ---------------------------------------------------------------------------
// Index / stride helpers
// ---------------------------------------------------------------------------

/// Calculate the flat index from multi-dimensional indices.
pub fn get_flat_index(comptime T: type, self: *Tensor(T), indices: []usize) !usize {
    if (indices.len != self.shape.len) return error.InvalidIndices;

    var flat_index: usize = 0;
    var stride: usize = 1;

    var i: usize = self.shape.len - 1;
    while (true) {
        flat_index += indices[i] * stride;
        stride *= self.shape[i];
        if (i == 0) break;
        i -= 1;
    }

    return flat_index;
}

/// Compute strides for the tensor (caller must free the returned slice).
pub fn getStrides(comptime T: type, self: *Tensor(T)) ![]usize {
    const num_dims = self.shape.len;
    var strides = try self.allocator.alloc(usize, num_dims);
    strides[num_dims - 1] = 1;
    var i: usize = num_dims - 1;
    while (i > 0) {
        strides[i - 1] = strides[i] * self.shape[i];
        i -= 1;
    }
    return strides;
}

/// Ensures the input shape is 4D by padding with 1s if necessary.
pub inline fn ensure_4D_shape(shape: []const usize, out: *[4]usize) ![]usize {
    const target_dims = 4;
    if (shape.len > target_dims) return error.InvalidDimensions;

    out.* = .{ 1, 1, 1, 1 };
    const start_index = target_dims - shape.len;
    for (shape, 0..) |dim, i| {
        out[start_index + i] = dim;
    }
    return out[0..target_dims];
}

/// Recursive function to copy data between tensor regions.
pub fn copy_data_recursive(
    comptime T: type,
    self: *Tensor(T),
    new_data: []T,
    new_data_index: *usize,
    start_indices: []usize,
    slice_shape: []usize,
    indices: []usize,
    dim: usize,
) !void {
    if (dim == self.shape.len) {
        var self_indices = try self.allocator.alloc(usize, self.shape.len);
        defer self.allocator.free(self_indices);

        for (0..self.shape.len) |i| {
            self_indices[i] = start_indices[i] + indices[i];
        }

        const flat_index = try get_flat_index(T, self, self_indices);
        new_data[new_data_index.*] = self.data[flat_index];
        new_data_index.* += 1;
    } else {
        for (0..slice_shape[dim]) |i| {
            indices[dim] = i;
            try copy_data_recursive(
                T,
                self,
                new_data,
                new_data_index,
                start_indices,
                slice_shape,
                indices,
                dim + 1,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Debug printing
// ---------------------------------------------------------------------------

pub fn info(comptime T: type, self: *Tensor(T)) void {
    std.log.debug("\ntensor infos: ", .{});
    std.log.debug("\n  data type:{}", .{@TypeOf(self.data[0])});
    std.log.debug("\n  size:{}", .{self.getSize()});
    std.log.debug("\n shape.len:{} shape: [ ", .{self.shape.len});
    for (0..self.shape.len) |i| {
        std.log.debug("{} ", .{self.shape[i]});
    }
    std.log.debug("] ", .{});
}

pub fn print(comptime T: type, self: *const Tensor(T)) void {
    std.log.debug("\n  tensor data: ", .{});
    for (0..self.getSize()) |i| {
        std.log.debug("{} ", .{self.data[i]});
    }
    std.log.debug("\n", .{});
}

pub fn printMultidim(comptime T: type, self: *const Tensor(T)) void {
    printMultidimHelper(T, self, 0, 0);
}

fn printMultidimHelper(comptime T: type, self: *const Tensor(T), offset: usize, idx: usize) void {
    for (0..idx) |_| {
        std.log.debug("    ", .{});
    }
    std.log.debug("[", .{});

    if (idx == self.shape.len - 1) {
        for (0..self.shape[self.shape.len - 1]) |i| {
            const local_idx = offset + i;
            std.log.debug("{}, ", .{self.data[local_idx]});
        }
        std.log.debug("],\n", .{});
    } else {
        std.log.debug("\n", .{});
        for (0..self.shape[idx]) |i| {
            printMultidimHelper(T, self, offset + self.shape[idx + 1] * i, idx + 1);
        }
        std.log.debug("\n", .{});

        for (0..idx) |_| {
            std.log.debug("    ", .{});
        }
        std.log.debug("]", .{});
        if (idx != 0) {
            std.log.debug(",\n", .{});
        }
    }
}

pub fn info_metal(comptime T: type, self: *const Tensor(T)) void {
    const tensor_log = std.log.scoped(.tensor);
    tensor_log.debug("Tensor size: {}", .{self.getSize()});
}

// ---------------------------------------------------------------------------
// Benchmarking
// ---------------------------------------------------------------------------

pub fn flatten_index_original(comptime T: type, self: *const Tensor(T), indices: []const usize) !usize {
    if (indices.len != self.shape.len) return error.InvalidIndexLength;

    var idx: usize = 0;
    var stride: usize = 1;

    var i = self.shape.len;
    while (i > 0) : (i -= 1) {
        const rev_idx = i - 1;
        const index = indices[rev_idx];
        if (index >= self.shape[rev_idx]) return error.IndexOutOfBounds;
        idx += index * stride;
        stride *= self.shape[rev_idx];
    }

    return idx;
}

pub fn benchmark_flatten_index(comptime T: type, self: *const Tensor(T), iterations: usize) struct { optimized: u64, original: u64 } {
    var optimized_time: u64 = 0;
    var original_time: u64 = 0;

    const indices = self.allocator.alloc(usize, self.shape.len) catch return .{ .optimized = 0, .original = 0 };
    defer self.allocator.free(indices);

    for (indices, 0..) |*idx, i| {
        idx.* = i % self.shape[i % self.shape.len];
    }

    {
        const start = std.time.milliTimestamp();
        var result: usize = 0;
        for (0..iterations) |_| {
            result +%= self.flatten_index(indices) catch 0;
        }
        const end = std.time.milliTimestamp();
        optimized_time = @intCast(end - start);
        if (result == 0) std.log.info("Benchmark result: {}\n", .{result});
    }

    {
        const start = std.time.milliTimestamp();
        var result: usize = 0;
        for (0..iterations) |_| {
            result +%= flatten_index_original(T, self, indices) catch 0;
        }
        const end = std.time.milliTimestamp();
        original_time = @intCast(end - start);
        if (result == 0) std.log.info("Benchmark result: {}\n", .{result});
    }

    return .{
        .optimized = optimized_time,
        .original = original_time,
    };
}

// ---------------------------------------------------------------------------
// Internal helpers (also used by tensor.zig)
// ---------------------------------------------------------------------------

pub fn calculateProduct(slices: []usize) usize {
    var product: usize = 1;
    for (slices) |elem| product *= elem;
    return product;
}

pub fn flattenArray(comptime T: type, arr: anytype, flatArr: []T, startIndex: usize) usize {
    var idx = startIndex;

    const arrTypeInfo = @typeInfo(@TypeOf(arr));

    if (arrTypeInfo == .array or arrTypeInfo == .pointer) {
        if (@TypeOf(arr[0]) == T) {
            for (arr) |val| {
                flatArr[idx] = val;
                idx += 1;
            }
        } else {
            for (arr) |subArray| {
                idx = flattenArray(T, subArray, flatArr, idx);
            }
        }
    } else {
        @panic("The type of `arr` is not compatible with the required type.");
    }

    return idx;
}
