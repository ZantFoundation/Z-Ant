//! Tensor — the foundational data structure of Z-Ant.
//!
//! A tensor is a multi-dimensional array that generalizes scalars (0-D),
//! vectors (1-D), matrices (2-D) to arbitrary dimensions.
//!
//! This file contains only the minimal struct definition and essential
//! construction / access methods.  Utilities such as debug printing,
//! layout conversion and benchmarking live in `utils.zig`.
pub const utils = @import("utils.zig");

// Re-export layout converters for backward compatibility
pub const from_NCHW_to_NHWC = utils.from_NCHW_to_NHWC;
pub const from_NHWC_to_NCHW = utils.from_NHWC_to_NCHW;

const std = @import("std");

// Internal helpers from utils
const calculateProduct = utils.calculateProduct;
const flattenArray = utils.flattenArray;

pub const AnyTensor = union(enum) {
    i64: *Tensor(i64),
    f64: *Tensor(f64),
    u64: *Tensor(u64),

    f32: *Tensor(f32),
    i32: *Tensor(i32),
    u32: *Tensor(u32),

    f16: *Tensor(f16),
    i16: *Tensor(i16),
    u16: *Tensor(u16),

    i8: *Tensor(i8),
    u8: *Tensor(u8),

    pub fn init(comptime T: type, tensor: *Tensor(T)) AnyTensor {
        inline for (@typeInfo(AnyTensor).@"union".fields) |field| if (field.type == T) {
            return @unionInit(AnyTensor, field.name, tensor);
        };

        @compileError("Unsupported tensor type");
    }

    pub fn deinit(self: *AnyTensor) void {
        return switch (self.*) {
            inline else => |t| t.deinit(),
        };
    }

    pub fn get_shape(self: *const AnyTensor) []usize {
        return switch (self.*) {
            inline else => |t| t.shape,
        };
    }

    pub fn get_size(self: *const AnyTensor) usize {
        return switch (self.*) {
            inline else => |t| t.getSize(),
        };
    }

    pub fn set_shape(self: *AnyTensor, new_shape: []usize) []usize {
        return switch (self.*) {
            inline else => |t| t.shape = new_shape,
        };
    }

    pub fn get_data_as(self: *AnyTensor, comptime T: type) []T {
        switch (self.*) {
            inline else => |t| {
                const tensor_type = @typeInfo(@TypeOf(t.data)).pointer.child;
                if (tensor_type == T) return t.data;
            },
        }

        unreachable;
    }

    pub fn get_data_bytes(self: *const AnyTensor) []const u8 {
        return switch (self.*) {
            inline else => |t| std.mem.sliceAsBytes(t.data),
        };
    }
};

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: *const std.mem.Allocator,
        size: usize = 0,
        data: []T,
        shape: []usize,

        pub fn init(allocator: *const std.mem.Allocator) !Self {
            return .{
                .allocator = allocator,
                .data = &.{},
                .shape = &.{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
        }

        pub fn fromArray(
            allocator: *const std.mem.Allocator,
            arr: anytype,
            shape: []usize,
        ) error{OutOfMemory}!Self {
            const tensor = try fromShape(allocator, shape);
            _ = flattenArray(T, arr, tensor.data, 0);
            return tensor;
        }

        pub fn copy(self: *const Self) !Tensor(T) {
            return .fromArray(self.allocator, self.data, self.shape);
        }

        pub fn fromShape(allocator: *const std.mem.Allocator, shape: []usize) error{OutOfMemory}!Self {
            const size = calculateProduct(shape);

            const tensor_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(tensor_shape);

            const tensor_data = try allocator.alloc(T, size);
            @memset(tensor_data, 0);

            return .{
                .data = tensor_data,
                .shape = tensor_shape,
                .allocator = allocator,
                .size = size,
            };
        }

        /// Initialize a tensor from a const buffer without allocation.
        /// The data and shape buffers must outlive the tensor.
        ///
        /// TODO: constCast SHOULD NOT be used. I do not remove this function because is used
        /// directly by 'codegen'. The problem on top is passing parameters as constants.
        pub fn fromConstBuffer(
            allocator: *const std.mem.Allocator,
            data: []const T,
            shape: []const usize,
        ) Self {
            return .{
                .allocator = allocator,
                .data = @constCast(data),
                .shape = @constCast(shape),
                .size = data.len,
            };
        }

        pub fn getSize(self: *const Self) usize {
            return self.size;
        }

        pub fn get(self: *const Self, index: usize) error{IndexOutOfBounds}!T {
            return if (index >= self.data.len) error.IndexOutOfBounds else self.data[index];
        }

        pub fn set(self: *Self, idx: usize, value: T) error{IndexOutOfBounds}!void {
            if (idx >= self.data.len) return error.IndexOutOfBounds;
            self.data[idx] = value;
        }

        pub fn get_at(self: *const Self, indices: []const usize) !T {
            const idx = try self.flatten_index(indices);
            return self.get(idx);
        }

        pub fn set_at(
            self: *Self,
            indices: []const usize,
            value: T,
        ) error{ IndexOutOfBounds, InvalidIndexLength }!void {
            const idx = try self.flatten_index(indices);
            return self.set(idx, value);
        }

        /// Map multi-dimensional indices to a flat offset into `data`.
        pub fn flatten_index(
            self: *const Self,
            indices: []const usize,
        ) error{ IndexOutOfBounds, InvalidIndexLength }!usize {
            if (indices.len != self.shape.len) return error.InvalidIndexLength;

            return switch (self.shape.len) {
                1 => if (indices[0] >= self.shape[0]) error.IndexOutOfBounds else indices[0],
                2 => blk: {
                    const i = indices[0];
                    const j = indices[1];
                    if (i >= self.shape[0] or j >= self.shape[1]) return error.IndexOutOfBounds;
                    break :blk i * self.shape[1] + j;
                },
                3 => blk: {
                    const i = indices[0];
                    const j = indices[1];
                    const k = indices[2];
                    if (i >= self.shape[0] or j >= self.shape[1] or k >= self.shape[2])
                        break :blk error.IndexOutOfBounds;

                    break :blk i * (self.shape[1] * self.shape[2]) + j * self.shape[2] + k;
                },
                4 => blk: {
                    const i = indices[0];
                    const j = indices[1];
                    const k = indices[2];
                    const l = indices[3];
                    if (i >= self.shape[0] or j >= self.shape[1] or
                        k >= self.shape[2] or l >= self.shape[3])
                        break :blk error.IndexOutOfBounds;

                    const s1 = self.shape[1] * self.shape[2] * self.shape[3];
                    const s2 = self.shape[2] * self.shape[3];
                    break :blk i * s1 + j * s2 + k * self.shape[3] + l;
                },
                else => blk: {
                    var idx: usize = 0;
                    var stride: usize = 1;
                    var dim = self.shape.len;
                    while (dim > 0) : (dim -= 1) {
                        const d = dim - 1;
                        if (indices[d] >= self.shape[d]) break :blk error.IndexOutOfBounds;
                        idx += indices[d] * stride;
                        stride *= self.shape[d];
                    }
                    break :blk idx;
                },
            };
        }

        pub fn setToZero(self: *Self) !void {
            if (self.getSize() == 0) return error.TensorNotInitialized;
            @memset(self.data, 0);
        }
    };
}
