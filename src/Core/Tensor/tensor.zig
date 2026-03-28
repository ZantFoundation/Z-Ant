//! Tensor has a crucial role in all the project. Is the foundamental class around witch everything
//! is constructed. A tensor is a multi-dimensional array or a mathematical object that generalizes
//! the concept of scalars, vectors, and matrices to higher dimensions. A scalar is a 0-dimensional
//! tensor, a vector is a 1-dimensional tensor, and a matrix is a 2-dimensional tensor. Tensors can extend
//! to even higher dimensions (3D, 4D, etc.).

pub const math_lean = @import("TensorMath/tensor_math_standard.zig");
pub const math_standard = @import("TensorMath/tensor_math_standard.zig");
pub const quantized_math = @import("QuantTensorMath/quant_tensor_math_standard.zig");
pub const accelerators = @import("Accelerators/mod.zig");

const std = @import("std");
const zant = @import("../../zant.zig");

const pkgAllocator = zant.utils.allocator;
const tMath = math_standard;
const error_handler = zant.utils.error_handler;
const TensorError = error_handler.TensorError;

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

pub const TensorType = enum {
    Tensor,
    QuantTensor,
    ClusterTensor,
    null,
};

pub const QuantDetails = struct {
    tensorType: TensorType,
    scale_factor: f32, // hardcoded data type
    zero_point: i32,
};

pub const ClusterDetails = struct {
    tensorType: TensorType,
    lookup_table: []f32,
    table_size: usize,
};

pub const TensorDetails = union(enum) {
    none,
    quant: QuantDetails,
    cluster: ClusterDetails,
};

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: *const std.mem.Allocator,

        // duplicated field as data.len, now this field
        // cannot be removed because too much methods calls it.
        // tests says that this field is used to check if computed
        // size is equal to data.len, it should not be present..
        size: usize = 0,
        // representation of 2D data of tensor
        data: []T,
        // multidimensional structure
        shape: []usize,

        /// Method used to initialize an undefined Tensor. It just set the allocator.
        /// More usefull methods are:
        ///  - fromArray()
        ///  - copy()
        ///  - fromShape()
        /// TODO: init should not return never an error.
        /// To fix this tiny error too much work is required.
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

        /// Given a multidimensional array with its shape, returns the equivalent Tensor.
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn fromArray(
            allocator: *const std.mem.Allocator,
            arr: anytype,
            shape: []usize,
        ) error{OutOfMemory}!Self {
            const tensor = try fromShape(allocator, shape);
            _ = flattenArray(T, arr, tensor.data, 0);
            return tensor;
        }

        /// Given the Tensor (self) returns the equivalent multidimensional array.
        /// See constructMultidimensionalArray() in this file.
        /// NOTE: memory ownership will be transfered to return value.
        pub fn toArray(
            self: Self,
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

        /// Returns a Tensor witch is the copy of this Tensor (self).
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn copy(self: *const Self) !Tensor(T) {
            return .fromArray(self.allocator, self.data, self.shape);
        }

        /// Return a all-zero tensor starting from the given shape
        /// It substitute init(), but defer yourTensor.deinit() is still necessary.
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

        /// Initialize a tensor from a const buffer without allocation
        /// Useful for freestanding targets where dynamic allocation is not available
        /// The data and shape buffers must outlive the tensor
        ///
        /// @ANDRVV TODO: constCast SHOULD NOT be used. I do not remove this function because is used
        /// directly by 'codegen'. The problem on top is passing parameters as constants.
        pub fn fromConstBuffer(
            allocator: std.mem.Allocator,
            data: []const T,
            shape: []const usize,
        ) Self {
            return .{
                .allocator = allocator,
                .data = @constCast(data),
                .shape = @constCast(shape),
            };
        }

        pub fn getSize(self: *const Self) usize {
            return self.size;
        }

        pub fn get(self: *const Self, index: usize) error{IndexOutOfBounds}!T {
            return if (index >= self.data.len)
                error.IndexOutOfBounds
            else
                self.data[index];
        }

        pub fn set(self: *Self, idx: usize, value: T) error{IndexOutOfBounds}!void {
            if (idx >= self.data.len) return error.IndexOutOfBounds;
            self.data[idx] = value;
        }

        /// Given the coordinates (indices) it returns the correspondant value in the
        /// multidimensional array.
        /// See flatten_index().
        pub fn get_at(self: *const Self, indices: []const usize) !T {
            const idx = try self.flatten_index(indices);
            return self.get(idx);
        }

        /// Given the the value and the coordinates (indices), it sets the value in
        /// the multidimensional array at the specified coordinates.
        /// See flatten_index().
        pub fn set_at(
            self: *Self,
            indices: []const usize,
            value: T,
        ) error{ IndexOutOfBounds, InvalidIndexLength }!void {
            const idx = try self.flatten_index(indices);
            return self.set(idx, value);
        }

        /// Starting from the monodimensional array self.data and the shape self.shape,
        /// it returns the equivalent multidimensional array
        fn constructMultidimensionalArray(
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

        fn MagicalReturnType(comptime DataType: type, comptime dim_count: usize) type {
            return if (dim_count == 1) []DataType else []MagicalReturnType(DataType, dim_count - 1);
        }

        /// Given the coordinates (indices) of a multidimensional Tensor returns
        /// the correspondant position in the monodimensional space of self.data
        pub fn flatten_index(
            self: *const Self,
            indices: []const usize,
        ) error{ IndexOutOfBounds, InvalidIndexLength }!usize {
            if (indices.len != self.shape.len) return error.InvalidIndexLength;

            // Fast paths for common dimensions
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

                    const stride1 = self.shape[1] * self.shape[2];
                    const stride2 = self.shape[2];
                    break :blk i * stride1 + j * stride2 + k;
                },
                4 => blk: {
                    const i = indices[0];
                    const j = indices[1];
                    const k = indices[2];
                    const l = indices[3];
                    if (i >= self.shape[0] or j >= self.shape[1] or
                        k >= self.shape[2] or l >= self.shape[3])
                        break :blk error.IndexOutOfBounds;

                    const stride1 = self.shape[1] * self.shape[2] * self.shape[3];
                    const stride2 = self.shape[2] * self.shape[3];
                    const stride3 = self.shape[3];
                    break :blk i * stride1 + j * stride2 + k * stride3 + l;
                },
                else => blk: {
                    // For 5D and higher dimensions
                    if (self.shape.len == 5) {
                        // Special case for 5D tensors - direct calculation without strides array
                        const i = indices[0];
                        const j = indices[1];
                        const k = indices[2];
                        const l = indices[3];
                        const m = indices[4];

                        if (i >= self.shape[0] or j >= self.shape[1] or
                            k >= self.shape[2] or l >= self.shape[3] or
                            m >= self.shape[4])
                            break :blk error.IndexOutOfBounds;

                        const stride1 = self.shape[1] * self.shape[2] * self.shape[3] * self.shape[4];
                        const stride2 = self.shape[2] * self.shape[3] * self.shape[4];
                        const stride3 = self.shape[3] * self.shape[4];
                        const stride4 = self.shape[4];

                        break :blk i * stride1 + j * stride2 + k * stride3 + l * stride4 + m;
                    } else {
                        // For dimensions 6+, use the original algorithm which is simpler and works well
                        var idx: usize = 0;
                        var stride: usize = 1;

                        // Process dimensions from right to left in a single pass
                        var i = self.shape.len;
                        while (i > 0) : (i -= 1) {
                            const rev_idx = i - 1;
                            const index = indices[rev_idx];

                            if (index >= self.shape[rev_idx]) {
                                break :blk error.IndexOutOfBounds;
                            }

                            idx += index * stride;
                            stride *= self.shape[rev_idx];
                        }

                        break :blk idx;
                    }
                },
            };
        }

        /// Original implementation of flatten_index for benchmarking
        pub fn flatten_index_original(self: *const Self, indices: []const usize) !usize {
            if (indices.len != self.shape.len) {
                return error.InvalidIndexLength;
            }

            var idx: usize = 0;
            var stride: usize = 1;

            // Process dimensions from right to left in a single pass
            var i = self.shape.len;
            while (i > 0) : (i -= 1) {
                const rev_idx = i - 1;
                const index = indices[rev_idx];

                if (index >= self.shape[rev_idx]) {
                    return error.IndexOutOfBounds;
                }

                idx += index * stride;
                stride *= self.shape[rev_idx];
            }

            return idx;
        }

        /// Benchmark function to compare flatten_index implementations
        pub fn benchmark_flatten_index(self: *const Self, iterations: usize) struct { optimized: u64, original: u64 } {
            var optimized_time: u64 = 0;
            var original_time: u64 = 0;

            // Create test indices
            const indices = self.allocator.alloc(usize, self.shape.len) catch return .{ .optimized = 0, .original = 0 };
            defer self.allocator.free(indices);

            for (indices, 0..) |*idx, i| {
                idx.* = i % self.shape[i % self.shape.len];
            }

            // Benchmark optimized version
            {
                const start = std.time.milliTimestamp();
                var result: usize = 0;

                for (0..iterations) |_| {
                    result +%= self.flatten_index(indices) catch 0;
                }

                const end = std.time.milliTimestamp();
                optimized_time = @intCast(end - start);

                // Use result to prevent optimization
                if (result == 0) {
                    std.log.info("Benchmark result: {}\n", .{result});
                }
            }

            // Benchmark original version
            {
                const start = std.time.milliTimestamp();
                var result: usize = 0;

                for (0..iterations) |_| {
                    result +%= self.flatten_index_original(indices) catch 0;
                }

                const end = std.time.milliTimestamp();
                original_time = @intCast(end - start);

                // Use result to prevent optimization
                if (result == 0) {
                    std.log.info("Benchmark result: {}\n", .{result});
                }
            }

            return .{
                .optimized = optimized_time,
                .original = original_time,
            };
        }

        pub fn slice(self: *Tensor(T), start_indices: []usize, slice_shape: []usize) !Tensor(T) {
            // Validate input
            if (start_indices.len != self.shape.len) return TensorError.InvalidSliceIndices;
            if (slice_shape.len != self.shape.len) return TensorError.InvalidSliceShape;

            // Verify that the slice is within bounds
            for (0..self.shape.len) |i| {
                if (start_indices[i] + slice_shape[i] > self.shape[i]) return TensorError.SliceOutOfBounds;
            }

            // Calculate the total size of the new tensor
            var new_size: usize = 1;
            for (slice_shape) |dim| {
                new_size *= dim;
            }

            // Allocate data for the new tensor
            const new_data = try self.allocator.alloc(T, new_size);

            // Prepare for copying data
            const num_dims = self.shape.len;

            // Strides for the original tensor
            const strides = try self.getStrides();
            defer self.allocator.free(strides);

            // Recursive function to copy data
            const indices = try self.allocator.alloc(usize, num_dims);
            defer self.allocator.free(indices);

            for (indices) |*idx| idx.* = 0;

            var new_data_index: usize = 0;

            try copy_data_recursive(
                self,
                new_data,
                &new_data_index,
                start_indices,
                slice_shape,
                indices,
                0,
            );

            // Create the new tensor
            var new_tensor = try Tensor(T){
                .data = new_data,
                .shape = try self.allocator.dupe(usize, slice_shape),
                .allocator = self.allocator,
            };

            _ = &new_tensor;

            return new_tensor;
        }

        // Recursive function to copy data
        fn copy_data_recursive(
            self: *Tensor(T),
            new_data: []T,
            new_data_index: *usize,
            start_indices: []usize,
            slice_shape: []usize,
            indices: []usize,
            dim: usize,
        ) !void {
            if (dim == self.shape.len) {
                // Calculate the index in the original tensor
                var self_indices = try self.allocator.alloc(usize, self.shape.len);
                defer self.allocator.free(self_indices);

                for (0..self.shape.len) |i| {
                    self_indices[i] = start_indices[i] + indices[i];
                }

                const flat_index = try self.get_flat_index(self_indices);
                new_data[new_data_index.*] = self.data[flat_index];
                new_data_index.* += 1;
            } else {
                for (0..slice_shape[dim]) |i| {
                    indices[dim] = i;
                    try copy_data_recursive(
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

        // Helper function to calculate the flat index from multi-dimensional indices
        pub fn get_flat_index(self: *Tensor(T), indices: []usize) !usize {
            if (indices.len != self.shape.len) return TensorError.InvalidIndices;

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

        // Function to calculate strides for the tensor
        pub fn getStrides(self: *Tensor(T)) ![]usize {
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

        /// Prints all the possible details of a tensor.
        /// Very usefull in debugging.
        pub fn info(self: *Self) void {
            std.log.debug("\ntensor infos: ", .{});
            std.log.debug("\n  data type:{}", .{@TypeOf(self.data[0])});
            std.log.debug("\n  size:{}", .{self.getSize()});
            std.log.debug("\n shape.len:{} shape: [ ", .{self.shape.len});
            for (0..self.shape.len) |i| {
                std.log.debug("{} ", .{self.shape[i]});
            }
            std.log.debug("] ", .{});
            //self.print();
        }

        /// Prints all the array self.data in an array.
        pub fn print(self: *const Self) void {
            std.log.debug("\n  tensor data: ", .{});
            for (0..self.getSize()) |i| {
                std.log.debug("{} ", .{self.data[i]});
            }
            std.log.debug("\n", .{});
        }

        /// Print thetry Tensor() to console in a more readable way.
        pub fn printMultidim(self: *const Self) void {
            // Allocate array to store the indices
            self._printMultidimHelper(0, 0);
        }

        fn _printMultidimHelper(self: *const Self, offset: usize, idx: usize) void {
            // Print opening bracket with a number of tab that is equals to idx
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
                    self._printMultidimHelper(offset + self.shape[idx + 1] * i, idx + 1);
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

        /// Set all tensor values to zero.
        pub fn setToZero(self: *Self) !void {
            if (self.getSize() == 0) {
                return TensorError.TensorNotInitialized;
            }
            @memset(self.data, 0);
        }

        /// Implements the ONNX slice operator (https://onnx.ai/onnx/operators/onnx__Slice.html)
        /// Takes a tensor and extracts a slice along multiple axes.
        /// starts: Starting indices for each axis
        /// ends: Ending indices for each axis (exclusive)
        /// axes: Which axes to slice (if null, assumes [0,1,2,...])
        /// steps: Step sizes for each axis (if null, assumes all 1s)
        pub fn slice_onnx(self: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !Tensor(T) {
            return tMath.slice_onnx(T, self, starts, ends, axes, steps);
        }

        // Ensures the input shape is 4D by padding with 1s if necessary. Returns an error if the shape
        // has more than 4 dimensions.
        //
        // - `shape`: The shape of the tensor
        //
        pub inline fn ensure_4D_shape(shape: []const usize, out: *[4]usize) ![]usize {
            // The fixed dimension should be 4. Will updatein future
            // [batch, channel, row, column]
            const target_dims = 4;

            if (shape.len > target_dims) {
                return error.InvalidDimensions;
            }

            out.* = .{ 1, 1, 1, 1 };

            // caulculate starting index to start
            const start_index = target_dims - shape.len;

            // copy values into last positions
            for (shape, 0..) |dim, i| {
                out[start_index + i] = dim;
            }

            return out[0..target_dims];
        }

        /// Bare metal version of tensor info that uses a logging function instead of std.debug.print
        pub fn info_metal(self: *const Self) void {
            const tensor_log = std.log.scoped(.tensor);
            tensor_log.debug("Tensor size: {}", .{self.getSize()});
        }
    };
}

fn calculateProduct(slices: []usize) usize {
    var product: usize = 1;
    for (slices) |elem| product *= elem;
    return product;
}

/// Recursive function to flatten a multidimensional array
fn flattenArray(comptime T: type, arr: anytype, flatArr: []T, startIndex: usize) usize {
    var idx = startIndex;

    const arrTypeInfo = @typeInfo(@TypeOf(arr));

    if (arrTypeInfo == .array or arrTypeInfo == .pointer) {
        // if arr is a lice or 1d  DIRECTLY COPY
        if (@TypeOf(arr[0]) == T) {
            for (arr) |val| {
                flatArr[idx] = val;
                idx += 1;
            }
        } else {
            // iff arr is mulltidimensional array recursive call
            for (arr) |subArray| {
                idx = flattenArray(T, subArray, flatArr, idx);
            }
        }
    } else {
        @panic("The type of `arr` is not compatible with the required type.");
    }

    return idx;
}

pub fn from_NCHW_to_NHWC(comptime T: type, alloc: *const std.mem.Allocator, tensor_nchw: *Tensor(T)) !*Tensor(T) {
    if (tensor_nchw.shape.len != 4) {
        return error.InvalidShape;
    }
    if (tensor_nchw.data.len == 0) {
        return error.EmptyTensor;
    }

    // Extract dimensions assuming NCHW layout
    const N = tensor_nchw.shape[0];
    const C = tensor_nchw.shape[1];
    const H = tensor_nchw.shape[2];
    const W = tensor_nchw.shape[3];

    // New shape will be NHWC
    var new_shape = [_]usize{ N, H, W, C };

    const result = try alloc.create(Tensor(T));
    errdefer alloc.destroy(result);

    result.* = try Tensor(T).fromShape(alloc, &new_shape);
    errdefer result.deinit();

    // Iterate through dimensions in the order of the DESTINATION (NHWC)
    // to keep memory writes sequential.
    for (0..N) |n| {
        for (0..H) |h| {
            for (0..W) |w| {
                for (0..C) |c| {
                    // Source is NCHW: ((n * C + c) * H + h) * W + w
                    const old_idx = ((n * C + c) * H + h) * W + w;

                    // Dest is NHWC: ((n * H + h) * W + w) * C + c
                    const new_idx = ((n * H + h) * W + w) * C + c;

                    result.data[new_idx] = tensor_nchw.data[old_idx];
                }
            }
        }
    }

    return result;
}

pub fn from_NHWC_to_NCHW(comptime T: type, alloc: *const std.mem.Allocator, tensor_nhwc: *Tensor(T)) !*Tensor(T) {
    if (tensor_nhwc.shape.len != 4) {
        return error.InvalidShape;
    }
    if (tensor_nhwc.data.len == 0) {
        return error.EmptyTensor;
    }

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
