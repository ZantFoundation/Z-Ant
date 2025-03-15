//! Tensor has a crucial role in all the project. Is the foundamental class around witch everything
//! is constructed. A tensor is a multi-dimensional array or a mathematical object that generalizes
//! the concept of scalars, vectors, and matrices to higher dimensions. A scalar is a 0-dimensional
//! tensor, a vector is a 1-dimensional tensor, and a matrix is a 2-dimensional tensor. Tensors can extend
//! to even higher dimensions (3D, 4D, etc.).
pub const math_lean = @import("TensorMath/tensor_math_standard.zig");
pub const math_standard = @import("TensorMath/tensor_math_standard.zig");

const std = @import("std");
const zant = @import("../../zant.zig");

const tMath = math_standard;
const error_handler = zant.utils.error_handler;
const TensorError = error_handler.TensorError;
const ArgumentError = error_handler.ArgumentError;

pub var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;

pub fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
}

///Class Tensor.
///Return a generic type structure
pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T, //contains all the data of the tensor in a monodimensional array
        size: usize, //dimension of the tensor, equal to data.len
        shape: []usize, //defines the multidimensional structure of the tensor
        allocator: *const std.mem.Allocator, //allocator used in the memory initialization of the tensor
        owns_memory: bool, //whether this tensor owns its memory and should free it

        ///Method used to initialize an undefined Tensor. It just set the allocator.
        /// More usefull methods are:
        ///  - fromArray()
        ///  - copy()
        ///  - fromShape()
        pub fn init(allocator: *const std.mem.Allocator) !@This() {
            return @This(){
                .data = &[_]T{},
                .size = 0,
                .shape = &[_]usize{},
                .allocator = allocator,
                .owns_memory = true,
            };
        }

        ///Free all the possible allocation, use it every time you create a new Tensor ( defer yourTensor.deinit() )
        pub fn deinit(self: *@This()) void {
            if (self.owns_memory) {
                if (self.data.len > 0) {
                    self.allocator.free(self.data);
                    self.data = &[_]T{};
                }
                if (self.shape.len > 0) {
                    self.allocator.free(self.shape);
                    self.shape = &[_]usize{};
                }
            }
        }

        ///Given a multidimensional array with its shape, returns the equivalent Tensor.
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn fromArray(allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !@This() {

            //const adjusted_shape = try ensure_4D_shape(shape);

            // Calculate total size based on shape
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            // Allocate memory for tensor shape
            const tensorShape = try allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            // Allocate memory for tensor data
            const tensorData = try allocator.alloc(T, total_size);

            // Flatten the input array into tensor data
            _ = flattenArray(T, inputArray, tensorData, 0);

            // Return the new tensor
            return @This(){
                .data = tensorData,
                .size = total_size,
                .shape = tensorShape,
                .allocator = allocator,
                .owns_memory = true,
            };
        }

        /// Given the Tensor (self) returns the equivalent multidimensional array.
        /// See constructMultidimensionalArray() in this file.
        /// IMPORTANT: Remember to cal yourAllocator.free(yourMultidimArray) otherwise it generates a memory leak!
        pub fn toArray(self: @This(), comptime dimension: usize) !MagicalReturnType(T, dimension) {
            if (dimension == 1) {
                return self.data;
            }
            return constructMultidimensionalArray(self.allocator, T, self.data, self.shape, 0, dimension);
        }

        fn setAllocator(tensor: *Tensor(T), alloc: *const std.mem.Allocator) void {
            tensor.allocator = alloc;
        }

        /// Returns a Tensor witch is the copy of this Tensor (self).
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn copy(self: *@This()) !Tensor(T) {
            return try Tensor(T).fromArray(self.allocator, self.data, self.shape);
        }

        /// Return a all-zero tensor starting from the given shape
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn fromShape(allocator: *const std.mem.Allocator, shape: []usize) !@This() {
            //const adjusted_shape = try ensure_4D_shape(shape);

            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            const tensorShape = try allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            const tensorData = try allocator.alloc(T, total_size);
            @memset(tensorData, 0);

            return @This(){
                .data = tensorData,
                .size = total_size,
                .shape = tensorShape,
                .allocator = allocator,
                .owns_memory = true,
            };
        }

        /// Initialize a tensor from a const buffer without allocation
        /// Useful for freestanding targets where dynamic allocation is not available
        /// The data and shape buffers must outlive the tensor
        pub fn fromConstBuffer(allocator: *const std.mem.Allocator, data: []const T, shape: []const usize) @This() {
            return @This(){
                .data = @constCast(data),
                .size = data.len,
                .shape = @constCast(shape),
                .allocator = allocator,
                .owns_memory = false,
            };
        }

        /// Given any array and its shape it reshape the tensor and update .data
        pub fn fill(self: *@This(), inputArray: anytype, shape: []usize) !void {
            //const adjusted_shape = try ensure_4D_shape(shape);

            // Allocate new memory first
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            const tensorShape = try self.allocator.alloc(usize, shape.len);
            errdefer self.allocator.free(tensorShape);
            @memcpy(tensorShape, shape);

            // Create a copy of the input data to avoid issues if inputArray is part of self.data
            const tensorData = try self.allocator.alloc(T, total_size);
            errdefer self.allocator.free(tensorData);

            // Handle different input array types
            const InputType = @TypeOf(inputArray);
            if (@typeInfo(InputType) == .pointer and @typeInfo(@typeInfo(InputType).pointer.child) == .array) {
                // Handle multi-dimensional arrays
                _ = flattenArray(T, inputArray, tensorData, 0);
            } else {
                // Handle slices (like data from another tensor)
                if (inputArray.len != total_size) {
                    return error.InvalidInputArraySize;
                }
                @memcpy(tensorData, inputArray);
            }

            // Free old memory only if we own it
            if (self.owns_memory) {
                if (self.data.len > 0) {
                    self.allocator.free(self.data);
                }
                if (self.shape.len > 0) {
                    self.allocator.free(self.shape);
                }
            }

            // Update tensor with new data
            self.data = tensorData;
            self.size = total_size;
            self.shape = tensorShape;
            self.owns_memory = true;
        }

        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///--------------------------------------------------------------------------getters and setters---------------------------------------------------------------
        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///Set the shape of a Tensor.
        ///Returns the size of the Tensor.
        pub fn getSize(self: *@This()) usize {
            return self.size;
        }

        ///Given an index, return the value at self.data[index].
        /// Errors:
        ///     - error.IndexOutOfBounds;
        pub fn get(self: *const @This(), idx: usize) !T {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            return self.data[idx];
        }

        ///Set to value the data at self.data[idx].
        /// Errors:
        ///     - error.IndexOutOfBounds;
        pub fn set(self: *@This(), idx: usize, value: T) !void {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            self.data[idx] = value;
        }

        /// Given the coordinates (indices) it returns the correspondant value in the
        /// multidimensional array.
        /// See flatten_index().
        pub fn get_at(self: *const @This(), indices: []const usize) !T {
            const idx = try self.flatten_index(indices);
            return self.get(idx);
        }

        /// Given the the value and the coordinates (indices), it sets the value in
        /// the multidimensional array at the specified coordinates.
        /// See flatten_index().
        pub fn set_at(self: *@This(), indices: []const usize, value: T) !void {
            const idx = try self.flatten_index(indices);
            return self.set(idx, value);
        }

        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///-------------------------------------------------------------------------------------utils------------------------------------------------------------------
        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///Starting from the monodimensional array self.data and the shape self.shape, it returns the equivalent multidimensional array
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

            // defer allocator.free(result); ??????????? MARCO : era già commentata, ci va o meno la .free()? non credo vada liberato perchè è lui stesso l'array multidim.
            // non andrebbe però creato un metodo freeMultidimensionalArray() che fa la stessa cosa ma librando spazio?
            // AGGIORANEMENTO: nei tests_tensor mi è bastato fare: line 197 -> defer allocator.free(array_from_tensor);

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

        fn calculateProduct(slices: []usize) usize {
            var product: usize = 1;
            for (slices) |elem| {
                product *= elem;
            }
            return product;
        }

        /// Given the coordinates (indices) of a multidimensional Tensor returns the correspondant position
        /// in the monodimensional space of self.data
        pub fn flatten_index(self: *const @This(), indices: []const usize) !usize {
            if (indices.len != self.shape.len) {
                return error.InvalidIndexLength;
            }

            // Fast paths for common dimensions
            switch (self.shape.len) {
                1 => {
                    if (indices[0] >= self.shape[0]) return error.IndexOutOfBounds;
                    return indices[0];
                },
                2 => {
                    const i = indices[0];
                    const j = indices[1];
                    if (i >= self.shape[0] or j >= self.shape[1]) return error.IndexOutOfBounds;
                    return i * self.shape[1] + j;
                },
                3 => {
                    const i = indices[0];
                    const j = indices[1];
                    const k = indices[2];
                    if (i >= self.shape[0] or j >= self.shape[1] or k >= self.shape[2])
                        return error.IndexOutOfBounds;

                    const stride1 = self.shape[1] * self.shape[2];
                    const stride2 = self.shape[2];
                    return i * stride1 + j * stride2 + k;
                },
                4 => {
                    const i = indices[0];
                    const j = indices[1];
                    const k = indices[2];
                    const l = indices[3];
                    if (i >= self.shape[0] or j >= self.shape[1] or
                        k >= self.shape[2] or l >= self.shape[3])
                        return error.IndexOutOfBounds;

                    const stride1 = self.shape[1] * self.shape[2] * self.shape[3];
                    const stride2 = self.shape[2] * self.shape[3];
                    const stride3 = self.shape[3];
                    return i * stride1 + j * stride2 + k * stride3 + l;
                },
                else => {
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
                            return error.IndexOutOfBounds;

                        const stride1 = self.shape[1] * self.shape[2] * self.shape[3] * self.shape[4];
                        const stride2 = self.shape[2] * self.shape[3] * self.shape[4];
                        const stride3 = self.shape[3] * self.shape[4];
                        const stride4 = self.shape[4];

                        return i * stride1 + j * stride2 + k * stride3 + l * stride4 + m;
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
                                return error.IndexOutOfBounds;
                            }

                            idx += index * stride;
                            stride *= self.shape[rev_idx];
                        }

                        return idx;
                    }
                },
            }
        }

        /// Original implementation of flatten_index for benchmarking
        pub fn flatten_index_original(self: *const @This(), indices: []const usize) !usize {
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
        pub fn benchmark_flatten_index(self: *const @This(), iterations: usize) struct { optimized: u64, original: u64 } {
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
                    std.debug.print("Benchmark result: {}\n", .{result});
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
                    std.debug.print("Benchmark result: {}\n", .{result});
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
            var new_tensor = Tensor(T){
                .data = new_data,
                .shape = try self.allocator.dupe(usize, slice_shape),
                .size = new_size,
                .allocator = self.allocator,
                .owns_memory = true,
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
        fn get_flat_index(self: *Tensor(T), indices: []usize) !usize {
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
        pub fn info(self: *@This()) void {
            std.debug.print("\ntensor infos: ", .{});
            std.debug.print("\n  data type:{}", .{@TypeOf(self.data[0])});
            std.debug.print("\n  size:{}", .{self.size});
            std.debug.print("\n shape.len:{} shape: [ ", .{self.shape.len});
            for (0..self.shape.len) |i| {
                std.debug.print("{} ", .{self.shape[i]});
            }
            std.debug.print("] ", .{});
            //self.print();
        }

        /// Prints all the array self.data in an array.
        pub fn print(self: *@This()) void {
            std.debug.print("\n  tensor data: ", .{});
            for (0..self.size) |i| {
                std.debug.print("{} ", .{self.data[i]});
            }
            std.debug.print("\n", .{});
        }

        /// Print the Tensor() to console in a more readable way.
        pub fn printMultidim(self: *@This()) void {
            // Allocate array to store the indices
            self._printMultidimHelper(0, 0);
        }

        fn _printMultidimHelper(self: *@This(), offset: usize, idx: usize) void {
            // Print opening bracket with a number of tab that is equals to idx
            for (0..idx) |_| {
                std.debug.print("    ", .{});
            }
            std.debug.print("[", .{});

            if (idx == self.shape.len - 1) {
                for (0..self.shape[self.shape.len - 1]) |i| {
                    const local_idx = offset + i;
                    std.debug.print("{}, ", .{self.data[local_idx]});
                }
                std.debug.print("],\n", .{});
            } else {
                std.debug.print("\n", .{});
                for (0..self.shape[idx]) |i| {
                    self._printMultidimHelper(offset + self.shape[idx + 1] * i, idx + 1);
                }
                std.debug.print("\n", .{});

                for (0..idx) |_| {
                    std.debug.print("    ", .{});
                }
                std.debug.print("]", .{});
                if (idx != 0) {
                    std.debug.print(",\n", .{});
                }
            }
        }

        /// Set all tensor values to zero.
        pub fn setToZero(self: *@This()) !void {
            if (self.size == 0) {
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
        pub inline fn ensure_4D_shape(shape: []const usize) ![]usize {
            // The fixed dimension should be 4. Will updatein future
            // [batch, channel, row, column]
            const target_dims = 4;

            if (shape.len > target_dims) {
                return error.InvalidDimensions;
            }

            var padded_shape: [4]usize = .{ 1, 1, 1, 1 };

            // caulculate starting index to start
            const start_index = target_dims - shape.len;

            // copy values into last positions
            for (shape, 0..) |dim, i| {
                padded_shape[start_index + i] = dim;
            }

            return &padded_shape;
        }

        /// Bare metal version of tensor info that uses a logging function instead of std.debug.print
        pub fn info_metal(self: *@This()) void {
            if (log_function) |log| {
                var buffer: [512]u8 = undefined;

                // Log size
                if (std.fmt.bufPrint(&buffer, "Tensor size: {}\n", .{self.size})) |msg| {
                    log(@constCast(@ptrCast(&buffer[0..msg.len])));
                } else |_| return;

                // Log shape
                var shape_str: [256]u8 = undefined;
                var shape_pos: usize = 0;
                for (self.shape, 0..) |dim, i| {
                    if (i == 0) {
                        if (std.fmt.bufPrint(shape_str[shape_pos..], "{}", .{dim})) |msg| {
                            shape_pos += msg.len;
                        } else |_| return;
                    } else {
                        if (std.fmt.bufPrint(shape_str[shape_pos..], ", {}", .{dim})) |msg| {
                            shape_pos += msg.len;
                        } else |_| return;
                    }
                }
                if (std.fmt.bufPrint(&buffer, "Tensor shape: [{s}]\n", .{shape_str[0..shape_pos]})) |msg| {
                    log(@constCast(@ptrCast(&buffer[0..msg.len])));
                } else |_| return;

                // Log data
                var data_str: [256]u8 = undefined;
                var data_pos: usize = 0;
                const max_preview = @min(self.size, 4);
                for (0..max_preview) |i| {
                    if (i == 0) {
                        if (std.fmt.bufPrint(data_str[data_pos..], "{d:.2}", .{self.data[i]})) |msg| {
                            data_pos += msg.len;
                        } else |_| return;
                    } else {
                        if (std.fmt.bufPrint(data_str[data_pos..], ", {d:.2}", .{self.data[i]})) |msg| {
                            data_pos += msg.len;
                        } else |_| return;
                    }
                }
                if (self.size > max_preview) {
                    if (std.fmt.bufPrint(data_str[data_pos..], ", ...", .{})) |msg| {
                        data_pos += msg.len;
                    } else |_| return;
                }
                if (std.fmt.bufPrint(&buffer, "Tensor data: [{s}]\n", .{data_str[0..data_pos]})) |msg| {
                    log(@constCast(@ptrCast(&buffer[0..msg.len])));
                } else |_| return;
            }
        }
    };
}

// Helper functions for string conversion
fn intToString(value: usize, buffer: []u8) usize {
    if (value == 0) {
        buffer[0] = '0';
        return 1;
    }
    var n = value;
    var i: usize = 0;
    while (n > 0) : (n /= 10) {
        buffer[i] = @intCast('0' + @mod(n, 10));
        i += 1;
    }
    // Reverse the string
    var start: usize = 0;
    var end: usize = i - 1;
    while (start < end) {
        const temp = buffer[start];
        buffer[start] = buffer[end];
        buffer[end] = temp;
        start += 1;
        end -= 1;
    }
    return i;
}

fn floatToString(value: f32, buffer: []u8) usize {
    // Handle negative numbers
    var pos: usize = 0;
    if (value < 0) {
        buffer[pos] = '-';
        pos += 1;
    }

    // Convert integer part
    const abs_value = if (value < 0) -value else value;
    const int_part = @as(i32, @intFromFloat(abs_value));
    pos += intToString(@intCast(int_part), buffer[pos..]);

    // Add decimal point
    buffer[pos] = '.';
    pos += 1;

    // Convert decimal part (2 decimal places)
    const decimal_part = @as(i32, @intFromFloat((abs_value - @as(f32, @floatFromInt(int_part))) * 100.0));
    if (decimal_part < 10) {
        buffer[pos] = '0';
        pos += 1;
    }
    pos += intToString(@intCast(decimal_part), buffer[pos..]);

    return pos;
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
