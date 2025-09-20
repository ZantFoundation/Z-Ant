const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;
const DataType = onnx.DataType;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;
const allocator = zant.utils.allocator.allocator;

const IR_zant = @import("IR_zant");
const IR_graph = IR_zant.IR_graph;
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = IR_zant.TensorZant;

const testWriter = @import("tests_writer.zig");

// -------------------- GETTERS --------------------

//Given an element from DataType Enum in onnx.zig returns the equivalent zig type
pub inline fn getType(data_type: DataType) !type {
    switch (data_type) {
        .FLOAT => {
            return f32;
        },
        .UINT8 => {
            return u8;
        },
        .INT8 => {
            return i8;
        },
        .UINT16 => {
            return u16;
        },
        .INT16 => {
            return i16;
        },
        .INT32 => {
            return i32;
        },
        .INT64 => {
            return i64;
        },
        .FLOAT16 => {
            return f16;
        },
        .DOUBLE => {
            return f64;
        },
        .UNIT32 => {
            return u32;
        },
        .UINT64 => {
            return u64;
        },
        else => return error.DataTypeNotAvailable,
    }
}

//Given an element from DataType Enum in onnx.zig returns the equivalent string of a zig type
pub inline fn getTypeString(data_type: DataType) ![]const u8 {
    switch (data_type) {
        .FLOAT => {
            return "f32";
        },
        .UINT8 => {
            return "u8";
        },
        .INT8 => {
            return "i8";
        },
        .UINT16 => {
            return "u16";
        },
        .INT16 => {
            return "i16";
        },
        .INT32 => {
            return "i32";
        },
        .INT64 => {
            return "i64";
        },
        .FLOAT16 => {
            return "f16";
        },
        .DOUBLE => {
            return "f64";
        },
        .UINT32 => {
            return "u32";
        },
        .UINT64 => {
            return "u64";
        },
        else => return error.DataTypeNotAvailable,
    }
}

//Returns the sanitized tensor's name, removes all non alphanumeric chars
pub inline fn getSanitizedName(name: []const u8) ![]const u8 {
    var sanitized = try allocator.alloc(u8, name.len);

    for (name, 0..) |char, i| {
        sanitized[i] = if (std.ascii.isAlphanumeric(char) or char == '_')
            std.ascii.toLower(char)
        else
            '_';
    }

    //std.log.debug("\nfrom {s} to {s} ", .{ name, sanitized });

    return sanitized;
}

pub inline fn getConstantTensorDims(nodeProto: *NodeProto) ![]const i64 {
    //check the node is a Constant
    if (std.mem.indexOf(u8, try getSanitizedName(nodeProto.op_type), "constant")) |_| {} else return error.NodeNotConstant;

    return if (nodeProto.attribute[0].t) |tensorProto| tensorProto.dims else error.ConstantTensorAttributeNotAvailable;
}

// ----------------- DATA TYPE management -------------

pub inline fn i64SliceToUsizeSlice(input: []const i64) ![]usize {
    var output = try allocator.alloc(usize, input.len);

    const maxUsize = std.math.maxInt(usize);

    for (input, 0..) |value, index| {
        if (value < 0) {
            return error.NegativeValue;
        }
        if (value > maxUsize) {
            return error.ValueTooLarge;
        }
        output[index] = @intCast(value);
    }

    return output;
}

pub fn usizeSliceToI64Slice(input: []usize) ![]const i64 {
    var output = try allocator.alloc(i64, input.len);

    for (input, 0..) |value, index| {
        if (value > std.math.maxInt(i64)) {
            return error.ValueTooLarge;
        }
        output[index] = @intCast(value);
    }

    return output;
}

/// Converts any integer value to usize with proper bounds checking
/// Returns error.NegativeValue if the input is negative (for signed types)
/// Returns error.ValueTooLarge if the input exceeds the maximum usize value
pub inline fn toUsize(comptime T: type, value: T) !usize {
    // Ensure T is an integer type
    comptime {
        if (@typeInfo(T) != .Int) {
            @compileError("toUsize only supports integer types");
        }
    }

    // Check for negative values if T is signed
    if (@typeInfo(T).Int.signedness == .signed and value < 0) {
        return error.NegativeValue;
    }

    // Check if value exceeds maximum usize
    const maxUsize = std.math.maxInt(usize);
    if (@as(u128, @intCast(if (@typeInfo(T).Int.signedness == .signed) @as(u128, @intCast(@max(0, value))) else @as(u128, @intCast(value)))) > maxUsize) {
        return error.ValueTooLarge;
    }

    return @intCast(value);
}

pub inline fn sliceToUsizeSlice(this_allocator: std.mem.Allocator, slice: anytype) []usize {
    const T = @TypeOf(slice);
    const info = @typeInfo(T);

    switch (info) {
        .pointer => {
            const child = info.pointer.child;
            const child_info = @typeInfo(child);

            var output = this_allocator.alloc(usize, slice.len) catch @panic("Out of memory in sliceToUsizeSlice");
            const maxUsize = std.math.maxInt(usize);

            for (slice, 0..) |value, index| {
                if (child_info == .int) {
                    // Handle integer types
                    if (value < 0) {
                        if (value == -1) {
                            output[index] = std.math.maxInt(usize);
                        } else {
                            @panic("Invalid negative value in sliceToUsizeSlice (only -1 is allowed)");
                        }
                    } else {
                        if (@as(u128, @intCast(value)) > maxUsize) {
                            @panic("Value too large in sliceToUsizeSlice");
                        }
                        output[index] = @intCast(value);
                    }
                } else if (child_info == .float) {
                    // Handle float types
                    if (value < 0) {
                        if (value == -1.0) {
                            output[index] = std.math.maxInt(usize);
                        } else {
                            @panic("Invalid negative value in sliceToUsizeSlice (only -1 is allowed)");
                        }
                    } else {
                        if (value > @as(f64, @floatFromInt(maxUsize))) {
                            @panic("Value too large in sliceToUsizeSlice");
                        }
                        output[index] = @intFromFloat(value);
                    }
                } else {
                    @compileError("Unsupported element type for sliceToUsizeSlice: " ++ @typeName(child));
                }
            }

            return output;
        },
        else => {
            @compileError("Unsupported type for sliceToUsizeSlice: " ++ @typeName(T));
        },
    }
}

// Modify signature to accept allocator
pub inline fn sliceToIsizeSlice(alloc: std.mem.Allocator, slice: anytype) []isize {
    const T = @TypeOf(slice);
    const info = @typeInfo(T);

    switch (info) {
        .pointer => {
            const child = info.pointer.child;
            const child_info = @typeInfo(child);

            // Use the passed allocator
            var output = alloc.alloc(isize, slice.len) catch @panic("Out of memory in sliceToIsizeSlice");
            const maxIsize = std.math.maxInt(isize);
            const minIsize = std.math.minInt(isize);

            for (slice, 0..) |value, index| {
                if (child_info == .int) {
                    // Handle integer types
                    if (value < minIsize or value > maxIsize) {
                        @panic("Value out of isize range in sliceToIsizeSlice");
                    }
                    output[index] = @intCast(value);
                } else if (child_info == .float) {
                    // Handle float types
                    if (value < @as(f64, @floatFromInt(minIsize)) or value > @as(f64, @floatFromInt(maxIsize))) {
                        @panic("Value out of isize range in sliceToIsizeSlice");
                    }
                    output[index] = @intFromFloat(value);
                } else {
                    @compileError("Unsupported element type for sliceToIsizeSlice: " ++ @typeName(child));
                }
            }

            return output;
        },
        else => {
            @compileError("Unsupported type for sliceToIsizeSlice: " ++ @typeName(T));
        },
    }
}

pub fn i64ToI64ArrayString(values: []const i64) ![]const u8 {
    var buffer: [20]u8 = undefined;
    var res_string = try std.mem.concat(allocator, u8, &[_][]const u8{"&[_]i64{"});
    for (values, 0..) |val, i| {
        if (i > 0) res_string = try std.mem.concat(allocator, u8, &[_][]const u8{ res_string, "," });
        const val_string = std.fmt.bufPrint(&buffer, "{}", .{val}) catch unreachable;
        res_string = try std.mem.concat(allocator, u8, &[_][]const u8{ res_string, val_string });
    }
    res_string = try std.mem.concat(allocator, u8, &[_][]const u8{ res_string, "}" });

    return res_string;
}

pub fn u32ToUsize(alloc: std.mem.Allocator, input: [*]u32, size: u32) ![]usize {
    var output = try alloc.alloc(usize, size);

    const maxUsize = std.math.maxInt(usize);

    for (0..size) |i| {
        if (input[i] < 0) {
            return error.NegativeValue;
        }
        if (input[i] > maxUsize) {
            return error.ValueTooLarge;
        }
        output[i] = @intCast(input[i]);
    }

    return output;
}

pub fn parseNumbers(input: []const u8) ![]i64 {
    var list = std.ArrayList(i64).init(allocator);
    errdefer list.deinit();

    if (input.len == 0) return list.toOwnedSlice();

    var it = std.mem.splitScalar(u8, input, ',');
    while (it.next()) |num_str| {
        const num = try std.fmt.parseInt(i64, num_str, 10);
        try list.append(num);
    }

    return list.toOwnedSlice();
}

pub fn i64SliceToUsizeArrayString(values: []const i64) ![]const u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit(); // Frees all memory

    try list.appendSlice("&[_]usize{");
    for (values, 0..) |val, i| {
        if (i > 0) try list.append(',');
        try list.writer().print("{}", .{val});
    }
    try list.append('}');

    return try list.toOwnedSlice(); // Caller must free this!
}

/// Parses a raw byte slice (expected to be little-endian) into an allocated slice of i64.
pub fn parseI64RawData(raw_data: []const u8) ![]i64 {
    const element_size = @sizeOf(i64);
    if (raw_data.len % element_size != 0) {
        std.log.warn("ERROR: Raw data length ({}) is not a multiple of i64 size ({})\n", .{ raw_data.len, element_size });
        return error.InvalidRawDataSize;
    }

    const num_elements = raw_data.len / element_size;
    if (num_elements == 0) {
        // Return an empty slice if raw_data is empty (and length is valid multiple of 0)
        return try allocator.alloc(i64, 0);
    }

    // Allocate the result slice.
    const result = try allocator.alloc(i64, num_elements);
    errdefer allocator.free(result);

    // Fallback: Use pointer casting to interpret raw bytes as i64 (assumes alignment and little-endian)
    // Ensure alignment (optional, might panic on some archs if unaligned)
    // if (@alignOf(i64) > @alignOf(u8) and @ptrToInt(raw_data.ptr) % @alignOf(i64) != 0) {
    //     std.log.warn("ERROR: Raw data pointer is not aligned for i64 read.\n", .{});
    //     return error.UnalignedRawData;
    // }

    // Cast the byte slice pointer to an i64 slice pointer
    const i64_ptr: [*]const i64 = @ptrCast(@alignCast(raw_data.ptr));

    // Copy the data from the cast pointer into the result slice
    @memcpy(result, i64_ptr[0..num_elements]);

    return result;
}

// ----------------- FILE MANAGEMENT -----------------
// Copy file from src to dst
pub fn copyFile(src_path: []const u8, dst_path: []const u8) !void {
    var src_file = try std.fs.cwd().openFile(src_path, .{});
    defer src_file.close();

    var dst_file = try std.fs.cwd().createFile(dst_path, .{});
    defer dst_file.close();

    // Use a buffer to copy in chunks
    var buf: [4096]u8 = undefined;
    while (true) {
        const bytes_read = try src_file.read(&buf);
        if (bytes_read == 0) break;
        _ = try dst_file.write(buf[0..bytes_read]);
    }
}

// Read the user_tests json file and return a list of test cases
pub fn loadUserTests(comptime T_in: type, comptime T_out: type, user_tests_path: []const u8) !std.json.Parsed([]testWriter.UserTest(T_in, T_out)) {
    const user_tests_file = try std.fs.cwd().openFile(user_tests_path, .{});
    defer user_tests_file.close();

    const user_tests_content: []const u8 = try user_tests_file.readToEndAlloc(allocator, 20 * 1024 * 1024);
    defer allocator.free(user_tests_content);

    const parsed_user_tests = try std.json.parseFromSlice([]testWriter.UserTest(T_in, T_out), allocator, user_tests_content, .{});

    return parsed_user_tests;
}

fn getTensorRef(tensor: *TensorZant) ![]u8 {
    const name = try tensor.getNameSanitized();
    if (tensor.tc == tensorZant_lib.TensorCategory.INITIALIZER) {
        return try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{name});
    } else {
        return try std.fmt.allocPrint(allocator, "@constCast(&tensor_{s})", .{name});
    }
}
