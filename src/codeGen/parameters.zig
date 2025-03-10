const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen.zig");
const utils = codegen.utils;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const TensorProto = onnx.TensorProto;
const DataType = onnx.DataType;
const globals = codegen.globals;

/// Writes the Zig code required to initialize all tensor initializers in the ONNX model.
/// This function generates declarations and definitions for each tensor.
///
/// - `writer`: The file writer to output generated code.
/// - `model`: The ONNX model containing tensor initializers.
pub inline fn write_parameters(writer: std.fs.File.Writer, model: ModelOnnx) !void {

    //importing the libraries
    try write_libraries_parameters(writer);

    try writer.print(
        \\
        \\
        \\ // ---------------------------------------------------
        \\ // +         Initializing Weights and Biases         +
        \\ // ---------------------------------------------------
    , .{});

    // Iterate over all initializers in the ONNX model and generate code
    for (model.graph.?.initializers) |tensorProtoInitializer| {
        const dataTypeString: []const u8 = try utils.getTypeString(tensorProtoInitializer.data_type);
        const name: []const u8 = try utils.getSanitizedName(tensorProtoInitializer.name.?);

        try writer.print(
            \\
            \\
            \\ // ----------- Initializing tensor_{s};
        , .{name});

        // Generate the shape array for the tensor
        try wrtiteTensorShape(writer, tensorProtoInitializer, name);

        // Generate the data array for the tensor
        try writeArray(writer, tensorProtoInitializer, name);

        // Create the tensor instance
        try writer.print(
            \\
            \\pub const tensor_{s} = Tensor({s}).fromConstBuffer(&allocator, &array_{s}, &shape_tensor_{s});
        , .{ name, dataTypeString, name, name });
    }
}

/// Writes the required library imports to the generated Zig file for input tensor.
///
/// This function ensures that the necessary standard and package libraries are
/// imported into the generated Zig source file.
///
/// # Parameters
/// - `writer`: A file writer used to write the import statements.
///
/// # Errors
/// This function may return an error if writing to the file fails.
fn write_libraries_parameters(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\ const std = @import("std");
        \\ const zant = @import("zant");
        \\ const Tensor = zant.core.tensor.Tensor;
        \\ const pkgAllocator = zant.utils.allocator;
        \\ const allocator = pkgAllocator.allocator;
        \\
    , .{});
}

/// Writes the shape array for a tensor initializer.
///
/// - `writer`: The file writer to output generated code.
/// - `t`: The tensor initializer.
/// - `name`: The sanitized name of the tensor.
pub inline fn wrtiteTensorShape(writer: std.fs.File.Writer, t: *TensorProto, name: []const u8) !void {
    try writer.print(
        \\
        \\
        \\const shape_tensor_{s} : [{}]usize = [_]usize{{ 
    , .{ name, t.dims.len });

    for (0..t.dims.len) |i| {
        if (i > 0) try writer.print(
            \\, 
        , .{});

        try writer.print(
            \\ {}
        , .{t.dims[i]});
    }

    try writer.print(
        \\}} ;
    , .{});
}

/// Writes the array for a tensor initializer based on its data type.
///
/// - `writer`: The file writer to output generated code.
/// - `t`: The tensor initializer.
/// - `name`: The sanitized name of the tensor.
pub inline fn writeArray(writer: std.fs.File.Writer, t: *TensorProto, name: []const u8) !void {
    const dataTypeString: []const u8 = try utils.getTypeString(t.data_type);

    var size: i64 = 1;
    for (t.dims) |dims_i| {
        size *= dims_i;
    }
    try writer.print(
        \\
        \\const array_{s} : [{d}]{s} = [_]{s}{{ 
    , .{ name, size, dataTypeString, dataTypeString });

    // Select appropriate data storage format
    if (t.float_data) |d| {
        writeArrayData(writer, f32, d) catch return error.f32DataUnavailable;
    } else if (t.raw_data) |d| {
        writeArrayRawData(writer, t.data_type, d) catch return error.u8RawDataUnavailable;
    } else if (t.int32_data) |d| {
        writeArrayData(writer, i32, d) catch return error.i32DataUnavailable;
    } else if (t.int64_data) |d| {
        writeArrayData(writer, i64, d) catch return error.i64DataUnavailable;
    } else if (t.double_data) |d| {
        writeArrayData(writer, f64, d) catch return error.f64DataUnavailable;
    } else if (t.uint64_data) |d| {
        writeArrayData(writer, u64, d) catch return error.u64DataUnavailable;
    } else return error.DataTypeNotAvailable;

    try writer.print(
        \\}} ;
    , .{});
}

/// Converts raw binary tensor data into typed array representation.
///
/// - `writer`: The file writer to output generated code.
/// - `data_type`: The ONNX data type.
/// - `data`: The raw binary tensor data.
pub inline fn writeArrayRawData(writer: std.fs.File.Writer, data_type: DataType, data: []const u8) !void {
    switch (data_type) {
        .FLOAT => {
            const float_slice = @as([*]const f32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            try writeArrayData(writer, f32, float_slice);
        },
        .UINT8 => {
            const uint_slice = @as([*]const u8, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 1)];
            try writeArrayData(writer, u8, uint_slice);
        },
        .INT8 => {
            const int_slice = @as([*]const i8, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 1)];
            try writeArrayData(writer, i8, int_slice);
        },
        .UINT16 => {
            const uint_slice = @as([*]const u16, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 2)];
            try writeArrayData(writer, u16, uint_slice);
        },
        .INT16 => {
            const int_slice = @as([*]const i16, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 2)];
            try writeArrayData(writer, i16, int_slice);
        },
        .INT32 => {
            const int_slice = @as([*]const i32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            try writeArrayData(writer, i32, int_slice);
        },
        .INT64 => {
            const int_slice = @as([*]const i64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
            try writeArrayData(writer, i64, int_slice);
        },
        .FLOAT16 => {
            const float_slice = @as([*]const f16, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 2)];
            try writeArrayData(writer, f16, float_slice);
        },
        .DOUBLE => {
            const double_slice = @as([*]const f64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
            try writeArrayData(writer, f64, double_slice);
        },
        .UINT32 => {
            const uint_slice = @as([*]const u32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            try writeArrayData(writer, u32, uint_slice);
        },
        .UINT64 => {
            const uint_slice = @as([*]const u64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
            try writeArrayData(writer, u64, uint_slice);
        },
        else => {
            std.debug.print("\n Data type {s} not supported for raw data access \n", .{@tagName(data_type)});
        },
    }
}

/// Writes an array of tensor data.
///
/// - `writer`: The file writer to output generated code.
/// - `T`: The type of data in the tensor.
/// - `data`: The data array.
pub inline fn writeArrayData(writer: std.fs.File.Writer, comptime T: type, data: []const T) !void {
    try writer.print(
        \\{}
    , .{data[0]});
    for (1..data.len) |i| {
        try writer.print(
            \\, {}
        , .{data[i]});
    }
}
