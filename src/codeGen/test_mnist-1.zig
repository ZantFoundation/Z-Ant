const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const model = @import("model_options.zig");

test "Static Library - Random data Prediction Test" {
    std.debug.print("\n     test: Static Library - {s} Random data Prediction Test\n", .{model.name});

    var input_shape = model.input_shape;

    var input_data_size: u32 = 1;
    for (input_shape) |dim| {
        input_data_size *= dim;
    }

    // Create a zeroed array based on dynamic input_data_size
    var input_data = std.ArrayList(f32).init(allocator);
    defer input_data.deinit();

    try input_data.resize(input_data_size);

    // Generate random data from std.rand.defaultPrng
    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.posix.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rand = prng.random();

    // Iterate input_data_size times and add random values
    var i: u32 = 0;
    while (i < input_data_size) : (i += 1) {
        const value = rand.float(f32) * 100.0;
        try input_data.append(value);
    }

    // TODO: Figure out how to get output data length from the model
    var result_buffer: [10]f32 = undefined; // Allocate buffer for 10 MNIST classes
    var result: [*]f32 = @ptrCast(&result_buffer);

    // Create a logging function
    const LogFn = fn ([*c]u8) callconv(.C) void;
    const logFn: LogFn = struct {
        fn log(msg: [*c]u8) callconv(.C) void {
            std.debug.print("{s}", .{msg});
        }
    }.log;

    // Set the logging function
    model.lib.setLogFunction(logFn);

    // Run prediction
    model.lib.predict(
        @ptrCast(&input_data.items),
        @ptrCast(&input_shape),
        input_shape.len, // 4D tensor shape
        &result,
    );

    std.debug.print("\nPrediction done without erorrs:\n", .{});
}

test "Static Library - Wrong Input Shape" {
    std.debug.print("\n     test: Static Library - {s} Wrong Input Shape\n", .{model.name});

    // Test with wrong input shape

    // Generate random input shape based on the model input shape length

    const model_input_shape = model.input_shape;

    var input_data_size: u32 = 1;
    for (model_input_shape) |dim| {
        input_data_size *= dim;
    }

    // Create a zeroed array based on dynamic input_data_size
    var input_shape = std.ArrayList(u32).init(allocator);
    defer input_shape.deinit();

    input_data_size = 1;

    try input_shape.resize(model_input_shape.len);

    var i: u32 = 0;
    while (i < model_input_shape.len) : (i += 1) {
        const value = model_input_shape[i] + 1;
        input_data_size *= value;
        try input_shape.append(value);
    }

    // Init array with only ones with dynamic input_data_size
    var input_data = try allocator.alloc(f32, input_data_size);
    defer allocator.free(input_data);

    i = 0;
    while (i < input_data_size) : (i += 1) {
        input_data[i] = 1.0;
    }

    var result: [*]f32 = undefined;

    model.lib.predict(
        @ptrCast(&input_data),
        @ptrCast(&input_shape.items),
        model_input_shape.len,
        &result,
    );
}

test "Static Library - Empty Input" {
    std.debug.print("\n     test: Static Library - {s} Empty Input\n", .{model.name});

    // Test with empty input
    var input_data = [_]f32{};
    var input_shape = [_]u32{};
    var result: [*]f32 = undefined;

    model.lib.predict(
        @ptrCast(&input_data),
        @ptrCast(&input_shape),
        0,
        &result,
    );
}

test "Static Library - Wrong Number of Dimensions" {
    std.debug.print("\n     test: Static Library - {s} Wrong Number of Dimensions\n", .{model.name});

    const model_input_shape = model.input_shape;

    var input_data_size: u32 = 1;
    for (model_input_shape) |dim| {
        input_data_size *= dim;
    }

    // Test with wrong number of dimensions
    
    var input_shape = [_]u32{input_data_size}; // Should be 4D but only 1D

    var input_data = try allocator.alloc(f32, input_data_size);
    defer allocator.free(input_data);

    var i: u32 = 0;
    while (i < input_data_size) : (i += 1) {
        input_data[i] = 1.0;
    }

    var result: [*]f32 = undefined;

    model.lib.predict(
        @ptrCast(&input_data),
        @ptrCast(&input_shape),
        1,
        &result,
    );
}

// TODO: Implement the following tests

// test "Static Library - User Prediction Test" {
//     std.debug.print("\n     test: Static Library - {s} User Prediction Test\n", .{model.name});

//     var input_shape = model.input_shape;

//     var input_data_size: u32 = 1;
//     for (input_shape) |dim| {
//         input_data_size *= dim;
//     }

//     // Create a zeroed array based on dynamic input_data_size
//     var input_data = std.ArrayList(f32).init(allocator);
//     defer input_data.deinit();

//     try input_data.resize(input_data_size);

//     // Generate random data from std.rand.defaultPrng
//     var prng = std.rand.DefaultPrng.init(blk: {
//         var seed: u64 = undefined;
//         try std.posix.getrandom(std.mem.asBytes(&seed));
//         break :blk seed;
//     });
//     const rand = prng.random();

//     // Iterate input_data_size times and add random values
//     var i: u32 = 0;
//     while (i < input_data_size) : (i += 1) {
//         const value = rand.float(f32) * 100.0;
//         try input_data.append(value);
//     }

//     // TODO: Figure out how to get output data length from the model
//     var result_buffer: [10]f32 = undefined; // Allocate buffer for 10 MNIST classes
//     var result: [*]f32 = @ptrCast(&result_buffer);

//     // Create a logging function
//     const LogFn = fn ([*c]u8) callconv(.C) void;
//     const logFn: LogFn = struct {
//         fn log(msg: [*c]u8) callconv(.C) void {
//             std.debug.print("{s}", .{msg});
//         }
//     }.log;

//     // Set the logging function
//     model.lib.setLogFunction(logFn);

//     // Run prediction
//     model.lib.predict(
//         @ptrCast(&input_data.items),
//         @ptrCast(&input_shape),
//         input_shape.len, // 4D tensor shape
//         &result,
//     );

//     std.debug.print("\nPrediction results:\n", .{});
//     std.debug.print("{} ", .{result[0]});
// }
