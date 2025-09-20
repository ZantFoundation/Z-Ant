const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen").codegen_v1;
const utils = codegen.utils;
const Tensor = zant.core.tensor.Tensor;
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const model = @import("model_options.zig");

// ----------- FULL TEMPLATE -----------

test "model info" {
    var buf: [256]u8 = undefined;
    if (model.name.len > buf.len) return error.ModelNameTooLong;

    // Fill the first `model.name.len` bytes with '+'
    @memset(buf[0..model.name.len], '+');

    std.debug.print("\n\n +++++++++++++++++++++++++++++++++++++++++++++++{s}", .{buf[0..model.name.len]});
    std.debug.print("\n++++++++++++++++++ testing {s} ++++++++++++++++++", .{model.name});
    std.debug.print("\n+++++++++++++++++++++++++++++++++++++++++++++++{s}", .{buf[0..model.name.len]});
}

test "Static Library - Random data Prediction Test" {
    std.testing.log_level = .info;
    std.debug.print("\n--- Random data Prediction Test ---", .{});

    // Handle nodes with no inputs (e.g., nodes that only process initializers)
    if (!model.has_inputs) {
        std.debug.print("\n--- Node has no inputs, testing output generation only ---", .{});

        // Allocate output array
        var output_data: [*]model.output_data_type = undefined;

        // Call predict with null inputs since this node doesn't use external inputs
        const result_code = model.lib.predict(
            undefined, // input: not used for no-input nodes
            undefined, // input_shape: not used
            0, // shape_len: 0 since no inputs
            &output_data,
        );

        std.debug.print("\n--- Prediction result code: {} ---", .{result_code});
        try std.testing.expect(result_code == 0);

        // Basic validation that output was generated
        std.debug.print("\n--- Output generated successfully ---", .{});
        return;
    }

    var input_shape = model.input_shape;

    var input_data_size: u32 = 1;
    for (input_shape) |dim| {
        input_data_size *= dim;
    }

    // Create input data array directly instead of ArrayList
    var input_data = try allocator.alloc(model.input_data_type, input_data_size);
    defer allocator.free(input_data);

    // Generate random data
    var seed: u64 = undefined;
    try std.posix.getrandom(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    // Fill with random values
    for (0..input_data_size) |i| {
        switch (@typeInfo(model.input_data_type)) {
            .float => {
                input_data[i] = rand.float(model.input_data_type) * 100;
            },
            .int => {
                if (model.input_data_type == u8) {
                    input_data[i] = rand.int(u8);
                } else if (model.input_data_type == i8) {
                    input_data[i] = rand.int(i8);
                } else if (model.input_data_type == u16) {
                    input_data[i] = rand.int(u16);
                } else if (model.input_data_type == i16) {
                    input_data[i] = rand.int(i16);
                } else if (model.input_data_type == u32) {
                    input_data[i] = rand.int(u32);
                } else if (model.input_data_type == i32) {
                    input_data[i] = rand.int(i32);
                } else {
                    input_data[i] = @intCast(rand.int(u32));
                }
            },
            else => {
                // Fallback for other types
                input_data[i] = 128; // Middle value for UINT8
            },
        }
    }

    var result: [*]model.output_data_type = undefined;

    if (model.have_log) {
        // Create a logging function
        const LogFn = fn ([*c]u8) callconv(.C) void;
        const logFn: LogFn = struct {
            fn log(msg: [*c]u8) callconv(.C) void {
                std.debug.print("{s}", .{msg});
            }
        }.log;

        // Set the logging function
        model.lib.setLogFunction(logFn);
    }

    // Run prediction
    const return_code = model.lib.predict(
        input_data.ptr,
        @ptrCast(&input_shape),
        input_shape.len,
        &result,
    );

    if (model.is_dynamic and return_code == 0) {
        defer allocator.free(result[0..model.output_data_len]);
    }

    try std.testing.expectEqual(0, return_code);
    std.debug.print("\nPrediction done without errors", .{});
}

test "Static Library - Wrong Input Shape" {
    std.testing.log_level = .info;

    std.debug.print("\n\n--- Wrong Input Shape test ---", .{});

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
    var input_data = try allocator.alloc(model.input_data_type, input_data_size);
    defer allocator.free(input_data);

    i = 0;
    while (i < input_data_size) : (i += 1) {
        input_data[i] = 1;
    }

    var result: [*]model.output_data_type = undefined;

    const return_code = model.lib.predict(
        @ptrCast(&input_data),
        @ptrCast(&input_shape.items),
        model_input_shape.len,
        &result,
    );

    // below not necessary, it fails before
    // if (model.is_dynamic) {
    //     defer allocator.free(result[0..model.output_data_len]);
    // }
    try std.testing.expectEqual(return_code, -2);
}

test "Static Library - Empty Input" {
    std.testing.log_level = .info;

    std.debug.print("\n\n--- Empty Input test ---", .{});

    // Test with empty input
    var input_data = [_]model.input_data_type{};
    var input_shape = [_]u32{};
    var result: [*]model.output_data_type = undefined;

    const return_code = model.lib.predict(
        @ptrCast(&input_data),
        @ptrCast(&input_shape),
        0,
        &result,
    );

    // below not necessary, it fails before
    // if (model.is_dynamic) {
    //     defer allocator.free(result[0..model.output_data_len]);
    // }
    try std.testing.expectEqual(return_code, -2);
}

test "Static Library - Wrong Number of Dimensions" {
    std.testing.log_level = .info;

    std.debug.print("\n\n--- Wrong Number of Dimensions test ---", .{});

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

    var result: [*]model.output_data_type = undefined;

    const return_code = model.lib.predict(
        @ptrCast(&input_data),
        @ptrCast(&input_shape),
        1,
        &result,
    );

    // below not necessary, it fails before
    // if (model.is_dynamic) {
    //     defer allocator.free(result[0..model.output_data_len]);
    // }
    try std.testing.expectEqual(return_code, -2);
}

test "Static Library - User data Prediction Test" {
    std.testing.log_level = .info;

    std.debug.print("\n\n--- User data Prediction test ---", .{});

    if (!model.user_tests) {
        std.debug.print("\nUser tests are disabled for this model\n", .{});
        return;
    }

    if (model.have_log) {
        // Create a logging function
        const LogFn = fn ([*c]u8) callconv(.C) void;
        const logFn: LogFn = struct {
            fn log(msg: [*c]u8) callconv(.C) void {
                std.debug.print("{s}", .{msg});
            }
        }.log;

        // Set the logging function
        model.lib.setLogFunction(logFn);
    }

    var input_shape = model.input_shape;
    var error_counter: i32 = 0;
    var input_data_len: u32 = 1;
    for (input_shape) |dim| {
        input_data_len *= dim;
    }

    const parsed_user_tests = try utils.loadUserTests(model.input_data_type, model.output_data_type, model.user_tests_path);
    defer parsed_user_tests.deinit();

    const user_tests = parsed_user_tests.value;

    std.debug.print("\nUser tests loaded.\n", .{});

    for (user_tests) |user_test| {
        std.debug.print("\n\tRunning user test: {s}\n\n", .{user_test.name});

        try std.testing.expectEqual(user_test.input.len, input_data_len);

        var result: [*]model.output_data_type = undefined;

        // Run prediction
        const return_code = model.lib.predict(
            user_test.input.ptr,
            @ptrCast(&input_shape),
            input_shape.len,
            &result,
        );

        if (return_code != 0) {
            std.debug.print("\n     - detected ERROR type: {}", .{return_code});
            error_counter += 1;
        }

        if (std.mem.eql(u8, user_test.type, "classify")) {
            var max_value: model.input_data_type = 0;
            // Find the class with maximum value
            if (model.output_data_type == f32 or model.output_data_type == f64) {
                max_value = std.math.floatMin(model.output_data_type);
            } else {
                max_value = std.math.minInt(model.output_data_type);
            }

            var max_index: usize = 0;
            for (0..model.output_data_len) |i| {
                const value = result[i];
                if (value > max_value) {
                    max_value = value;
                    max_index = i;
                }
            }

            try std.testing.expectEqual(user_test.expected_class, max_index);
        } else if (std.mem.eql(u8, user_test.type, "regress")) {
            for (0.., user_test.output) |i, expected_output| {
                const result_value = result[i];
                const expected_output_value = expected_output;
                try std.testing.expectEqual(expected_output_value, result_value);
            }
        } else if (std.mem.eql(u8, user_test.type, "regress")) {
            // TODO: Calculate some sort of delta to compare the result with the expected output
            for (0.., user_test.output) |i, expected_output| {
                const result_value = result[i];
                const expected_output_value = expected_output;
                try std.testing.expectEqual(expected_output_value, result_value);
            }
        } else if (std.mem.eql(u8, user_test.type, "exact")) {
            for (0.., user_test.output) |i, expected_output| {
                const result_value = result[i];
                const big_diff: bool = @abs(expected_output - result_value) > marginFor(model.output_data_type);
                if (big_diff) {
                    std.debug.print("\n\n  >>>>>>>ERROR!!<<<<<< \nTest failed for input: {d} expected: {} got: {}, margin: {}\n", .{ i, expected_output, result_value, marginFor(model.output_data_type) });
                }
            }
        } else {
            std.debug.print("Unsupported test type: {s}\n", .{user_test.type});
            try std.testing.expect(false);
        }

        if (model.is_dynamic) {
            allocator.free(result[0..model.output_data_len]);
        }
    }

    try std.testing.expectEqual(error_counter, 0);
}

/// Returns `0` (of type `T`) for non-float `T`, otherwise `0.001` (of type `T`).
fn marginFor(comptime T: type) T {
    return if (@typeInfo(T) == .int)
        // integer types: zero tolerance
        0
    else
        // floating‑point (or any other type): tiny tolerance
        0.001;
}
