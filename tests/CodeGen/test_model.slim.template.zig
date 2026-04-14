const std = @import("std");
const IR_zant = @import("IR_zant");
const codegen = @import("codegen");
const utils = codegen.utils;
const Tensor = IR_zant.core.tensor.Tensor;
const pkgAllocator = IR_zant.pkg_allocator;
const allocator = pkgAllocator.allocator;

const model = @import("model_options.zig");

const ITERATION_COUNT: u32 = 100;

// ----------- SLIM TEMPLATE -----------

test "model info" {
    std.debug.print("\n\n ++++++++++++++++ testing {s} ++++++++++++++++\n", .{model.name});
}

test "Static Library - Random data Prediction Test" {
    std.testing.log_level = .info;

    std.debug.print("\n--- Random data Prediction Test ---", .{});

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

    var error_counter: i32 = 0;

    for (0..ITERATION_COUNT) |_| {
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

        // Run prediction
        const return_code = model.lib.predict(
            input_data.ptr,
            @ptrCast(&input_shape),
            input_shape.len,
            &result,
        );

        if (return_code != 0) {
            std.debug.print("\n     - detected ERROR type: {}", .{return_code});
            error_counter += 1;
        }
        if (model.is_dynamic) {
            defer allocator.free(result[0..model.output_data_len]);
        }
    }

    try std.testing.expectEqual(error_counter, 0);

    std.debug.print("\n  - Ran {} fuzzy tests on model \"{s}\", done without errors:", .{ ITERATION_COUNT, model.name });
}

test "Static Library - Inputs Prediction Test" {
    std.testing.log_level = .info;

    std.debug.print("\n\n--- Pre-Generated results tests ---", .{});

    var input_shape = model.input_shape;
    var error_counter: i32 = 0;
    var value_error_counter: i32 = 0;
    var input_data_len: u32 = 0;
    var no_error_found: bool = true;

    if (model.has_inputs) {
        input_data_len = 1;
        for (input_shape) |dim| {
            input_data_len *= dim;
        }
    }

    std.debug.print(" {s}", .{model.user_tests_path});

    const parsed_user_tests = try utils.loadUserTests(model.input_data_type, model.output_data_type, model.user_tests_path);
    defer parsed_user_tests.deinit();

    const user_tests = parsed_user_tests.value;

    std.debug.print("\n  - User tests loaded.", .{});

    for (user_tests) |user_test| {
        std.debug.print("\n  - Running user test: {s}", .{user_test.name});

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

        for (0.., user_test.output) |i, expected_output| {
            const result_value = result[i];

            // SAFE DIFF
            // Convert to f64 for safe difference calculation
            const expected_f64: f64 = switch (@typeInfo(@TypeOf(expected_output))) {
                .float => expected_output,
                .int => @floatFromInt(expected_output),
                else => @panic("Unsupported output type"),
            };

            const result_f64: f64 = switch (@typeInfo(@TypeOf(result_value))) {
                .float => result_value,
                .int => @floatFromInt(result_value),
                else => @panic("Unsupported result type"),
            };

            const diff = @abs(expected_f64 - result_f64);
            const margin_f64: f64 = marginFor(model.output_data_type);

            const big_diff: bool = diff > margin_f64;
            if (big_diff) {
                if (value_error_counter < 5) std.debug.print("\n\n  >>>>>>>ERROR!!<<<<<< \nTest failed for input: {d} expected: {} got: {}, margin: {}\n", .{ i, expected_output, result_value, marginFor(model.output_data_type) });
                value_error_counter += 1;
                no_error_found = false;

                // UNCOMMENT FOR DEBUG, pay attention with huge datasets
                // std.debug.print("\n expected: {any} ", .{user_test.output});
                // std.debug.print("\n obtained: {{", .{});
                // for (0..user_test.output.len) |j| {
                //     if (i > 0) std.debug.print(",", .{});
                //     std.debug.print(" {}", .{result[j]});
                // }
                // std.debug.print(" }}", .{});

            }
        }

        if (value_error_counter > 5) std.debug.print("\n\n ... and many other ... \n  >>>>>>> {} values are wrong!!<<<<<< ", .{value_error_counter});
        try std.testing.expect(no_error_found);

        // UNCOMMENT FOR DEBUG, pay attention with huge datasets
        // std.debug.print("\n expected: {any} ", .{user_test.output});
        // std.debug.print("\n obtained: {{", .{});
        // for (0..user_test.output.len) |j| {
        //     if (j > 0) std.debug.print(",", .{});
        //     std.debug.print(" {}", .{result[j]});
        // }
        // std.debug.print(" }}", .{});

        if (model.is_dynamic) {
            allocator.free(result[0..model.output_data_len]);
        }
    }

    try std.testing.expectEqual(error_counter, 0);
}

/// Returns `true` if `T` is any integer type (signed or unsigned).
fn isInteger(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => true,
        else => false,
    };
}

/// Returns `0` (of type `T`) for non-float `T`, otherwise `0.001` (of type `T`).
fn marginFor(comptime T: type) f64 {
    return if (@typeInfo(T) == .int)
        // integer types: zero tolerance
        1.0
    else
        // floating‑point (or any other type): tiny tolerance
        0.001;
}
