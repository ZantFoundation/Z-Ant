const std = @import("std");
const model_opts = @import("model_opts"); // Importa il modulo aggiunto in build.zig
const main_log = std.debug.print; //std.log.scoped(.main).info;

// Declare the external C ABI function exported by the static library
// Assumes T = f32 based on build options
extern fn predict(
    input: [*]const f32, // Changed to const T
    input_shape: [*]const u32, // Changed to const u32
    shape_len: u32,
    result: *[*]f32, // Pointer to receive the output slice pointer
) i32;

fn prepareInputData(allocator: std.mem.Allocator) ![]model_opts.input_data_type {
    const shape = model_opts.input_shape;
    var total_size: usize = 1;
    inline for (shape) |dim| {
        total_size *= dim;
    }

    const data = try allocator.alloc(model_opts.input_data_type, total_size);
    errdefer allocator.free(data);
    @memset(data, 1);

    return data;
}

fn getPredictOutputSize() usize {
    return model_opts.output_data_len;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // main_log("Preparing input data...\\n", .{});
    const input_data = try prepareInputData(allocator);
    const input_shape = model_opts.input_shape;
    main_log("Model expects input shape: {any}\n", .{input_shape});
    main_log("Prepared input data length: {}\n", .{input_data.len});

    const output_size = getPredictOutputSize();
    main_log("Expected output size: {}\n", .{output_size});

    const start_time: i64 = std.time.microTimestamp();

    const cycles: i64 = 10;

    var pass: u8 = 1;
    while (pass <= cycles) : (pass += 1) {
        // main_log("\n=== PASSATA {} ===\n", .{pass});

        var output_ptr: [*]model_opts.output_data_type = undefined;

        // main_log("Calling predict (via model_opts.lib)...\\n", .{});

        const res = model_opts.lib.predict(
            input_data.ptr,
            @ptrCast(@constCast(&input_shape)),
            @intCast(input_shape.len),
            &output_ptr,
        );

        // main_log("Predict returned: {}\n", .{res});
        if (res != 0) {
            main_log("\n !!!! ERROR!!! Predict failed with code: {}\n", .{res});
            main_log("Error codes: 0=success, -1=math ops failed, -2=init failed, -3=output failed\n", .{});
            allocator.free(input_data);
            return;
        }
        // main_log("Predict call finished.\n", .{});

        // main_log("Output pointer address: 0x{x}\n", .{@intFromPtr(output_ptr)});

        // Check if output_ptr is null before creating slice
        if (@intFromPtr(output_ptr) == 0) {
            main_log("Error: predict returned a null pointer.\n", .{});
            allocator.free(input_data);
            return;
        }

        // // Test access to first element before creating full slice
        // main_log("Testing first element access...\n", .{});
        // const first_val = output_ptr[0];
        // main_log("First element: {d}\n", .{first_val});

        // const output_slice = @as([*]model_opts.output_data_type, @ptrCast(output_ptr))[0..output_size];

        // //print the output
        // main_log("Output (first {} elements):\n", .{@min(output_size, 10)});
        // var i: usize = 0;
        // while (i < output_slice.len and i < 10) : (i += 1) {
        //     main_log("{d}, ", .{output_slice[i]});
        // }
        // if (output_slice.len > 10) {
        //     main_log("...\n", .{});
        // } else {
        //     main_log("\n", .{});
        // }

        // main_log.warn("WARNING: Memory for the predict output was NOT freed!\n", .{});
        // main_log("Passata {} completata.\n", .{pass});
    }

    const end_time: i64 = std.time.microTimestamp();

    main_log("\n average inference time: {} us \n", .{@divFloor(end_time - start_time, cycles)});

    // main_log("Attempting to free input memory...\\n", .{});
    allocator.free(input_data);

    // main_log("Program finished.\\n", .{});
}
