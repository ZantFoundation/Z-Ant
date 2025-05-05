const std = @import("std");
const zant = @import("zant");
const codegen = @import("codegen");
const Renderer = codegen.renderer;
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;

const ZigRenderer = Renderer.ZigRenderer;
const lowerAdd = zant.core.tensor.math_standard.lowerAdd;
const UOpBuilder = zant.uops.UOpBuilder;
const DType = zant.uops.DType;

// Import the generated kernel file
//const kernel = @import("loweradd_output_function.zig");

// /* REMOVED OLD TESTS
// test "Arithmetic operations" { ... }
// ... etc ...
// test "Rendering memory operations" { ... }
// */

test "LowerAdd Pipeline" {
    std.debug.print("Running zig renderer lowerAdd pipeline test\n", .{});
    const allocator = std.testing.allocator;

    // 1. Setup UOpBuilder
    var builder = UOpBuilder.init(allocator);
    // No defer needed if we don't take ownership of slice

    // 2. Define inputs for lowerAdd (example shapes/strides)
    const A_id: usize = 0; // Simulated input tensor ID
    const B_id: usize = 1; // Simulated input tensor ID
    const out_shape = &.{ 2, 3 }; // Example output shape
    const strideA = &.{ 3, 1 }; // Example strides for A (row-major)
    const strideB = &.{ 0, 1 }; // Example strides for B (broadcast dim 0)
    const out_dtype = DType.f32;

    // 3. Call lowerAdd to generate UOps
    const out_buf_id = lowerAdd(
        &builder,
        A_id,
        B_id,
        out_shape,
        strideA,
        strideB,
        out_dtype,
    );
    _ = out_buf_id; // Prevent unused warning

    // Take ownership of UOps
    const uops_list = try builder.toOwnedSlice();
    // Deinit builder immediately as its list is now empty
    builder.deinit();
    // Defer freeing the main slice AFTER freeing internal src slices
    defer allocator.free(uops_list);

    // DEBUG: Print the generated UOps
    std.debug.print("--- Generated UOps ---\n", .{});
    for (uops_list) |uop| {
        uop.dump(std.io.getStdErr().writer()) catch {}; // Dump to stderr
    }
    std.debug.print("--------------------\n", .{});

    // Add defer to free duplicated src slices within the owned list
    defer {
        std.debug.print("DEBUG: Freeing internal src for {d} uops in test\n", .{uops_list.len});
        for (uops_list) |uop| {
            // Same check as in (fixed) builder deinit
            if (uop.src.len > 0) {
                // Maybe add ptr check here too if needed
                allocator.free(@constCast(uop.src));
            }
        }
    }

    // 4. Render UOps to Zig code as a function
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    const Writer = @TypeOf(buffer.writer());
    var renderer = ZigRenderer(Writer).init(allocator, buffer.writer());
    defer renderer.deinit(); // Deinit renderer AFTER use

    // Specify which IDs are inputs
    const input_ids = &[_]usize{ A_id, B_id };
    try renderer.render_as_function(uops_list, input_ids); // Call the new method

    const actual_code = try buffer.toOwnedSlice();
    defer allocator.free(actual_code);

    std.debug.print("\n--- Generated Function ---\n{s}\n---------------------\n", .{actual_code});

    // 5. Save output to a file
    const output_filename = "tests/CodeGen/renderer/loweradd_output_function.zig"; // Save inside tests dir
    var file = try std.fs.cwd().createFile(output_filename, .{ .read = true }); // Ensure write permissions
    defer file.close();
    _ = try file.write(actual_code);
    std.debug.print("Generated function saved to {s}\n", .{output_filename});

    // Optional: Clean up the generated file
    // try std.fs.cwd().deleteFile(output_filename);
}

// test "Test Generated LowerAdd Kernel" {
//     std.debug.print("Testing generated kernel from loweradd_output_function.zig\n", .{});
//     const allocator = std.testing.allocator;

//     // 1. Define input data
//     // Kernel expects input_0 and input_1, both []const f32
//     // Based on the loop (0..6), they should have 6 elements
//     const input_data_0 = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
//     const input_data_1 = [_]f32{ 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 };

//     // 2. Call the generated kernel
//     // IMPORTANT: Match the argument order from the generated function signature!
//     // pub fn generated_kernel(allocator: std.mem.Allocator, input_1: []const f32, input_0: []const f32)
//     const result_slice = try kernel.generated_kernel(allocator, &input_data_1, &input_data_0);
//     defer allocator.free(result_slice); // Kernel allocates the output slice

//     // 3. Define expected output
//     // result[i] = input_0[i] + input_1[i]
//     const expected_result = [_]f32{ 11.0, 13.0, 15.0, 17.0, 19.0, 21.0 };

//     // 4. Compare results
//     try std.testing.expectEqualSlices(f32, &expected_result, result_slice);

//     std.debug.print("Generated kernel test passed!\n", .{});
// }
