const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test {
    _ = @import("test_lib_elementWise_math.zig");
    _ = @import("test_lib_logical_math.zig");
    _ = @import("test_lib_reduction_math.zig");
    _ = @import("test_lib_shape_math.zig");
    _ = @import("test_op_slice.zig");
    _ = @import("test_op_mat_mul.zig");
    _ = @import("test_op_gemm.zig");
    _ = @import("test_op_pooling.zig");
    _ = @import("test_op_convolution.zig");
}
