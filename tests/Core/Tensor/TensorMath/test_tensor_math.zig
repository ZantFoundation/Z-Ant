const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test {
    _ = @import("test_lib_elementWise_math.zig");
    _ = @import("test_lib_logical_math.zig");
    _ = @import("test_lib_reduction_math.zig");
    _ = @import("test_lib_shape_math.zig");
    _ = @import("test_op_mat_mul.zig");
    _ = @import("test_op_gemm.zig");
    _ = @import("test_op_qlinearconv.zig");
    _ = @import("test_op_convolution_stm32n6.zig");
    // _ = @import("test_op_pooling.zig");  Tests are obsolete! The ops are tested into the fuzzing
    // _ = @import("test_op_convolution.zig"); Tests are obsolete! The ops are tested into the fuzzing
    _ = @import("test_op_mean.zig");
}
