const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running quant_tensor_math tests\n", .{});
}

test {
    _ = @import("../op_dequantize/dequantize_test.zig");
    _ = @import("../op_quantize/quantize_test.zig");
    _ = @import("../op_qlinearaveragepool/qlinear_pooling_test.zig");
    // _ = @import("../op_qlinearconv/qlinearconv_test.zig");
    _ = @import("../op_qlinearmatmul/qlinearmatmul_test.zig");
}
