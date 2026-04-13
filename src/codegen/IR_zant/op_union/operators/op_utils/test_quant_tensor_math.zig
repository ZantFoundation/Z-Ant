const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running quant_tensor_math tests\n", .{});
}

test {
    _ = @import("../op_dequantize/test_dequantize.zig");
    _ = @import("../op_quantize/test_quantize.zig");
    _ = @import("../op_qlinearaveragepool/test_qlinear_pooling.zig");
    // _ = @import("../op_qlinearconv/test_qlinearconv.zig");
    _ = @import("../op_qlinearmatmul/test_qlinearmatmul.zig");
}
