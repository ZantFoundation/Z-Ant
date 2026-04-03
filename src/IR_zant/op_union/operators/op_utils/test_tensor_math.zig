const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test {
    _ = @import("test_elementWise_math.zig");
    _ = @import("test_logical_math.zig");
    _ = @import("test_reduction_math.zig");
    _ = @import("../op_clip/test_clip.zig");
    _ = @import("../op_concat/test_concat.zig");
    // _ = @import("../op_conv/test_conv.zig"); Tests are obsolete! The ops are tested into the fuzzing
    _ = @import("../op_conv/test_conv_stm32n6.zig");
    _ = @import("test_conv_clip.zig");
    _ = @import("../op_flatten/test_flatten.zig");
    _ = @import("../op_gather/test_gather.zig");
    _ = @import("../op_gelu/test_gelu.zig");
    _ = @import("../op_gemm/test_gemm.zig");
    _ = @import("../op_identity/test_identity.zig");
    _ = @import("../op_log/test_log.zig");
    _ = @import("../op_matMul/test_matMul.zig");
    _ = @import("../op_reduceMean/test_reduceMean.zig");
    _ = @import("../op_min/test_min.zig");
    _ = @import("../op_neg/test_neg.zig");
    _ = @import("../op_pad/test_pad.zig");
    // _ = @import("../op_maxPool/test_pooling.zig"); Tests are obsolete! The ops are tested into the fuzzing
    _ = @import("../op_pow/test_pow.zig");
    _ = @import("../op_reshape/test_reshape.zig");
    _ = @import("../op_resize/test_resize.zig");
    _ = @import("../op_shape/test_shape.zig");
    _ = @import("../op_split/test_split.zig");
    _ = @import("../op_sqrt/test_sqrt.zig");
    _ = @import("../op_squeeze/test_squeeze.zig");
    _ = @import("../op_transpose/test_transpose.zig");
    _ = @import("../op_unsqueeze/test_unsqueeze.zig");
    _ = @import("test_other.zig");
}
