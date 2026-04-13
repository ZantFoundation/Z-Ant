const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test {
    _ = @import("operators/op_utils/test_elementWise_math.zig");
    _ = @import("operators/op_utils/test_logical_math.zig");
    _ = @import("operators/op_utils/test_reduction_math.zig");
    _ = @import("operators/op_clip/test_clip.zig");
    _ = @import("operators/op_concat/test_concat.zig");
    // _ = @import("operators/op_conv/test_conv.zig"); Tests are obsolete! The ops are tested into the fuzzing
    _ = @import("operators/op_utils/test_conv_clip.zig");
    _ = @import("operators/op_flatten/test_flatten.zig");
    _ = @import("operators/op_gather/test_gather.zig");
    _ = @import("operators/op_gelu/test_gelu.zig");
    _ = @import("operators/op_gemm/test_gemm.zig");
    _ = @import("operators/op_identity/test_identity.zig");
    _ = @import("operators/op_log/test_log.zig");
    _ = @import("operators/op_matMul/test_matMul.zig");
    _ = @import("operators/op_reduceMean/test_reduceMean.zig");
    _ = @import("operators/op_min/test_min.zig");
    _ = @import("operators/op_neg/test_neg.zig");
    _ = @import("operators/op_pad/test_pad.zig");
    // _ = @import("operators/op_maxPool/test_pooling.zig"); Tests are obsolete! The ops are tested into the fuzzing
    _ = @import("operators/op_pow/test_pow.zig");
    _ = @import("operators/op_reshape/test_reshape.zig");
    _ = @import("operators/op_resize/test_resize.zig");
    _ = @import("operators/op_shape/test_shape.zig");
    _ = @import("operators/op_split/test_split.zig");
    _ = @import("operators/op_sqrt/test_sqrt.zig");
    _ = @import("operators/op_squeeze/test_squeeze.zig");
    _ = @import("operators/op_transpose/test_transpose.zig");
    _ = @import("operators/op_unsqueeze/test_unsqueeze.zig");
    _ = @import("operators/op_utils/test_other.zig");
}

test {
    std.debug.print("\n--- Running tensor core tests\n", .{});
    _ = @import("core_tests");
}
