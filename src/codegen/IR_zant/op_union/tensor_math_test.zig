const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

comptime {
    _ = @import("operators/op_utils/elementWise_math_test.zig");
    _ = @import("operators/op_utils/logical_math_test.zig");
    _ = @import("operators/op_utils/reduction_math_test.zig");
    _ = @import("operators/op_clip/clip_test.zig");
    _ = @import("operators/op_concat/concat_test.zig");
    _ = @import("operators/op_conv/conv_test.zig");
    _ = @import("operators/op_utils/conv_clip_test.zig");
    _ = @import("operators/op_flatten/flatten_test.zig");
    _ = @import("operators/op_gather/gather_test.zig");
    _ = @import("operators/op_gelu/gelu_test.zig");
    _ = @import("operators/op_gemm/gemm_test.zig");
    _ = @import("operators/op_identity/identity_test.zig");
    _ = @import("operators/op_log/log_test.zig");
    _ = @import("operators/op_matMul/matMul_test.zig");
    _ = @import("operators/op_reduceMean/reduceMean_test.zig");
    _ = @import("operators/op_min/min_test.zig");
    _ = @import("operators/op_neg/neg_test.zig");
    _ = @import("operators/op_pad/pad_test.zig");
    _ = @import("operators/op_maxPool/pooling_test.zig");
    _ = @import("operators/op_pow/pow_test.zig");
    _ = @import("operators/op_reshape/reshape_test.zig");
    _ = @import("operators/op_resize/resize_test.zig");
    _ = @import("operators/op_shape/shape_test.zig");
    _ = @import("operators/op_split/split_test.zig");
    _ = @import("operators/op_sqrt/sqrt_test.zig");
    _ = @import("operators/op_squeeze/squeeze_test.zig");
    _ = @import("operators/op_transpose/transpose_test.zig");
    _ = @import("operators/op_unsqueeze/unsqueeze_test.zig");
    _ = @import("operators/op_utils/other_test.zig");
}

