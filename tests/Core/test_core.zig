const std = @import("std");

test {
    _ = @import("../../src/core/test_tensor.zig");
    _ = @import("../../src/IR_zant/op_union/operators/op_utils/test_conv_relu.zig");
}
