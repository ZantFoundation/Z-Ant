//! Test entry point for the `IR_zant` module.
//! Aggregates every *_test.zig file in this module.
const std = @import("std");

comptime {
    // --- core ---
    _ = @import("IR_zant/core/tensor_test.zig");

    // --- IR graph ---
    _ = @import("IR_zant/tensorZant_test.zig");
    _ = @import("IR_zant/graphZant_test.zig");
    _ = @import("IR_zant/linearization_test.zig");
    _ = @import("IR_zant/utils_test.zig");
    // write_op_test depends on local model files; opt-in only.
    // _ = @import("IR_zant/write_op_test.zig");

    // --- ONNX parser ---
    // onnx_loader_test is run via the dedicated `zig build onnx-parser` step
    // (it has pre-existing leaks that block aggregation with other tests).
    // _ = @import("IR_zant/onnx/onnx_loader_test.zig");

    // --- operators (see op_union/tensor_math_test.zig for the full list) ---
    _ = @import("IR_zant/op_union/tensor_math_test.zig");
}

test "IR_zant tests entry" {
    std.debug.print("\n--- Running IR_zant tests\n", .{});
}
