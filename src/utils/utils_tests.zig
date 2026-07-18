//! Test entry point for the `zant_utils` module.
//! Aggregates every *_test.zig file in this module.
const std = @import("std");

comptime {
    // No tests yet. Add `_ = @import("<name>_test.zig");` as they land.
}

test "zant_utils tests entry" {
    std.debug.print("\n--- Running zant_utils tests\n", .{});
}
