//! Test entry point for the `codegen` module.
//! Aggregates every *_test.zig file in this module.
const std = @import("std");

comptime {
    // No tests yet. Add `_ = @import("codegen/<name>_test.zig");` as they land.
}

test "codegen tests entry" {
    std.debug.print("\n--- Running codegen tests\n", .{});
}
