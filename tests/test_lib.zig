const std = @import("std");

comptime {
    _ = @import("core/test_tensor.zig");
    _ = @import("Utils/test_utils.zig");
    _ = @import("IR_graph/IR_graph.zig");
}
