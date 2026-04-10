const std = @import("std");

comptime {
    _ = @import("Core/test_core.zig");
    _ = @import("Utils/test_utils.zig");
    _ = @import("IR_graph/IR_graph.zig");
    _ = @import("TensorToImage/test_TensorToImage.zig");
}
