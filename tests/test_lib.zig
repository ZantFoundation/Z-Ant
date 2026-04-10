const std = @import("std");

comptime {
    _ = @import("Core/test_core.zig");
    _ = @import("Utils/test_utils.zig");
    _ = @import("IR_graph/IR_graph.zig");
    _ = @import("TensorToImage/test_mtf.zig");
    _ = @import("TensorToImage/test_gasf.zig");
    _ = @import("TensorToImage/gadf/test_gadf.zig");
}
