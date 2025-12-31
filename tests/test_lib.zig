const std = @import("std");
const test_options = @import("test_options");
const test_name = test_options.test_name;

comptime {
    _ = @import("Core/test_core.zig");
    _ = @import("Utils/test_utils.zig");
    _ = @import("ImageToTensor/test_imgToTensor.zig");
    _ = @import("IR_graph/IR_graph.zig");
}
