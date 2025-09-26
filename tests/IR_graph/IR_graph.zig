const std = @import("std");

test {
    std.testing.log_level = .info;
    comptime {
        _ = @import("tensorZant.zig");
        _ = @import("graph_init.zig");
        _ = @import("linearization.zig");
        // _ = @import("test_all_write_op.zig");
        _ = @import("utils.zig");
        _ = @import("fused_dequant_clip_quant.zig");
    }
}
