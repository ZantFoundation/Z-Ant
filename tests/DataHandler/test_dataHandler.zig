const std = @import("std");
const test_options = @import("test_options");

test {
    _ = @import("test_dataLoader.zig");
    _ = @import("test_dataProcessor.zig");

    if (test_options.heavy) {
        _ = @import("test_trainer.zig");
    }
}
