const std = @import("std");

pub const Testing_flags = struct {
    op_to_test_option: []const u8,

    pub fn init(b: *std.Build) !Testing_flags {
        return Testing_flags{
            .op_to_test_option = b.option([]const u8, "op", "operator name") orelse "all",
        };
    }
};
