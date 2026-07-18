const std = @import("std");

const testing = @import("testing_flags.zig");
const codegen = @import("codegen_flags.zig");

pub const ZantOptions = struct {
    testing_flags: testing.Testing_flags,
    codegen_flags: codegen.Codegen_flags,

    pub fn init(b: *std.Build) !ZantOptions {
        return ZantOptions{
            .testing_flags = try testing.Testing_flags.init(b),
            .codegen_flags = try codegen.Codegen_flags.init(b),
        };
    }
};
