const std = @import("std");

pub const Cmsis_flags = struct {
    enable_cmsis: bool,
    //cmsis_path: ?[]const u8,

    pub fn init(b: *std.Build) !Cmsis_flags {
        return Cmsis_flags{
            .enable_cmsis = b.option(bool, "enable_CMSIS", "Enable CMSIS support") orelse false,
            // .cmsis_path = b.option([]const u8, "cmsis_path", "Optional CMSIS include path") orelse "",
        };
    }
};
