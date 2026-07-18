const std = @import("std");

const ZantStepOptions = @import("zantStepOptions.zig").ZantStepOptions;
const ZantOptions = @import("zantOptions.zig").ZantOptions;
const ZantModules = @import("zantModules.zig").ZantModules;

pub const ZantBuild = struct {
    zantOptions: ZantOptions,
    zantStepOptions: ZantStepOptions,
    zantModules: ZantModules,

    pub fn init(b: *std.Build) !ZantBuild {
        // --------------------------------------------------------------
        // ------------------------ flag options ------------------------
        // --------------------------------------------------------------
        const zantOptions = try ZantOptions.init(b);

        // --------------------------------------------------------------
        // ------------------------ Step options ------------------------
        // --------------------------------------------------------------
        const zantStepOptions = try ZantStepOptions.init(b, zantOptions);

        // --------------------------------------------------------------
        // ---------------------- Modules creation ----------------------
        // --------------------------------------------------------------
        const zantModules = try ZantModules.init(b, zantStepOptions);

        return ZantBuild{
            .zantOptions = zantOptions,
            .zantStepOptions = zantStepOptions,
            .zantModules = zantModules,
        };
    }
};
