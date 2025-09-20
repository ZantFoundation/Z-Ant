pub const AutoPadMode = enum {
    notset,
    valid,
    same_upper,
    same_lower,
};

pub const ConvPreparedParams = struct {
    stride: [2]usize,
    dilations: [2]usize,
    pads: [4]usize,
    group: usize,
    filters_per_group: usize,
    channels_per_group: usize,
    auto_pad: AutoPadMode,
};
