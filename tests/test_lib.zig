const std = @import("std");

comptime {
    _ = @import("Core/Tensor/TensorMath/test_op_convolution.zig");
    _ = @import("Core/test_core.zig");
    _ = @import("DataHandler/test_dataHandler.zig");
    _ = @import("Model/test_model.zig");
    _ = @import("Utils/test_utils.zig");
}
