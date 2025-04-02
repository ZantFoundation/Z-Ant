const std = @import("std");

test {
    _ = @import("Tensor/test_tensor.zig");
    _ = @import("Quantization/test_clustering.zig");
}
