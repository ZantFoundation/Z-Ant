//! Root namespace for the Zant library.
//! Re-exports all public sub-packages: `core` (tensor operations),
//! `onnx` (model parsing), `utils` (allocator, errors, helpers), and
//! `ImageToTensor` (image decoding and preprocessing).
pub const core = struct {
    pub const tensor = @import("core/tensor.zig");
};
pub const utils = @import("Utils/utils.zig");
pub const onnx = @import("onnx/onnx.zig");
