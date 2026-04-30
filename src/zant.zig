//! Root namespace for the Zant library.
//! Re-exports all public sub-packages: `core` (tensor operations),
//! `onnx` (model parsing), `utils` (allocator, errors, helpers), and
//! `ImageToTensor` (image decoding and preprocessing).
pub const core = @import("Core/core.zig");
pub const utils = @import("Utils/utils.zig");
pub const onnx = @import("onnx/onnx.zig");
// TODO: find a more appropriate place for this export
pub const TensorToImage = @import("TensorToImage/mod.zig");

// Force compilation of C-callable GAF API into any library that imports zant.
// Necessary because top-level `pub const` declarations are evaluated lazily:
// lib_ecg-classifier.zig never accesses zant.TensorToImage, so TensorToImage/
// would otherwise be skipped entirely. A comptime block is always executed
// when the file is analyzed, bypassing lazy evaluation.
comptime {
    _ = &TensorToImage.timeseries_to_image_api.zant_timeseries_to_image;
}
