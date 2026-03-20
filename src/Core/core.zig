//! Core runtime library for Zant.
//! Exposes the `tensor` module, which provides the generic `Tensor(T)` type,
//! its standard and quantized math operations, and hardware accelerator backends.
pub const tensor = @import("Tensor/tensor.zig");

// pub const clustering = @import("Quantization/clustering.zig");
// pub const clustering_debug = @import("Quantization/clustering_debug.zig");
