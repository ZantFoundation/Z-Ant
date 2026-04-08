//! Cross-cutting utilities shared by all Zant packages.
//! - `allocator`: project-wide allocator facade (swappable via build options).
//! - `type_converter`: helpers for converting between Zig and ONNX data types.
pub const allocator = @import("allocator.zig");
pub const type_converter = @import("typeConverter.zig");
//pub const time_keeper = @import("timeKeeper.zig");
