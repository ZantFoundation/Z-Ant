//! Cross-cutting utilities shared by all Zant packages.
//! - `allocator`: project-wide allocator facade (swappable via build options).
pub const allocator = @import("allocator.zig");
