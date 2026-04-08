//! Code-generation entry point for Zant.
//!
//! Provides two back-ends:
//! - `codegen_v1` (active): translates a `GraphZant` into a self-contained C/Arduino
//!   inference library (`.ino`, `.h`, and parameter files). Entry function: `codeGenerateFromOnnx`.
//! - `codegen_v2` (experimental, not yet fully functional): alternative renderer-based pipeline.
//!
//! Typical usage: call `codegen_v1.codeGenerateFromOnnx(model_name, output_path, model)`.
const std = @import("std");
const zant = @import("zant");
const IR = @import("IR_zant");

pub const codegen_v1 = @import("cg_v1/codegen_v1.zig");
pub const codegen_v1_exe = @import("cg_v1/main.zig");

// --- importing codegen options
pub const codegen_options = @import("codegen_options");
