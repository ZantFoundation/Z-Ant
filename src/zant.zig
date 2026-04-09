//! Root namespace for the Zant library.
//! Re-exports public sub-packages via the independent modules:
//! `core` (tensor operations), `onnx` (model parsing), `codegen` (.zig codegeneration)
//!
//! This module acts as a compatibility shim so that downstream code
//! (generated libraries, tests, mains) can still `@import("zant")`.
pub const core = @import("IR_zant").core;
pub const utils = @import("zant_utils");
pub const onnx = @import("onnx");
pub const codegen = @import("codegen");
