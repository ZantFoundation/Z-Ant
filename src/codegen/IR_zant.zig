//! Zant Intermediate Representation (IR).
//!
//! Converts a parsed ONNX `ModelProto` into Zant's own computation graph,
//! which is used by the code-generator. The IR consists of three node types:
//! - `GraphZant`: directed acyclic graph of `NodeZant` instances; supports
//!   linearisation, DAG validation, and kernel fusion.
//! - `NodeZant`: a single computation node wrapping an `Op_union` (one of the
//!   ~60 supported ONNX operators or fused variants).
//! - `TensorZant`: metadata record for a tensor (name, type, shape, category).
//!
//! Entry point: call `IR_zant.init(modelProto)` to build the graph.
//! The `fusion` sub-package contains the generic pattern-matcher used to fold
//! operator sequences (e.g. Conv+Relu) into a single fused node before codegen.

// === Module-level imports (from build system) ===
pub const onnx = @import("IR_zant/onnx.zig");
pub const zant_utils = @import("zant_utils");
pub const pkg_allocator = zant_utils.allocator;


// === Internal sub-packages ===
pub const core = struct {
    pub const tensor = @import("IR_zant/core/tensor.zig");
    pub const Tensor = tensor.Tensor;
    pub const math_standard = @import("IR_zant/op_union/operators/zant_math_standard.zig");
};

// --- IR Zant Library ---
pub const graphZant_lib = @import("IR_zant/graphZant.zig");
pub const GraphZant = graphZant_lib.GraphZant;

pub const NodeZant_lib = @import("IR_zant/nodeZant.zig");
pub const NodeZant = NodeZant_lib.NodeZant;

pub const tensorZant_lib = @import("IR_zant/tensorZant.zig");
pub const TensorZant = tensorZant_lib.TensorZant;
pub const TensorCategory = tensorZant_lib.TensorCategory;

pub const op_union_lib = @import("IR_zant/op_union/op_union.zig");
pub const Op_union = op_union_lib.Op_union;
pub const operators = op_union_lib.operators;
pub const fused_operators = op_union_lib.fused_operators;
pub const utils = @import("IR_zant/utils.zig");

pub const pattern_matcher = @import("IR_zant/fusion/pattern_matcher.zig");
pub const pattern_collection = @import("IR_zant/fusion/pattern_collection.zig");

pub fn init(modelProto: *onnx.ModelProto) !GraphZant {

    //initialize the tensor hash map
    try tensorZant_lib.initialize_tensorZantMap(modelProto);

    //check
    const graphProto = try if (modelProto.graph) |graph| graph else error.GraphNotAvailable;

    //initialize the graphZant
    var graphZant = try GraphZant.init(graphProto);

    //constructing the graph
    try graphZant.build_graph();

    return graphZant;
}
