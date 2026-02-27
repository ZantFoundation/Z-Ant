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
const std = @import("std");
const zant = @import("zant");

const allocator = zant.utils.allocator.allocator;
const Tensor = zant.core.Tensor;

//--- protos ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;

//--- IR Zant Library ---
pub const graphZant_lib = @import("graphZant.zig");
pub const GraphZant = graphZant_lib.GraphZant;

pub const NodeZant_lib = @import("nodeZant.zig");
pub const NodeZant = NodeZant_lib.NodeZant;

pub const tensorZant_lib = @import("tensorZant.zig");
pub const TensorZant = tensorZant_lib.TensorZant;
pub const TensorCategory = tensorZant_lib.TensorCategory;

pub const operators = @import("op_union/op_union.zig").operators;
pub const fused_operators = @import("op_union/op_union.zig").fused_operators;
pub const utils = @import("utils.zig");

pub const pattern_matcher = @import("fusion/pattern_matcher.zig");
pub const pattern_collection = @import("fusion/pattern_collection.zig");

pub fn init(modelProto: *ModelProto) !GraphZant {

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
