const std = @import("std");
const protobuf = @import("protobuf.zig");

/// This file defines a unified error set for ONNX parsing operations.
///
/// Why is this necessary?
/// - AttributeProto can contain a GraphProto (tag 6: 'g' field)
/// - GraphProto contains NodeProto instances
/// - NodeProto contains AttributeProto instances
///
/// This creates a recursive dependency chain:
///   AttributeProto.parse() -> GraphProto.parse() -> NodeProto.parse() -> AttributeProto.parse()
///
/// Zig's compiler cannot infer error sets for recursive function calls,
/// so we must explicitly define a shared error set for all parsing functions.
///
/// Without this, you'll get: "error: unable to resolve inferred error set"
pub const ParseError = protobuf.Error || std.mem.Allocator.Error || error{
    TagNotAvailable,
    UnknownOperator,
    InvalidAttributeType,
    MissingRequiredField,
};
