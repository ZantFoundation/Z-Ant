const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
pub const AnyTensor = zant.core.tensor.AnyTensor;

const utils = @import("utils.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;
const ValueInfoProto = onnx.ValueInfoProto;

const allocator = std.heap.page_allocator;

pub const TensorType = enum {
    // Floating point types
    f16,
    f32,
    f64,

    // Signed integer types
    i4,
    i8,
    i16,
    i32,
    i64,

    // Unsigned integer types
    u4,
    u8,
    u16,
    u32,
    u64,

    // Boolean (often used for masks)
    bool,

    undefined,

    pub fn fromType(comptime T: type) TensorType {
        return switch (T) {
            f16 => TensorType.f16,
            f32 => TensorType.f32,
            f64 => TensorType.f64,
            i4 => TensorType.i4,
            i8 => TensorType.i8,
            i16 => TensorType.i16,
            i32 => TensorType.i32,
            i64 => TensorType.i64,
            u4 => TensorType.u4,
            u8 => TensorType.u8,
            u16 => TensorType.u16,
            u32 => TensorType.u32,
            u64 => TensorType.u64,
            bool => TensorType.bool,
            else => TensorType.undefined,
        };
    }

    pub fn toString(self: TensorType) []const u8 {
        return switch (self) {
            .f16 => "f16",
            .f32 => "f32",
            .f64 => "f64",
            .i4 => "i4",
            .i8 => "i8",
            .i16 => "i16",
            .i32 => "i32",
            .i64 => "i64",
            .u4 => "u4",
            .u8 => "u8",
            .u16 => "u16",
            .u32 => "u32",
            .u64 => "u64",
            .bool => "bool",
            .undefined => "undefined",
        };
    }
};

pub const TensorCategory = enum {
    INPUT,
    OUTPUT,
    INITIALIZER,
    LINK,
    CONSTANT,

    pub fn toString(self: TensorCategory) []const u8 {
        return switch (self) {
            .INPUT => ".INPUT",
            .OUTPUT => ".OUTPUT",
            .INITIALIZER => ".INITIALIZER",
            .LINK => ".LINK",
            .CONSTANT => ".CONSTANT",
        };
    }
};

pub const TensorZant = struct {
    name: []const u8,
    ty: TensorType,
    tc: TensorCategory,
    ptr: ?*AnyTensor,
    shape: []usize,
    stride: []usize,

    pub fn init(name: []const u8, tensorProto: ?*TensorProto, value_info: ?*ValueInfoProto, shape: ?[]usize, tensorCategory: TensorCategory) !TensorZant {
        // std.debug.print("\n ----------- init({s}, {s}, {s}, {s}, {s}) ", .{ name, if (tensorProto) |_| "tp" else "null", if (value_info) |_| "vi" else "null", if (shape) |_| "shape" else "null", tensorCategory.toString() });

        var tensor: ?*AnyTensor = null;
        var shape_i64: []i64 = undefined;
        var shape_usize: []usize = undefined;
        var ty: TensorType = TensorType.undefined; // Initialize to undefined instead of leaving uninitialized

        if (tensorProto) |tp| { //if the tensorProto is given it means that the tensor we are initializing is
            tensor = try allocator.create(AnyTensor);
            tensor.?.* = try utils.protoTensor2AnyTensor(tp); //create the ptr to AnyTensor
            shape_usize = tensor.?.get_shape(); //saves the shape
            ty = utils.getAnyTensorType(tensor.?.*);
        } else if (value_info) |vi| {
            if (utils.getTensorShapeFromValueInfo(vi)) |s| {
                shape_i64 = s;
                shape_usize = try utils.i64SliceToUsizeSlice(shape_i64); //saves the shape
                ty = try utils.getTypeFromValueInfo(vi);
            } else {
                // ValueInfo exists but no shape - fall through to placeholder handling
                std.debug.print("\n WARNING: {s} value_info exists but shape not found, using placeholder ", .{name});
                shape_usize = try allocator.alloc(usize, 1);
                shape_usize[0] = 1; // Default placeholder shape
                ty = try utils.getTypeFromValueInfo(vi); // Still try to get type
            }
        } else if (shape) |s| {
            shape_usize = s;
            // ty remains TensorType.undefined for this case
        } else {
            std.debug.print("\n ERROR: {s} not found ", .{name});
            return error.shapeNotfound;
        }

        // std.debug.print("\n                shape:{any} ", .{shape_usize});
        // std.debug.print("\n                type:{s} ", .{ty.toString()});

        return TensorZant{
            .name = name,
            .ty = ty, //if .ty is set to undefined it means that it is a "link" tensor between 2 nodes, the .ty must be set when the nodes are created
            .tc = tensorCategory,
            .ptr = tensor,
            .shape = shape_usize,
            .stride = try TensorZant.computeStride(shape_usize),
        };
    }

    pub fn deint(self: *TensorZant) void {
        if (self.ptr) |at| {
            at.deint();
        }

        allocator.free(self.ptr);
        allocator.free(self.name);
        allocator.free(self.shape);
        allocator.free(self.stride);
    }

    pub fn getNameSanitized(self: *TensorZant) ![]const u8 {
        var sanitized = try allocator.alloc(u8, self.name.len);

        for (self.name, 0..) |char, i| {
            sanitized[i] = if (std.ascii.isAlphanumeric(char) or char == '_')
                std.ascii.toLower(char)
            else
                '_';
        }

        return sanitized;
    }

    pub fn getShape(self: *TensorZant) []usize {
        return self.shape;
    }

    pub fn getStride(self: *TensorZant) []usize {
        return self.stride;
    }

    pub fn getSize(self: *TensorZant) usize {
        var size: usize = 1;
        for (self.shape) |dim| {
            size *= dim;
        }
        return size;
    }

    pub fn computeStride(shape: []usize) ![]usize {
        const num_dims = shape.len;
        var strides = try allocator.alloc(usize, num_dims);

        // Handle scalar case (0 dimensions)
        if (num_dims == 0) return strides;

        var i: usize = num_dims - 1;
        strides[i] = 1;

        while (i > 0) {
            strides[i - 1] = strides[i] * shape[i];
            i -= 1;
        }
        return strides;
    }

    pub fn set_tensorType(self: *TensorZant, ty: TensorType) void {
        if (ty == TensorType.undefined) {
            std.debug.print("\n ERROR: illegal behavior! you cannot set a tensor type to undefined! ", .{});
            return error.illegalBehavior;
        }
        self.ty = ty;
    }

    // Returns the id of a tensorZant from the hashMap
    pub fn get_tensorZantID(self: *TensorZant) usize {
        var hasher = std.hash.Wyhash.init(0);
        std.hash.autoHashStrat(&hasher, self.name, .Deep);
        return @as(usize, @intCast(hasher.final()));
    }
};

// ----------------------- HASH MAP -----------------------
pub var tensorMap: std.StringHashMap(TensorZant) = std.StringHashMap(TensorZant).init(allocator);

// Returns the id of a tensorZant from the hashMap
pub fn get_tensorZantID(self: TensorZant) u64 {
    var hasher = std.hash.Wyhash.init(0);
    std.hash.hash(&hasher, self.name, .Deep);
    return hasher.final();
}

// Populates tensorHashMap with the tensors used in the onnx graph, where the key is the name of the tensor
pub fn initialize_tensorZantMap(modelProto: *ModelProto) !void {
    //free and reinit the tensorMap
    tensorMap.deinit();
    tensorMap = std.StringHashMap(TensorZant).init(allocator);

    // std.debug.print("\n ---- initialize_tensorZantMap ---- ", .{});

    const protoGraph = try if (modelProto.graph) |graph| graph else error.GraphNotAvailable;

    //adding initializers to the hash map
    // std.debug.print("\n -------- initializers ", .{});

    //initializer : *TensorProto,
    for (protoGraph.initializers) |init_ptr| {
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            init_ptr.name.?,
            init_ptr,
            null,
            null,
            TensorCategory.INITIALIZER,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding inputs to the hash map
    // std.debug.print("\n -------- inputs: {d} ", .{protoGraph.inputs.len});

    for (protoGraph.inputs) |inputs_ptr| { //inputs : *ValueInfoProto,
        if (tensorMap.getPtr(inputs_ptr.name.?) != null) continue;
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            inputs_ptr.name.?,
            null,
            inputs_ptr,
            null,
            TensorCategory.INPUT,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding outputs to the hash map
    // std.debug.print("\n -------- outputs: {d}", .{protoGraph.outputs.len});

    for (protoGraph.outputs) |outputs_ptr| { //outputs : *ValueInfoProto,
        if (tensorMap.getPtr(outputs_ptr.name.?) != null) continue;
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            outputs_ptr.name.?,
            null,
            outputs_ptr,
            null,
            TensorCategory.OUTPUT,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    // std.debug.print("\n -------- nodes: {d}", .{protoGraph.nodes.len});
    //adding all the nodes inputs and outputs
    for (protoGraph.nodes) |node| { //for each NodeProto in the GraphProto

        // std.debug.print("\n --- {} :  {s} - {any} ", .{ i, node.op_type, node.name });
        // node.print(null); //DEBUG

        //WHy CONSTANT nodes need a different initialization? because is has many different variants and is hard to generalize
        if (std.mem.eql(u8, node.op_type, "Constant")) {
            const tensorZant: TensorZant = try TensorZant.init(
                node.output[0],
                node.attribute[0].t.?,
                null,
                null,
                TensorCategory.CONSTANT,
            );
            //add the readyTensor to the HashMap
            try tensorMap.put(tensorZant.name, tensorZant);
        } else {
            for (node.input) |input_name| {
                // std.debug.print("\n    inputs >>> {s}", .{input_name});
                if (tensorMap.getPtr(input_name) != null) continue;

                //if the tensor is null is represented by an empty string in the onnx, so It must not be initialized in the hashMap
                if (std.mem.eql(u8, input_name, "")) continue;
                //if the tensor already exists is means it is an onnx_initializer and it don't need to be initialized again
                if (tensorMap.getPtr(input_name) != null) continue;

                //create the readyTensor
                const tensorZant: TensorZant = try TensorZant.init(
                    input_name,
                    null,
                    utils.getValueInfoTensorFromGraphInfo(input_name, protoGraph),
                    null,
                    TensorCategory.LINK,
                );
                //add the readyTensor to the HashMap
                try tensorMap.put(tensorZant.name, tensorZant);
            }
            for (node.output) |output_name| {
                // std.debug.print("\n    >>> outputs {s}", .{output_name});
                if (tensorMap.getPtr(output_name) != null) continue;

                //WHy RESHAPE nodes need a different initialization? Because the output shape is sometime specified in the attributes, sometime is passed as an initializer and sometimes is a ValueInfoProto
                if (std.mem.eql(u8, node.op_type, "Reshape")) {

                    // ------------------ is it a ValueInfoProto? most probable option
                    if (utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph)) |vip_tensor| {
                        const tensorZant: TensorZant = try TensorZant.init(
                            output_name,
                            null,
                            vip_tensor,
                            null,
                            TensorCategory.LINK,
                        );
                        //add the readyTensor to the HashMap
                        try tensorMap.put(tensorZant.name, tensorZant);

                        continue;
                    }

                    // ------------------ is it passed as an input to the node? most probable option
                    //TODO
                    // ------------------ is it an initializer? it shoul be already initialized
                    //TODO

                    return error.Reshape_outputShape_NotFound;
                } else if (std.mem.eql(u8, node.op_type, "QLinearAveragePool")) {
                    // Special handling for QLinearAveragePool when value_info is missing
                    if (utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph)) |vip_tensor| {
                        const tensorZant: TensorZant = try TensorZant.init(
                            output_name,
                            null,
                            vip_tensor,
                            null,
                            TensorCategory.LINK,
                        );
                        try tensorMap.put(tensorZant.name, tensorZant);
                        continue;
                    } else {
                        // Calculate shape for QLinearAveragePool manually
                        if (utils.calculateQLinearAveragePoolShape(node, tensorMap)) |calculated_shape| {
                            const tensorZant: TensorZant = try TensorZant.init(
                                output_name,
                                null,
                                null,
                                calculated_shape,
                                TensorCategory.LINK,
                            );
                            try tensorMap.put(tensorZant.name, tensorZant);
                            continue;
                        } else {
                            std.debug.print("\n ERROR: Could not calculate shape for QLinearAveragePool {s}", .{output_name});
                            return error.QLinearAveragePool_shapeCalculationFailed;
                        }
                    }
                } else if (std.mem.eql(u8, node.op_type, "QLinearConcat")) {
                    // Special handling for QLinearConcat when value_info is missing
                    if (utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph)) |vip_tensor| {
                        const tensorZant: TensorZant = try TensorZant.init(
                            output_name,
                            null,
                            vip_tensor,
                            null,
                            TensorCategory.LINK,
                        );
                        try tensorMap.put(tensorZant.name, tensorZant);
                        continue;
                    } else {
                        // Calculate shape for QLinearConcat manually
                        if (utils.calculateQLinearConcatShape(node, tensorMap)) |calculated_shape| {
                            const tensorZant: TensorZant = try TensorZant.init(
                                output_name,
                                null,
                                null,
                                calculated_shape,
                                TensorCategory.LINK,
                            );
                            try tensorMap.put(tensorZant.name, tensorZant);
                            continue;
                        } else {
                            std.debug.print("\n ERROR: Could not calculate shape for QLinearConcat {s}", .{output_name});
                            return error.QLinearConcat_shapeCalculationFailed;
                        }
                    }
                } else if (std.mem.startsWith(u8, node.op_type, "QLinear")) {
                    // Generic handling for any QLinear operation when value_info is missing
                    if (utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph)) |vip_tensor| {
                        const tensorZant: TensorZant = try TensorZant.init(
                            output_name,
                            null,
                            vip_tensor,
                            null,
                            TensorCategory.LINK,
                        );
                        try tensorMap.put(tensorZant.name, tensorZant);
                        continue;
                    } else {
                        // For QLinear ops without value_info, create with undefined type and let the op handle shape inference
                        // std.debug.print("\n  +++ (QLinear without value_info)", .{});
                        // Set default shape based on first input
                        var default_shape: ?[]usize = null;
                        if (node.input.len > 0) {
                            // For most QLinear ops, first input is the data tensor
                            const first_input_name = node.input[0];
                            if (tensorMap.getPtr(first_input_name)) |first_tensor| {
                                const input_shape = first_tensor.getShape();
                                default_shape = allocator.alloc(usize, input_shape.len) catch null;
                                if (default_shape) |shape| {
                                    // Special case for QLinearGlobalAveragePool
                                    if (std.mem.eql(u8, node.op_type, "QLinearGlobalAveragePool")) {
                                        // Global average pool reduces spatial dimensions to 1x1
                                        for (input_shape, 0..) |dim, idx| {
                                            if (idx < 2) {
                                                shape[idx] = dim; // Keep batch and channel dimensions
                                            } else {
                                                shape[idx] = 1; // Reduce spatial dimensions to 1
                                            }
                                        }
                                    } else {
                                        for (input_shape, 0..) |dim, idx| {
                                            shape[idx] = dim;
                                        }
                                    }
                                }
                            }
                        }

                        const tensorZant: TensorZant = try TensorZant.init(
                            output_name,
                            null,
                            null,
                            default_shape,
                            TensorCategory.LINK,
                        );
                        try tensorMap.put(tensorZant.name, tensorZant);
                        continue;
                    }
                } else {
                    // std.debug.print("\n  +++", .{});

                    //if the tensor is null is represented by an empty string in the onnx, so It must not be initialized in the hashMap
                    if (std.mem.eql(u8, output_name, "")) continue;
                    //if the tensor already exists is means it is an onnx_initializer and it don't need to be initialized again
                    if (tensorMap.getPtr(output_name) != null) continue;
                    //create the readyTensor
                    const tensorZant: TensorZant = try TensorZant.init(
                        output_name,
                        null,
                        utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph),
                        null,
                        TensorCategory.LINK,
                    );
                    //add the readyTensor to the HashMap
                    try tensorMap.put(tensorZant.name, tensorZant);
                }
            }
        }
    }
}

// First pass: Initialize all tensors (same as original code)
fn initializeAllTensors(protoGraph: *GraphProto) !void {
    //adding initializers to the hash map
    // std.debug.print("\n -------- initializers ", .{});

    //initializer : *TensorProto,
    for (protoGraph.initializers) |init_ptr| {
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            init_ptr.name.?,
            init_ptr,
            null,
            null,
            TensorCategory.INITIALIZER,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding inputs to the hash map
    // std.debug.print("\n -------- inputs: {d} ", .{protoGraph.inputs.len});

    for (protoGraph.inputs) |inputs_ptr| { //inputs : *ValueInfoProto,
        if (tensorMap.getPtr(inputs_ptr.name.?) != null) continue;
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            inputs_ptr.name.?,
            null,
            inputs_ptr,
            null,
            TensorCategory.INPUT,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    //adding outputs to the hash map
    // std.debug.print("\n -------- outputs: {d}", .{protoGraph.outputs.len});

    for (protoGraph.outputs) |outputs_ptr| { //outputs : *ValueInfoProto,
        if (tensorMap.getPtr(outputs_ptr.name.?) != null) continue;
        //create the readyTensor
        const tensorZant: TensorZant = try TensorZant.init(
            outputs_ptr.name.?,
            null,
            outputs_ptr,
            null,
            TensorCategory.OUTPUT,
        );
        //add the readyTensor to the HashMap
        try tensorMap.put(tensorZant.name, tensorZant);
    }

    // std.debug.print("\n -------- nodes: {d}", .{protoGraph.nodes.len});
    //adding all the nodes inputs and outputs
    for (protoGraph.nodes) |node| { //for each NodeProto in the GraphProto

        // std.debug.print("\n --- {} :  {s} - {any} ", .{ i, node.op_type, node.name });
        // node.print(null); //DEBUG

        //WHy CONSTANT nodes need a different initialization? because is has many different variants and is hard to generalize
        if (std.mem.eql(u8, node.op_type, "Constant")) {
            const tensorZant: TensorZant = try TensorZant.init(
                node.output[0],
                node.attribute[0].t.?,
                null,
                null,
                TensorCategory.CONSTANT,
            );
            //add the readyTensor to the HashMap
            try tensorMap.put(tensorZant.name, tensorZant);
        } else {
            for (node.input) |input_name| {
                // std.debug.print("\n    inputs >>> {s}", .{input_name});
                if (tensorMap.getPtr(input_name) != null) continue;

                //if the tensor is null is represented by an empty string in the onnx, so It must not be initialized in the hashMap
                if (std.mem.eql(u8, input_name, "")) continue;
                //if the tensor already exists is means it is an onnx_initializer and it don't need to be initialized again
                if (tensorMap.getPtr(input_name) != null) continue;

                //create the readyTensor
                const tensorZant: TensorZant = try TensorZant.init(
                    input_name,
                    null,
                    utils.getValueInfoTensorFromGraphInfo(input_name, protoGraph),
                    null,
                    TensorCategory.LINK,
                );
                //add the readyTensor to the HashMap
                try tensorMap.put(tensorZant.name, tensorZant);
            }
            for (node.output) |output_name| {
                // std.debug.print("\n    >>> outputs {s}", .{output_name});
                if (tensorMap.getPtr(output_name) != null) continue;

                //WHy RESHAPE nodes need a different initialization? Because the output shape is sometime specified in the attributes, sometime is passed as an initializer and sometimes is a ValueInfoProto
                if (std.mem.eql(u8, node.op_type, "Reshape")) {

                    // ------------------ is it a ValueInfoProto? most probable option
                    if (utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph)) |vip_tensor| {
                        const tensorZant: TensorZant = try TensorZant.init(
                            output_name,
                            null,
                            vip_tensor,
                            null,
                            TensorCategory.LINK,
                        );
                        //add the readyTensor to the HashMap
                        try tensorMap.put(tensorZant.name, tensorZant);

                        continue;
                    }

                    // ------------------ is it passed as an input to the node? most probable option
                    //TODO
                    // ------------------ is it an initializer? it shoul be already initialized
                    //TODO

                    return error.Reshape_outputShape_NotFound;
                } else {
                    // std.debug.print("\n  +++", .{});

                    //if the tensor is null is represented by an empty string in the onnx, so It must not be initialized in the hashMap
                    if (std.mem.eql(u8, output_name, "")) continue;
                    //if the tensor already exists is means it is an onnx_initializer and it don't need to be initialized again
                    if (tensorMap.getPtr(output_name) != null) continue;
                    //create the readyTensor
                    const tensorZant: TensorZant = try TensorZant.init(
                        output_name,
                        null,
                        utils.getValueInfoTensorFromGraphInfo(output_name, protoGraph),
                        null,
                        TensorCategory.LINK,
                    );
                    //add the readyTensor to the HashMap
                    try tensorMap.put(tensorZant.name, tensorZant);
                }
            }
        }
    }
}
