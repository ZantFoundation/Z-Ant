const std = @import("std");
const zant = @import("zant");
const Tensor = zant.tensor.Tensor;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
//--- proto
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeProto = onnx.AttributeProto;
const allocator = zant.utils.allocator.allocator;

//--- other
const codegen = @import("codegen.zig");
const utils = codegen.utils;
const mathGen = codegen.math_handler;
const shapeGen = codegen.shape_handler;
const codegen_options = @import("codegen_options");

pub var readyGraph: std.ArrayList(ReadyNode) = std.ArrayList(ReadyNode).init(allocator);
pub var tensorHashMap: std.StringHashMap(ReadyTensor) = std.StringHashMap(ReadyTensor).init(allocator); //key: TensorProto.name

pub var onnxModel: ModelOnnx = undefined; //initialized in setGlobalAttributes(), it is mandatory

pub const io_struct = struct {
    name: []const u8,
    shape: []const i64,
};

pub var networkInput = io_struct{
    .name = "",
    .shape = &[_]i64{},
};

pub var networkOutput = io_struct{
    .name = "",
    .shape = &[_]i64{},
};

pub var inputType: type = f32;

pub const TensorTag = enum {
    INITIALIZER,
    CONSTANT,
    INPUT,
    OUTPUT,
    LINK, //with "LINK" I mean a tensor that is used to link two nodes, it means that is is the output of a node and the input of another
};

// Struct to represent a tensor that is ready for computation
pub const ReadyTensor = struct {
    name: []const u8,
    ready: bool,
    shape: []const i64,
    tensorProto: ?*TensorProto,
    tag: TensorTag,

    pub fn createInitializer(tensorProto: *TensorProto) !ReadyTensor {
        return ReadyTensor{
            .name = tensorProto.name.?,
            .ready = true,
            .shape = tensorProto.dims,
            .tensorProto = tensorProto,
            .tag = TensorTag.INITIALIZER,
        };
    }

    pub fn createInput(name: []const u8) !ReadyTensor {
        return ReadyTensor{
            .name = name,
            .ready = true,
            .shape = networkInput.shape,
            .tensorProto = null,
            .tag = TensorTag.INPUT,
        };
    }

    pub fn createConstant(name: []const u8, tensorProto: *TensorProto) !ReadyTensor {
        return ReadyTensor{
            .name = name,
            .ready = true,
            .shape = networkInput.shape,
            .tensorProto = tensorProto,
            .tag = TensorTag.CONSTANT,
        };
    }

    pub fn createLink(name: []const u8) !ReadyTensor {
        return ReadyTensor{ //default
            .name = name,
            .ready = false,
            .shape = networkInput.shape,
            .tensorProto = null,
            .tag = TensorTag.LINK,
        };
    }

    pub fn print(tensor: *ReadyTensor, detailed: bool) void {
        std.debug.print("\n      READY TENSOR : {s}", .{tensor.name});
        std.debug.print("\n         status:{s}ready", .{if (!tensor.ready) " not " else " "});
        std.debug.print("\n         tag: {any}", .{tensor.tag});
        std.debug.print("\n         shape: {any}", .{tensor.shape});
        if (detailed) if (tensor.tensorProto) |tp| tp.print("         ") else std.debug.print("\n         tensor.tensorProto :(null)", .{});
    }
};

// Struct representing a computational node in the ONNX model
pub const ReadyNode = struct {
    nodeProto: *NodeProto,
    inputs: std.ArrayList(?*ReadyTensor),
    outputs: std.ArrayList(*ReadyTensor),
    ready: bool,

    // Creates a ReadyNode by preparing its input and output tensors
    pub fn create(nodeProto: *NodeProto) !ReadyNode {
        // std.debug.print("\n\nReadyNode.create() --> {s}", .{nodeProto.name.?});
        var newReadyNode = ReadyNode{
            .nodeProto = nodeProto,
            .inputs = std.ArrayList(?*ReadyTensor).init(allocator),
            .outputs = std.ArrayList(*ReadyTensor).init(allocator),
            .ready = false,
        };

        for (nodeProto.input) |input_name| { //for each input tensor in NodeProto

            //adding the readyTensor to the model
            if (std.mem.eql(u8, input_name, "")) {
                try newReadyNode.inputs.append(null);
            } else {
                try newReadyNode.inputs.append(if (tensorHashMap.getPtr(input_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            }
        }
        for (nodeProto.output) |output_name| { //for each output tensor

            //adding the readyTensor to the model
            try newReadyNode.outputs.append(if (tensorHashMap.getPtr(output_name)) |V_ptr| V_ptr else return error.keyNotAvailable);
            // std.debug.print("\n   added output {s} to node {s} ", .{ output_name, nodeProto.name.? });
        }

        // -- COMPUTING THE OUTPUT SHAPE --
        try shapeGen.compute_output_shape(&newReadyNode);

        return newReadyNode;
    }

    pub fn print(node: *ReadyNode, detailed: bool) void {
        std.debug.print("\n ------ READY NODE : ", .{});
        if (detailed) node.nodeProto.print("  ") else std.debug.print("\n {s} ", .{node.nodeProto.name.?});
        std.debug.print("\n  ---inputs : ", .{});
        for (node.inputs.items) |in| if (in) |i| i.print(detailed) else std.debug.print("\n      NULL INPUT", .{});
        std.debug.print("\n  ---outputs : ", .{});
        for (node.outputs.items) |out| out.print(detailed);
    }
};

pub fn setGlobalAttributes(model: ModelOnnx) !void {
    //initializing global attributes
    onnxModel = model;

    //ready graph
    readyGraph.deinit();
    readyGraph = std.ArrayList(ReadyNode).init(allocator);

    //hash map
    tensorHashMap.deinit();
    tensorHashMap = std.StringHashMap(ReadyTensor).init(allocator);

    //First convert the optional String of numbers divided by a comma into an array
    const parsedInputshape: []const i64 = try utils.parseNumbers(codegen_options.shape);

    //setting the input
    const inputs = model.graph.?.inputs;
    std.debug.print("\n SETTING networkInput \n name = {s} \n shape={any}", .{ inputs[0].name.?, inputs[0].type.?.tensor_type.?.shape.?.shape });
    networkInput.name = inputs[0].name.?;
    networkInput.shape = inputs[0].type.?.tensor_type.?.shape.?.shape;

    //setting the output
    const outputs = model.graph.?.outputs;
    std.debug.print("\n SETTING networkInput \n name = {s} \n shape={any}", .{ outputs[0].name.?, outputs[0].type.?.tensor_type.?.shape.?.shape });
    networkOutput.name = outputs[0].name.?;
    networkOutput.shape = outputs[0].type.?.tensor_type.?.shape.?.shape;

    // Use -Dshape if provided, otherwise keep the ONNX model's shape
    if (parsedInputshape.len > 0) {
        networkInput.shape = parsedInputshape;
    } else if (networkInput.shape.len == 0) {
        std.debug.print("\n\n ERROR: \n     Input shape is necessary to proceed! \n     Ensure that the onnx model has one or compile with -Dshape=''<your_shape>''", .{});
        return error.NoInputShape;
    }

    //create the hashMap
    try populateReadyTensorHashMap(model);

    //create the ReadyGraph
    try populateReadyGraph(model);

    std.debug.print("\n NODE: {s}", .{model.graph.?.nodes[0].output[0]});
}

// ----------------------- HASH MAP -----------------------
// Populates tensorHashMap with the tensors used in the onnx graph, where the key is the name of the tensor
fn populateReadyTensorHashMap(model: ModelOnnx) !void {
    const protoGraph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    //adding initializers to the hash map
    for (protoGraph.initializers) |init_ptr| {
        //create the readyTensor
        const readyTensor: ReadyTensor = try ReadyTensor.createInitializer(init_ptr);
        //add the readyTensor to the HashMap
        try tensorHashMap.put(readyTensor.name, readyTensor);
    }

    //adding all the nodes inputs and outputs
    for (protoGraph.nodes) |node| { //for each NodeProto in the GraphProto
        for (node.input) |input_name| {
            try addToTensorHashMap(input_name, node);
        }
        for (node.output) |output_name| {
            try addToTensorHashMap(output_name, node);
        }
    }
}

pub fn addToTensorHashMap(name: []const u8, nodeProto: *NodeProto) !void {
    if (tensorHashMap.get(name) != null or std.mem.eql(u8, name, "")) {
        return;
    } else {
        //if input
        if (utils.isInput(name)) {
            //add the readyTensor to the HashMap
            try tensorHashMap.put(name, try ReadyTensor.createInput(name));
            return;
        }
        //if constant, pay attention, we add the Constatant only if it is a TENSOR (aka AttributeProto.t)
        if (std.mem.eql(u8, nodeProto.op_type, "Constant")) {
            //add the readyTensor to the HashMap
            if (nodeProto.attribute[0].type == onnx.AttributeType.TENSOR) try tensorHashMap.put(name, try ReadyTensor.createConstant(name, nodeProto.attribute[0].t.?));
            return;
        }

        //else default
        //add the readyTensor to the HashMap
        try tensorHashMap.put(name, try ReadyTensor.createLink(name));
    }
}

// ----------------------- READY GRAPH -----------------------
// Creates a graph representation with all nodes in a ready-to-compute state
fn populateReadyGraph(model: ModelOnnx) !void {
    const graph = try if (model.graph) |graph| graph else error.GraphNotAvailable;

    for (graph.nodes) |node_ptr| { //for each NodeProto in the GraphProto

        try readyGraph.append(try ReadyNode.create(node_ptr));
    }
}
