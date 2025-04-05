const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeProto = onnx.AttributeProto;
const allocator = zant.utils.allocator.allocator;

const codegen = @import("codegen.zig");
const utils = codegen.utils;
const mathGen = codegen.math_handler;
const codegen_options = @import("codegen_options");

const globals = codegen.globals;
const ReadyNode = globals.ReadyNode;
const ReadyTensor = globals.ReadyTensor;

// Writes the computation function for predicting outputs
pub inline fn writePredict(writer: std.fs.File.Writer, do_export: bool) !void {
    //declare all the outputs of each node of the network
    try write_outputsInitialization(writer);

    //method to reset the tensors values
    try write_outputsResetMethod(writer);

    _ = try writer.print(
        \\
        \\
        \\
        \\pub {s} fn predict( 
        \\    input: [*]T,
        \\    input_shape: [*]u32,
        \\    shape_len: u32,
        \\    result: *[*]T,
        \\) void {{
    , .{if (do_export == true) "export" else ""});

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Starting prediction...\n")));
            \\    }}
        , .{});
    }

    _ = try writer.print(
        \\
        \\    // Reset all output tensors to zero before each prediction
        \\    resetOutputTensors();
    , .{});

    try write_checks(writer);

    try write_predictInitialization(writer);

    try write_graphSerialization(writer);

    try writeReturn(writer);

    _ = try writer.print(
        \\
        \\}} 
    , .{});
}

// Processes and writes the computation graph
inline fn write_graphSerialization(writer: std.fs.File.Writer) !void {
    var iteration: usize = 0;
    var lastNode: *ReadyNode = undefined;
    while (true) {
        const computableNodes: std.ArrayList(*ReadyNode) = try utils.getComputableNodes(&globals.readyGraph);

        //DEBUG
        // for (globals.readyGraph.items) |*readyNode| readyNode.print(false);
        try utils.printComputableNodes(computableNodes, true);

        if (computableNodes.items.len == 0) break;
        //else set the last node as the network output
        lastNode = computableNodes.items[computableNodes.items.len - 1];

        for (computableNodes.items) |node_ptr| {
            //writing the operation
            try writeOperation(writer, node_ptr);
            std.debug.print("\n AAAAAAAAAAAAAAAA", .{});
            //set the output as ready
            try utils.setOutputsReady(node_ptr, &globals.tensorHashMap);
            std.debug.print("\n BBBBBBBBBBBBB", .{});
        }
        iteration += 1;
    }

    // If this is the output node, we don't need to check its outputs
    if (std.mem.eql(u8, try utils.getSanitizedName(lastNode.nodeProto.name.?), "output")) {
        return;
    }

    //check if it is different from the one already parsed, if not present, set to lastNode
    if (lastNode.outputs.items.len == 0) {
        return error.NoOutputsFound;
    }
    if (std.mem.eql(u8, globals.networkOutput.name, "")) {
        //setting te network output
        globals.networkOutput.name = lastNode.outputs.items[0].name;
        globals.networkOutput.shape = lastNode.outputs.items[0].shape;
    } else {
        //check the output tensor name is the same
        if (!std.mem.eql(u8, globals.networkOutput.name, lastNode.outputs.items[0].name)) {
            std.debug.print("\n\n   ERROR!!\n    DifferentOutputNames: \n      {s} vs {s}\n    LastNode:{s}\n\n", .{ globals.networkOutput.name, lastNode.outputs.items[0].name, lastNode.nodeProto.name.? });
            lastNode.print(true);
            return error.DifferentOutputNames;
        }
    }
}

// -------------------------------- WRITE OUTPUTS --------------------------------

// Initializes output tensors in the computation graph
fn write_outputsInitialization(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\
        \\
        \\ // ---------------------------------------------------
        \\ // +         Initializing output Tensors             +
        \\ // ---------------------------------------------------
    , .{});

    for (globals.readyGraph.items) |*node| {

        //writing the outputs, OSS: two nodes shpuld never have the same output by definition, so we don't need to check for duplicates
        for (node.outputs.items) |output| {
            if (std.mem.eql(u8, node.nodeProto.op_type, "Constant") and node.inputs.items.len == 0) { //A node is constant if it only has one output and no inputs
                if (node.outputs.items.len > 1) return error.MultipleOutputConstant else {
                    try write_constantTensor(writer, node);
                    //set the node and tensor to Ready
                    var mutableNode: *ReadyNode = @constCast(node);
                    mutableNode.ready = true;
                }
            } else {
                if (@as(?*ReadyTensor, output) == null) return error.InvalidOutput;
                const size = try write_OutputShape(
                    writer,
                    output,
                );
                try write_OutputTensor(
                    writer,
                    output.name,
                    size,
                );
            }
        }
    }
}

fn write_OutputShape(writer: std.fs.File.Writer, output: *ReadyTensor) !i64 {
    if (@as(?*ReadyTensor, output) == null) return error.InvalidOutput;
    const shape = output.shape;
    var size: i64 = 1;

    try writer.print(
        \\
        \\
        \\var shape_tensor_{s} : [{}]usize = [_]usize{{
    , .{
        try utils.getSanitizedName(output.name),
        shape.len,
    });

    for (0..shape.len) |i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print(
            \\ {}
        , .{shape[i]});
        size *= shape[i];
    }

    try writer.print(
        \\}} ;
    , .{});

    return size;
}

fn write_constantTensor(writer: std.fs.File.Writer, readyNode: *const ReadyNode) !void {
    try writer.print(
        \\
        \\ // ---- CONSTANT TENSOR ---- 
    , .{});

    // Get the output tensor (constant nodes have exactly one output)
    const output = readyNode.outputs.items[0];
    const sanitized_name = try utils.getSanitizedName(output.name);

    // Find the value attribute which contains the constant tensor
    var value_attr: ?*AttributeProto = null;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "value")) {
            value_attr = attr;
            break;
        }
    }

    if (value_attr == null or value_attr.?.t == null) return error.MissingConstantValue;
    const tensor = value_attr.?.t.?;

    // Write shape array
    try writer.print(
        \\
        \\const shape_tensor_{s} : [{}]usize = [_]usize{{
    , .{ sanitized_name, output.shape.len });

    for (0..output.shape.len) |i| {
        if (i > 0) try writer.print(",", .{});
        try writer.print(
            \\ {}
        , .{output.shape[i]});
    }

    try writer.print(
        \\}} ;
    , .{});

    // Write data array
    var total_size: i64 = 1;
    for (tensor.dims) |dim| {
        total_size *= dim;
    }

    //const dataTypeString = try utils.getTypeString(tensor.data_type);
    try writer.print(
        \\
        \\const array_{s} : [{d}]T = [_]T{{
    , .{ sanitized_name, total_size });

    // Write the actual data values
    if (tensor.float_data) |data| {
        for (0..data.len) |i| {
            if (i > 0) try writer.print(",", .{});
            try writer.print(" {d}", .{data[i]});
        }
    } else if (tensor.int64_data) |data| {
        for (0..data.len) |i| {
            if (i > 0) try writer.print(",", .{});
            try writer.print(" {d}", .{data[i]});
        }
    } else if (tensor.raw_data) |data| {
        switch (tensor.data_type) {
            .FLOAT => {
                const float_data = @as([*]const f32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
                for (0..float_data.len) |i| {
                    if (i > 0) try writer.print(",", .{});
                    try writer.print(" {d}", .{float_data[i]});
                }
            },
            .INT64 => {
                const int_data = @as([*]const i64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
                for (0..int_data.len) |i| {
                    if (i > 0) try writer.print(",", .{});
                    try writer.print(" {d}", .{int_data[i]});
                }
            },
            else => return error.UnsupportedDataType,
        }
    } else return error.NoDataAvailable;

    try writer.print(
        \\ }};
    , .{});

    // Write tensor initialization using fromArray
    try writer.print(
        \\
        \\const tensor_{s} = Tensor(T).fromConstBuffer(&allocator, &array_{s}, &shape_tensor_{s});
    , .{ sanitized_name, sanitized_name, sanitized_name });
}

fn write_OutputTensor(writer: std.fs.File.Writer, name: []const u8, size: i64) !void {
    const sanitized_name = try utils.getSanitizedName(name);
    try writer.print(
        \\
        \\var array_{s}: [{}]T = [_]T{{0}} ** {};
        \\var tensor_{s} = Tensor(T).fromConstBuffer( &allocator, &array_{s}, &shape_tensor_{s});
    , .{ sanitized_name, size, size, sanitized_name, sanitized_name, sanitized_name });
}

fn write_outputsResetMethod(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\
        \\
        \\//Function to reset all output tensors to zero
        \\fn resetOutputTensors() void {{
    , .{});

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Resetting output tensors...\n")));
            \\    }}
        , .{});
    }

    for (globals.readyGraph.items) |*node| {
        // Skip constant nodes
        if (std.mem.eql(u8, node.nodeProto.op_type, "Constant") and node.inputs.items.len == 0) {
            continue;
        }

        for (node.outputs.items) |output| {
            _ = try writer.print(
                \\
                \\    @memset(array_{s}[0..], 0);
            , .{try utils.getSanitizedName(output.name)});
        }
    }

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Output tensors reset.\n")));
            \\    }}
        , .{});
    }

    try writer.print(
        \\
        \\}}
    , .{});
}

// -------------------------------- WRITE CHECKS --------------------------------

fn write_checks(writer: std.fs.File.Writer) !void {
    // Autogen a check for the input shape as arg VS input shape as codegen option

    //check on the number of dims
    _ = try writer.print(
        \\
        \\    //checks on the input parameters
        \\    if (shape_len == 0) return ;
        \\    if(shape_len != {}) return ;
    , .{globals.networkInput.shape.len});

    //check on dims correspondance
    for (globals.networkInput.shape, 0..) |dim, i| {
        _ = try writer.print(
            \\
            \\    if( input_shape[{}] != {}) return ;
        , .{ i, dim });
    }
}

// -------------------------------- WRITE PREDICT() --------------------------------

fn write_predictInitialization(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\  
        \\    //computing the size of the input tensor
        \\    var size: u32 = 1;
        \\    for(0..shape_len) |dim_i| {{
        \\        size *= input_shape[dim_i];
        \\    }}
        \\     
        \\    //allocating space in memory for the data
        \\    const data = allocator.alloc(T, size) catch return;
        \\    defer allocator.free(data);
        \\    for (0..size) |i| {{
        \\        data[i] = input[i]; // Copying input elements 
        \\    }}
        \\    
        \\    //converting the shape from [*]u32 to []usize
        \\    const usized_shape: []usize = utils.u32ToUsize(input_shape, shape_len) catch return;
        \\    var tensor_{s} = Tensor(T).fromShape(&allocator, @constCast(usized_shape)) catch return;
        \\    defer allocator.free(usized_shape);
        \\    defer tensor_{s}.deinit();
        \\    @memcpy(tensor_{s}.data, data);
    , .{
        try utils.getSanitizedName(globals.networkInput.name),
        try utils.getSanitizedName(globals.networkInput.name),
        try utils.getSanitizedName(globals.networkInput.name),
    });
}

fn writeOperation(writer: std.fs.File.Writer, readyNode: *ReadyNode) !void {
    try mathGen.write_math_op(writer, readyNode);
}

fn writeReturn(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\    result.* = tensor_{s}.data.ptr;
        \\
    , .{try utils.getSanitizedName(globals.networkOutput.name)});

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Prediction completed.\n")));
            \\    }}
        , .{});
    }
}
