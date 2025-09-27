const std = @import("std");
const zant = @import("zant");

const IR_zant = @import("IR_zant");

const IR_codegen = IR_zant.IR_codegen;

// --- zant IR
const IR_utils = IR_zant.utils;
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const NodeZant = IR_zant.NodeZant;

const tensorZantMap: *std.StringHashMap(TensorZant) = &IR_zant.tensorZant_lib.tensorMap;

const allocator = std.heap.page_allocator;

const codegen_options = @import("codegen_options");
const templates = @import("templates.zig");
const emit = @import("emit.zig");
const plan = @import("plan.zig");
const accelerators = zant.core.tensor.accelerators;
const operators = IR_zant.operators;

fn positiveCast(value: i64) usize {
    return if (value > 0) @as(usize, @intCast(value)) else 0;
}

fn readSpatial(values_opt: ?[]i64, default_value: usize) [2]usize {
    var result = [2]usize{ default_value, default_value };
    if (values_opt) |values| {
        if (values.len > 0) {
            result[0] = positiveCast(values[0]);
        }
        if (values.len > 1) {
            result[1] = positiveCast(values[1]);
        } else if (values.len == 1) {
            result[1] = result[0];
        }
    }
    return result;
}

fn readPads(values_opt: ?[]i64) [4]usize {
    var result = [4]usize{ 0, 0, 0, 0 };
    if (values_opt) |values| {
        if (values.len > 0) result[0] = positiveCast(values[0]);
        if (values.len > 1) result[1] = positiveCast(values[1]);
        if (values.len > 2) result[2] = positiveCast(values[2]);
        if (values.len > 3) result[3] = positiveCast(values[3]);
    }
    return result;
}

fn safeProduct(values: []const usize) ?usize {
    var result: usize = 1;
    for (values) |val| {
        result = std.math.mul(usize, result, val) catch return null;
    }
    return result;
}

// Writes the computation function for predicting outputs
pub inline fn writePredict(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant), do_export: bool) !void {

    // Static initialization for output tensors if not using dynamic allocation
    //
    // declare all the outputs for each node, aka: linkers
    if (!codegen_options.dynamic) try write_linkersInitialization(writer);

    // declare all the outputs of  the network
    try write_outputsInitialization(writer);

    // method to reset the tensors values
    if (!codegen_options.dynamic) try write_linkersResetMethod(writer);

    const inputs = try IR_utils.getInputs(tensorZantMap);
    const outputs = try IR_utils.getOutputs(tensorZantMap);

    //write input type
    if (inputs.len == 0) {
        _ = try writer.print(
            \\
            \\ const T_in : type = {s};
        , .{if (outputs.len > 0) outputs[0].ty.toString() else @as([]const u8, "f32")});
    } else {
        _ = try writer.print(
            \\
            \\ const T_in : type = {s};
        , .{inputs[0].ty.toString()});
    }

    //write output type
    _ = try writer.print(
        \\
        \\ const T_out : type = {s};
    , .{if (outputs.len > 0) outputs[0].ty.toString() else @as([]const u8, "f32")});

    // Emit log helper function outside the predict function
    if (codegen_options.log) {
        try templates.emitLogHelper(writer);
    }

    try templates.emitFunctionSignature(writer, do_export);

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\    logMsg("Starting prediction...\\n");
        , .{});
    }

    //if I'm using statical allocation I'll reset all the Link tensors to zero
    if (!codegen_options.dynamic) {
        _ = try writer.print(
            \\
            \\    // Reset all linker tensors to zero before each prediction
            \\    resetOutputTensors();
        , .{});
    }

    // Suppress unused parameter warnings for nodes with no inputs
    if (inputs.len == 0) {
        _ = try writer.print(
            \\
            \\    // Suppress unused parameter warnings for no-input nodes
            \\    _ = input;
            \\    _ = input_shape;
            \\    _ = shape_len;
        , .{});
    }

    try write_checks(writer, linearizedGraph);

    try write_predictInitialization(writer, linearizedGraph);

    // Allocate output tensors for dynamic mode (only when NOT using plan-based execution)
    if (codegen_options.dynamic) {
        // TODO: When using plan-based execution, the PlanEmitter handles all tensor allocation
        // For now, we skip traditional output tensor allocation to avoid duplicates
        // In a complete implementation, we'd check if plan-based execution is enabled

        // Temporarily skip output tensor allocation to avoid conflicts with PlanEmitter
        // const output_tensors: []TensorZant = try IR_utils.getOutputs(tensorZantMap);
        // for (output_tensors) |*tz| {
        //     _ = try emit.ShapeEmitter.emit(writer, tz);
        //     const sanitized_name = try tz.getNameSanitized();
        //     const type_str = tz.ty.toString();
        //     try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return {d};\n", .{ sanitized_name, type_str, sanitized_name, templates.RC.INIT_ERROR });
        //     //since we are using dynamic inference  we also have to free the output_tensor so to avoid leaks, seee how I return the output tensor in writeReturn()
        //     try writer.print("    defer tensor_{s}.deinit();", .{sanitized_name});
        // }
    }

    // Use plan-based execution if enabled, otherwise fallback to old method
    if (codegen_options.dynamic) {
        try write_graphSerializationPlan(writer, linearizedGraph);
    } else {
        try write_graphSerialization(writer, linearizedGraph);
    }

    try writeReturn(writer);

    _ = try writer.print(
        \\
        \\    return {d};
        \\
        \\}}
    , .{templates.RC.OK});
}

// -------------------------------- WRITE LINKERS --------------------------------

// Initializes output tensor of each node in the computation graph
fn write_linkersInitialization(writer: std.fs.File.Writer) !void {
    try writer.print(
        \\
        \\
        \\ // ---------------------------------------------------
        \\ // +         Initializing linkers Tensors             +
        \\ // ---------------------------------------------------
    , .{});

    const linkers: []TensorZant = try IR_utils.getLinkers(tensorZantMap);

    for (linkers) |*tz| {
        const size = try emit.ShapeEmitter.emit(writer, tz);
        try emit.TensorEmitter.emitAllocation(writer, tz, size, codegen_options.dynamic);
    }
}

fn write_linkersResetMethod(writer: std.fs.File.Writer) !void {
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
            \\        log(@constCast(@ptrCast("Resetting output tensors...\\n")));
            \\    }}
        , .{});
    }

    // --------- linkers
    const linkers: []TensorZant = try IR_utils.getLinkers(tensorZantMap);

    for (linkers) |*tz| {
        if (!codegen_options.dynamic) {
            _ = try writer.print(
                \\
                \\    @memset(array_{s}[0..], 0);
            , .{try tz.getNameSanitized()});
        }

        if (codegen_options.log) {
            _ = try writer.print(
                \\
                \\    if (log_function) |log| {{
                \\        log(@constCast(@ptrCast("Linker tensor {s} reset.\n")));
                \\    }}
            , .{try tz.getNameSanitized()});
        }
    }

    // --------- outputs
    const outputs: []TensorZant = try IR_utils.getOutputs(tensorZantMap);

    for (outputs) |*tz| {
        if (!codegen_options.dynamic) {
            _ = try writer.print(
                \\
                \\    @memset(array_{s}[0..], 0);
            , .{try tz.getNameSanitized()});
        }

        if (codegen_options.log) {
            _ = try writer.print(
                \\
                \\    if (log_function) |log| {{
                \\        log(@constCast(@ptrCast("Output tensor {s} reset.\n")));
                \\    }}
            , .{try tz.getNameSanitized()});
        }
    }

    try writer.print(
        \\
        \\}}
    , .{});
}

// -------------------------------- WRITE OUTPUT --------------------------------
// Initializes output tensor of each node in the computation graph
fn write_outputsInitialization(writer: std.fs.File.Writer) !void {
    if (!codegen_options.dynamic) {
        try writer.print(
            \\
            \\
            \\ // ---------------------------------------------------
            \\ // +         Initializing Output Tensors             +
            \\ // ---------------------------------------------------
        , .{});

        const outputs: []TensorZant = try IR_utils.getOutputs(tensorZantMap);

        for (outputs) |*tz| {
            const size = try emit.ShapeEmitter.emit(writer, tz);
            try emit.TensorEmitter.emitAllocation(writer, tz, size, codegen_options.dynamic);
        }
    }
}

// -------------------------------- WRITE PREDICT() --------------------------------

fn write_predictInitialization(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    const inputs: []TensorZant = try IR_utils.getInputs(tensorZantMap);

    // if there are no external inputs (e.g., node extracted model with only initializers), skip input setup
    if (inputs.len == 0) {
        return;
    }

    // Identify the primary external input (first non-initializer)
    var primary_index: usize = std.math.maxInt(usize);
    for (inputs, 0..) |*tz, idx| {
        if (tz.tc != tensorZant_lib.TensorCategory.INITIALIZER) {
            primary_index = idx;
            break;
        }
    }

    if (primary_index == std.math.maxInt(usize)) {
        // No runtime-provided inputs needed
        return;
    }

    // If the external input is not used by any node, skip emitting it to save memory
    if (!isTensorUsedInGraph(linearizedGraph, inputs[primary_index])) {
        return;
    }

    //checks
    // Allow multiple inputs; only the primary is sourced from the user pointer
    if (inputs.len > 1) {
        // no-op: other inputs will be allocated below
    }

    // Zero-copy input binding: view directly over caller's input pointer
    // 1) compute runtime input_size from provided input_shape
    _ = try writer.print(
        \\  
        \\    //computing the size of the input tensor (runtime)
        \\    var input_size: usize = 1;
        \\    for(0..shape_len) |dim_i| {{
        \\        input_size *= @as(usize, input_shape[dim_i]);
        \\    }}
    , .{});

    // 2) emit fixed input shape from model (already validated by checks)
    const fixed_shape = inputs[primary_index].getShape();
    _ = try writer.print(
        \\
        \\    // Fixed input shape (validated above)
        \\    var input_shape_fixed: [{d}]usize = .{{ 
    , .{fixed_shape.len});
    for (fixed_shape, 0..) |dim, i| {
        if (i > 0) try writer.print(", ", .{});
        try writer.print("{d}", .{dim});
    }
    _ = try writer.print(
        \\ }};
        \\
        \\    // Zero-copy tensor pointing directly to input data
        \\    var tensor_{s} = Tensor(T_in){{
        \\        .data = input[0..input_size],
        \\        .shape = input_shape_fixed[0..],
        \\        .size = input_size,
        \\        .allocator = &allocator, // non-owning view
        \\    }};
    , .{try inputs[primary_index].getNameSanitized()});

    // For any additional non-initializer inputs, allocate zero-initialized tensors using their declared shapes
    for (inputs, 0..) |*tz, idx| {
        if (idx == primary_index) continue;
        if (tz.tc == tensorZant_lib.TensorCategory.INITIALIZER) continue;
        _ = try emit.ShapeEmitter.emit(writer, tz);
        const sanitized_name = try tz.getNameSanitized();
        const type_str = tz.ty.toString();
        try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return {d};\n", .{ sanitized_name, type_str, sanitized_name, templates.RC.INIT_ERROR });
        try writer.print("    defer tensor_{s}.deinit();\n", .{sanitized_name});
        try writer.print("    @memset(tensor_{s}.data[0..], 0);\n", .{sanitized_name});
    }
}

fn writeReturn(writer: std.fs.File.Writer) !void {
    const outputs: []TensorZant = try IR_utils.getOutputs(tensorZantMap);

    //checks
    if (outputs.len > 1) return error.MoreThanOneOutput;
    if (outputs.len < 1) return error.NoOutput;

    if (codegen_options.dynamic) {
        _ = try writer.print(
            \\     
            \\     const output_zant_slice = allocator.alloc(T_out, tensor_{s}.size) catch return {d};
            \\     @memcpy(output_zant_slice, tensor_{s}.data[0..tensor_{s}.size]);
            \\     
            \\     // Deallocate the output tensor after copying its data
            \\     tensor_{s}.deinit();
            \\      
            \\     //The Caller must handle the memory of output_zant_slice
            \\     result.* = output_zant_slice.ptr;
            \\
        , .{ try outputs[0].getNameSanitized(), templates.RC.RETURN_ERROR, try outputs[0].getNameSanitized(), try outputs[0].getNameSanitized(), try outputs[0].getNameSanitized() });
    } else {
        _ = try writer.print(
            \\
            \\    result.* = tensor_{s}.data.ptr;
            \\
        , .{try outputs[0].getNameSanitized()});
    }

    // Add deallocation for dynamic tensors -> OSS!! not necessary, the tensor are deallocated in deallocate_useless_link_tensors()
    //
    // if (codegen_options.dynamic) {
    //     const linkers: []TensorZant = try IR_utils.getLinkers(tensorZantMap);
    //     for (linkers) |*tz| {
    //         _ = try writer.print(
    //             \\    tensor_{s}.deinit();
    //             \\
    //         , .{try tz.getNameSanitized()});
    //     }
    // }

    if (codegen_options.log) {
        _ = try writer.print(
            \\
            \\    logMsg("Prediction completed.\\n");
        , .{});
    }
}

// -------------------------------- OTHER WRITE --------------------------------

fn write_checks(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    // Autogen a check for the input shape as arg VS input shape as codegen option

    const inputs: []TensorZant = try IR_utils.getInputs(tensorZantMap);

    // if there are no external inputs, there is nothing to check
    if (inputs.len == 0) {
        return;
    }

    // Allow multiple inputs; only validate against the first non-initializer input
    var check_index: usize = std.math.maxInt(usize);
    for (inputs, 0..) |*tz, idx| {
        if (tz.tc != tensorZant_lib.TensorCategory.INITIALIZER) {
            check_index = idx;
            break;
        }
    }

    // If no suitable external input or it is not used by any node, skip checks and suppress args usage
    if (check_index == std.math.maxInt(usize) or !isTensorUsedInGraph(linearizedGraph, inputs[check_index])) {
        _ = try writer.print(
            \\
            \\    // No external input used by the graph; suppress unused args
            \\    _ = input;
            \\    _ = input_shape;
            \\    _ = shape_len;
        , .{});
        return;
    }

    //check on the number of dims
    _ = try writer.print(
        \\ 
        \\    //checks on the input parameters
        \\    if (shape_len == 0) return {d};
        \\    if(shape_len != {}) return {d};
    , .{ templates.RC.INIT_ERROR, inputs[check_index].getShape().len, templates.RC.INIT_ERROR });

    //check on dims correspondance
    for (inputs[check_index].getShape(), 0..) |dim, i| {
        _ = try writer.print(
            \\
            \\    if( input_shape[{}] != {}) return {d};
        , .{ i, dim, templates.RC.INIT_ERROR });
    }
}

// Returns true if the given tensor is consumed as an input by any node in the linearized graph
fn isTensorUsedInGraph(linearizedGraph: std.ArrayList(*NodeZant), tensor: TensorZant) bool {
    // Iterate nodes and their input tensors; compare names
    for (linearizedGraph.items) |node| {
        const maybe_inputs = node.get_input_tensors() catch {
            continue;
        };
        for (maybe_inputs) |t_in| {
            if (std.mem.eql(u8, t_in.name, tensor.name)) {
                return true;
            }
        }
    }
    return false;
}

fn write_graphSerializationPlan(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    // Build execution plan with liveness analysis
    std.log.info("Attempting to build ExecutionPlan with {d} nodes...", .{linearizedGraph.items.len});
    var execution_plan = plan.buildExecutionPlan(allocator, linearizedGraph) catch |err| {
        // Fallback to old method if plan building fails
        std.log.warn("ExecutionPlan building failed: {}, falling back to old allocation method", .{err});
        return write_graphSerialization(writer, linearizedGraph);
    };
    std.log.info("ExecutionPlan built successfully with {d} steps!", .{execution_plan.steps.items.len});
    defer execution_plan.deinit();

    if (codegen_options.log) {
        try writer.print(
            \\
            \\    logMsg("Using plan-based execution with {d} steps\\n");
        , .{execution_plan.steps.items.len});
    }

    // Use the new PlanEmitter to emit the ENTIRE graph (including input/output handling)
    try emit.PlanEmitter.emitGraph(writer, &execution_plan, codegen_options.dynamic);

    // TODO: The PlanEmitter should handle input/output tensors completely
    // For now, we need to make sure no duplicate tensors are generated
}

fn write_graphSerialization(writer: std.fs.File.Writer, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    for (linearizedGraph.items, 0..) |node, i| {
        if (codegen_options.comm) {
            try write_op_info(writer, node);
        }
        if (codegen_options.log) {
            try writer.print(
                \\ 
                \\
                \\    if (log_function) |log| {{
                \\        log(@constCast(@ptrCast("Running {s} operation...\n")));
                \\    }}
            , .{node.*.op_type});
        }

        //Before computing the OP, init link tensors when we are in dynamic allocation
        if (codegen_options.dynamic) try allocate_output_link_tensors(writer, node); //ERE YOU ALLOCATE

        try node.write_op(writer);

        //After computing the OP, delete link tensors that are not useful anymore when we are in dynamic allocation
        if (codegen_options.dynamic) try deallocate_useless_link_tensors(writer, i, linearizedGraph); //HERE YOU DEALLOCATE
    }
}

//dynamically allocate a linker tensor
fn allocate_output_link_tensors(writer: std.fs.File.Writer, node: *NodeZant) !void {

    //if not used anymore in the rest of the graph
    for (try node.get_output_tensors()) |output_tensor| {
        if (output_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
            _ = try emit.ShapeEmitter.emit(writer, output_tensor);

            const sanitized_name = try output_tensor.getNameSanitized();

            // --- ADD CHECK FOR UNDEFINED TYPE ---
            if (output_tensor.ty == .undefined) {
                std.log.warn("\n\nCODEGEN ERROR: Attempted to generate output tensor '{s}' but its data type is UNDEFINED. Check ONNX graph.\n\n", .{sanitized_name});
                return error.DataTypeNotAvailable; // Or a more specific error like CannotGenerateUndefinedType
            }
            // --- END CHECK ---

            const type_str = output_tensor.ty.toString();

            // Dynamic allocation: Use fromShape to allow mutation
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return {d};", .{ sanitized_name, type_str, sanitized_name, templates.RC.INIT_ERROR });
        }
    }
}

//free a linker tensor after dynamical allocation
fn deallocate_useless_link_tensors(writer: std.fs.File.Writer, starting_node: usize, linearizedGraph: std.ArrayList(*NodeZant)) !void {
    const node = linearizedGraph.items[starting_node];

    //After computing the OP, delete link tensors that are not useful anymore when we are in dynamic allocation

    var used: bool = false;
    for (try node.get_input_tensors()) |my_input_tensor| { //for each input tensor of my node
        used = false;
        //if not used anymore in the rest of the graph
        for (starting_node + 1..linearizedGraph.items.len) |j| {
            for (try linearizedGraph.items[j].get_input_tensors()) |other_input_tens| {
                if (std.mem.eql(u8, my_input_tensor.name, other_input_tens.name)) {
                    used = true;
                    break;
                }
            }
            if (used) break;
        }

        //if it is not used anymore in the graph and it is an initializer I can deinit it
        if (!used and my_input_tensor.tc == tensorZant_lib.TensorCategory.LINK) {
            _ = try writer.print("    tensor_{s}.deinit();\n", .{try my_input_tensor.getNameSanitized()});
        }
    }

    //
}

fn write_op_info(writer: std.fs.File.Writer, node: *NodeZant) !void {
    try writer.print(
        \\
        \\
        \\   //forwarding operation : {s}
    , .{node.*.op_type});

    try writer.print(
        \\
        \\   //parameters:
        \\   //   inputs: 
    , .{});

    //write the inputs
    for (try node.get_input_tensors()) |input| {
        try writer.print(
            \\
            \\   //      -> {s}
        , .{input.name});
    }
    try writer.print(
        \\
        \\   //    outputs:
    , .{});

    //write the outputs
    for (try node.get_output_tensors()) |output| {
        try writer.print(
            \\
            \\   //      <- {s}
        , .{output.name});
    }
}
