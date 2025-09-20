const std = @import("std");
const IR_zant = @import("IR_zant");
const TensorZant = IR_zant.TensorZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorCategory = tensorZant_lib.TensorCategory;
const templates = @import("templates.zig");
const plan = @import("plan.zig");
const codegen_options = @import("codegen_options");

// Global allocator for name sanitization
var sanitize_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
const sanitize_allocator = sanitize_arena.allocator();

const ExecutionPlan = plan.ExecutionPlan;
const PlanStep = plan.PlanStep;
const PlanTensor = plan.PlanTensor;

/// Centralized shape emission - replaces write_TensorShape
pub const ShapeEmitter = struct {
    /// Emits tensor shape declaration and returns computed size
    /// Replaces the existing write_TensorShape function
    pub fn emit(writer: std.fs.File.Writer, tz: *TensorZant) !i64 {
        var size: i64 = 1;
        const tensor_shape = tz.getShape();

        try writer.print(
            \\
            \\
            \\var shape_tensor_{s} : [{}]usize = [_]usize{{
        , .{
            try tz.getNameSanitized(),
            tensor_shape.len,
        });

        for (tensor_shape, 0..) |dim_i, i| {
            if (i > 0) try writer.print(",", .{});
            try writer.print(
                \\ {}
            , .{dim_i});

            size *= @intCast(dim_i);
        }

        try writer.print(
            \\}} ;
        , .{});

        return size;
    }
};

/// Centralized tensor allocation emission - will replace write_TensorAllocation
pub const TensorEmitter = struct {
    /// Emits tensor allocation given a shape size
    /// This will eventually replace write_TensorAllocation with policy-based approach
    pub fn emitAllocation(writer: std.fs.File.Writer, tz: *TensorZant, size: i64, dynamic: bool) !void {
        const sanitized_name = try tz.getNameSanitized();

        // --- ADD CHECK FOR UNDEFINED TYPE ---
        if (tz.ty == .undefined) {
            std.log.warn("\n\nCODEGEN ERROR: Attempted to generate output tensor '{s}' but its data type is UNDEFINED. Check ONNX graph analysis in globals.zig.\n\n", .{sanitized_name});
            return error.DataTypeNotAvailable;
        }
        // --- END CHECK ---

        const type_str = tz.ty.toString();

        if (dynamic) {
            // Dynamic allocation: Use fromShape
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return {d};", .{ sanitized_name, type_str, sanitized_name, templates.RC.INIT_ERROR });
        } else {
            // Static allocation: Use fromConstBuffer to allow mutation
            // Add tensor_pool linksection for large arrays when the option is enabled
            const use_tensor_pool = codegen_options.use_tensor_pool and size >= 10; // 10 element threshold (~40 bytes for f32)
            if (use_tensor_pool) {
                try writer.print("    var array_{s}: [{d}]{s} linksection(\".tensor_pool\") = [_]{s}{{0}} ** {d};", .{ sanitized_name, size, type_str, type_str, size });
            } else {
                try writer.print("    var array_{s}: [{d}]{s} = [_]{s}{{0}} ** {d};", .{ sanitized_name, size, type_str, type_str, size });
            }
            try writer.print("    var tensor_{s} = Tensor({s}).fromConstBuffer(&fba, &array_{s}, &shape_tensor_{s});", .{ sanitized_name, type_str, sanitized_name, sanitized_name });
        }
    }
};

/// Plan-based emitter that replaces O(N^2) allocation logic
pub const PlanEmitter = struct {
    /// Emits the entire graph execution using ExecutionPlan
    pub fn emitGraph(writer: std.fs.File.Writer, execution_plan: *const ExecutionPlan, dynamic: bool) !void {
        // First, emit all input tensors (these don't have allocation events)
        try emitInputOutputTensors(writer, execution_plan, dynamic);

        for (execution_plan.steps.items, 0..) |*step, step_idx| {
            // Allocate tensors needed for this step
            for (step.allocs.items) |tensor| {
                try emitTensorAllocation(writer, &tensor, dynamic);
            }

            // Comment with operation info
            try writer.print(
                \\
                \\   // Step {d}: {s} operation
                \\
            , .{ step_idx, sanitizeName(step.node.*.op_type) });

            // Execute the operation using the node's write_op method
            try step.node.*.write_op(writer);

            // Deallocate tensors no longer needed (only in dynamic mode)
            if (dynamic) {
                for (step.frees.items) |tensor| {
                    try emitTensorDeallocation(writer, &tensor);
                }
            }
        }
    }

    /// Emits input and output tensors that are not handled by allocation events
    fn emitInputOutputTensors(writer: std.fs.File.Writer, execution_plan: *const ExecutionPlan, dynamic: bool) !void {
        // For now, we'll emit input/output tensors based on the plan
        // This is a placeholder - in a complete implementation, we'd extract
        // input/output tensors from the execution plan
        _ = writer;
        _ = execution_plan;
        _ = dynamic;

        // TODO: Implement proper input/output tensor emission
        // For now, this prevents the traditional method from generating them
    }

    /// Emits allocation for a plan tensor
    fn emitTensorAllocation(writer: std.fs.File.Writer, tensor: *const PlanTensor, dynamic: bool) !void {
        // First emit the shape
        _ = try emitTensorShape(writer, tensor);

        // Then emit the allocation
        const sanitized_name = sanitizeName(tensor.name);
        const type_str = tensor.ty.toString();

        if (dynamic) {
            try writer.print("    var tensor_{s} = Tensor({s}).fromShape(&allocator, &shape_tensor_{s}) catch return {d};\n", .{ sanitized_name, type_str, sanitized_name, templates.RC.INIT_ERROR });
            // Only add defer for intermediate tensors, not for outputs (which are returned to caller)
            if (tensor.category != TensorCategory.OUTPUT) {
                try writer.print("    defer tensor_{s}.deinit();\n", .{sanitized_name});
            }
        } else {
            // Calculate size from shape
            var size: i64 = 1;
            for (tensor.shape) |dim| {
                size *= @intCast(dim);
            }

            // Add tensor_pool linksection for large arrays when the option is enabled
            const use_tensor_pool = codegen_options.use_tensor_pool and size >= 10; // 10 element threshold (~40 bytes for f32)
            if (use_tensor_pool) {
                try writer.print("    var array_{s}: [{d}]{s} linksection(\".tensor_pool\") = [_]{s}{{0}} ** {d};\n", .{ sanitized_name, size, type_str, type_str, size });
            } else {
                try writer.print("    var array_{s}: [{d}]{s} = [_]{s}{{0}} ** {d};\n", .{ sanitized_name, size, type_str, type_str, size });
            }
            try writer.print("    var tensor_{s} = Tensor({s}).fromConstBuffer(&fba, &array_{s}, &shape_tensor_{s});\n", .{ sanitized_name, type_str, sanitized_name, sanitized_name });
        }
    }

    /// Emits deallocation for a plan tensor (only for dynamic mode)
    fn emitTensorDeallocation(writer: std.fs.File.Writer, tensor: *const PlanTensor) !void {
        const sanitized_name = sanitizeName(tensor.name);
        try writer.print("    tensor_{s}.deinit();\n", .{sanitized_name});
    }

    /// Emits shape for a plan tensor
    fn emitTensorShape(writer: std.fs.File.Writer, tensor: *const PlanTensor) !i64 {
        var size: i64 = 1;
        const sanitized_name = sanitizeName(tensor.name);

        try writer.print(
            \\
            \\
            \\var shape_tensor_{s} : [{}]usize = [_]usize{{
        , .{ sanitized_name, tensor.shape.len });

        for (tensor.shape, 0..) |dim, i| {
            if (i > 0) try writer.print(",", .{});
            try writer.print(" {}", .{dim});
            size *= @intCast(dim);
        }

        try writer.print("}} ;\n", .{});
        return size;
    }
};

/// Helper to sanitize tensor names (replaces getNameSanitized for PlanTensor)
fn sanitizeName(name: []const u8) []const u8 {
    var sanitized = sanitize_allocator.alloc(u8, name.len) catch {
        // Fallback: return original name if allocation fails
        return name;
    };

    for (name, 0..) |char, i| {
        sanitized[i] = if (std.ascii.isAlphanumeric(char) or char == '_')
            std.ascii.toLower(char)
        else
            '_';
    }

    return sanitized;
}
