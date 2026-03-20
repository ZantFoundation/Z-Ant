const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;
const Any = zant.uops.Any;

const MemoryRender = @import("memory_render.zig");
const ArithmeticRender = @import("arithmetic_render.zig");
const ReduceRender = @import("reduce_render.zig");
const ConditionalRender = @import("conditional_render.zig");
const UnaryRender = @import("unary_render.zig");
const GepRender = @import("gep_render.zig");
const ControlFlowRender = @import("controlflow_render.zig");
const ViewManagerModule = @import("view_manager.zig");
const ReshapeRender = @import("reshape_render.zig");
const ViewInfo = ViewManagerModule.ViewInfo;

const Uops = zant.uops;
const DTypeInfo = Uops.DTypeInfo;
const DType = Uops.DType;

// NEW: Structure to hold buffer info (inputs/outputs)
pub const BufferInfo = struct {
    id: usize,
    name: []const u8, // Note: lifetime managed by buffer_map allocator
    is_input: bool,
    dtype: DType,
    shape: []const usize, // Note: lifetime managed by buffer_map allocator
    len: usize,

    // Deinit is simple as allocator owns name/shape copies
};

pub fn ZigRenderer(comptime WriterType: type) type {
    return struct {
        allocator: std.mem.Allocator,
        writer: WriterType,
        rendered_ids: std.AutoHashMap(usize, void), // Changed to HashMap(usize, void)
        view_map: std.AutoHashMap(usize, ViewInfo),
        buffer_map: std.AutoHashMap(usize, BufferInfo),
        indent_level: usize, // <<< RENAMED from loop_indent
        final_output_buffer_id: ?usize = null, // Store ID of the presumed output buffer

        // Define local errors
        pub const RendererError = error{
            InvalidOperation,
            EmptyUOpList,
            InvalidLastUOp,
            OutputBufferNotFound,
            InputBufferNotFound,
            UnsupportedUOp, // Added this error
            VariableNotFound, // Added for missing IDs in ptr_map
            InvalidArgs, // Added for invalid args in render_mulacc
            OutOfMemory, // Added for memory allocation failures
            ShapeInfoNotFound,
        };

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, writer: WriterType) *Self {
            const self = allocator.create(Self) catch unreachable;
            self.* = Self{
                .allocator = allocator,
                .writer = writer,
                .rendered_ids = std.AutoHashMap(usize, void).init(allocator),
                .view_map = std.AutoHashMap(usize, ViewInfo).init(allocator),
                .buffer_map = std.AutoHashMap(usize, BufferInfo).init(allocator),
                .indent_level = 0, // <<< RENAMED from loop_indent
            };
            return self;
        }

        pub fn deinit(self: *Self) void {
            // NEW: Free buffer info copies
            var iter = self.buffer_map.valueIterator();
            while (iter.next()) |info| {
                // Free the strings/slices allocated in identify_buffers
                self.allocator.free(info.name);
                self.allocator.free(info.shape);
            }

            self.buffer_map.deinit();
            self.rendered_ids.deinit();
            self.view_map.deinit();
            self.allocator.destroy(self);
        }

        // NEW: Helper to populate buffer_map
        fn identify_buffers(self: *Self, uops_list: []const UOp, input_ids: []const usize) !void {
            self.buffer_map.clearRetainingCapacity();

            // Add explicit entries for input IDs
            for (input_ids) |in_id| {
                // --- Find defining UOp for input type/shape ---
                var input_dtype: DType = .f32; // Default placeholder
                var input_shape: []const usize = &[_]usize{}; // Default placeholder
                var defining_op_found = false;

                // Search for VIEW or DEFINE_GLOBAL associated with this input ID
                // VIEW is more likely for inputs used in operations.
                for (uops_list) |uop| {
                    // Check if this UOp *is* the input DEFINE_GLOBAL
                    if (uop.id == in_id and uop.op == .DEFINE_GLOBAL) { // <<< Check op type first
                        if (uop.arg) |arg| { // <<< THEN check and unwrap arg
                            if (arg == .shape) input_shape = arg.shape;
                        }
                        input_dtype = uop.dtype;
                        defining_op_found = true;
                        break;
                        // Check if this UOp is a VIEW *using* the input ID as its source
                    } else if (uop.op == .VIEW and uop.src.len > 0 and uop.src[0] == in_id) {
                        if (uop.arg) |arg| {
                            if (arg == .view_meta) input_shape = arg.view_meta.shape;
                        }
                        input_dtype = uop.dtype; // Use VIEW's dtype
                        defining_op_found = true;
                        break;
                    } // TODO: Add other cases? LOAD maybe?
                }
                // If not found, log warning? Use placeholders.
                // if (!defining_op_found) { std.log.warn("Could not find defining op for input {d}", .{in_id}); }
                // --- End find defining UOp ---

                const name = try std.fmt.allocPrint(self.allocator, "input_{d}", .{in_id});
                // Duplicate the found shape (if any)
                const shape_copy = if (input_shape.len == 0) &[_]usize{} else try self.allocator.dupe(usize, input_shape);
                const needs_free_shape = shape_copy.len > 0;

                var entry = try self.buffer_map.getOrPut(in_id);
                if (!entry.found_existing) {
                    entry.value_ptr.* = BufferInfo{
                        .id = in_id,
                        .name = name,
                        .is_input = true,
                        .dtype = input_dtype, // <<< Use found dtype
                        .shape = shape_copy, // <<< Use found shape
                        .len = 0,
                    };
                } else {
                    // Input ID already existed (shouldn't happen if reset worked?)
                    // Keep existing entry, but ensure is_input=true.
                    if (entry.value_ptr.is_input) {
                        self.allocator.free(name); // Free newly allocated name
                        if (needs_free_shape) self.allocator.free(shape_copy); // Free shape copy too
                    } else {
                        // This case seems problematic - entry exists but wasn't input?
                        // Overwrite cautiously?
                        entry.value_ptr.is_input = true;
                        // Decide if name/shape/dtype should be overwritten or kept.
                        // Forcing name overwrite for consistency:
                        self.allocator.free(entry.value_ptr.name);
                        entry.value_ptr.name = name;
                        // Keep existing dtype/shape for now?
                        if (needs_free_shape) self.allocator.free(shape_copy);
                    }
                }
            }

            // Add/overwrite entries for ALL defining UOps
            for (uops_list) |uop| {
                const entry = try self.buffer_map.getOrPut(uop.id);
                if (!entry.found_existing) {
                    std.debug.print("Found buffer: {any} (id={d})\n", .{ uop.op, uop.id });
                    const is_an_input = for (input_ids) |in_id| {
                        if (uop.id == in_id) break true;
                    } else false;

                    const name_prefix = if (uop.op == .CONST) "const_" else switch (uop.op) {
                        .DEFINE_GLOBAL => if (is_an_input) "input_" else "output_",
                        .GEP => "addr_",
                        .RANGE => "idx_",
                        .DEFINE_ACC => "acc_",
                        // .CONST is handled by the if condition above
                        else => "buf_", // Default for ALU, LOAD, etc.
                    };
                    const name = try std.fmt.allocPrint(self.allocator, "{s}{d}", .{ name_prefix, uop.id });
                    var shape_copy: []const usize = &[_]usize{};
                    var needs_free_shape = false;
                    if (uop.op == .DEFINE_GLOBAL) {
                        if (uop.arg) |arg| {
                            if (arg == .shape) {
                                shape_copy = try self.allocator.dupe(usize, arg.shape);
                                needs_free_shape = shape_copy.len > 0;
                            }
                        }

                        if (!is_an_input) {
                            std.debug.print("Found output buffer: {s} (id={d})\n", .{ name, uop.id });
                            self.final_output_buffer_id = uop.id;
                        }
                    }
                    // Assign:
                    entry.value_ptr.* = BufferInfo{ .id = uop.id, .name = name, .is_input = is_an_input, .dtype = uop.dtype, .shape = shape_copy, .len = 0 };
                } else {
                    // Entry ALREADY exists (Must be an input from first loop, or non-input from prev iter)
                    const existing_info = entry.value_ptr;
                    if (existing_info.is_input) {
                        // Input buffer info was set by first loop.
                        // However, if this UOp (which has an ID of an input) is actually a CONST,
                        // its name should be const_N, not input_N for the purpose of definition.
                        if (uop.op == .CONST) {
                            // Free the old "input_N" name and assign "const_N"
                            self.allocator.free(existing_info.name);
                            const new_name = try std.fmt.allocPrint(self.allocator, "const_{d}", .{uop.id});
                            existing_info.name = new_name;
                            // It's still an input parameter to the function, but its primary definition as a UOp is CONST.
                            // The is_input flag remains true for function signature generation.
                        }
                        // Otherwise, if it's an input and not a CONST, its name "input_N" is correct.
                        continue; // Continue to next UOp, processing for this one is done.
                    } else {
                        // Non-input existing entry (e.g. a CONST already processed with const_ prefix).
                        // Do nothing, let it keep its values.
                        continue;
                    }
                }
            }
        }

        // UNCOMMENTING OLD RENDER METHOD
        pub fn render(self: *@This(), uops: []UOp) !void {
            self.rendered_ids.clearRetainingCapacity();
            for (uops) |uop| {
                // Use the new hasRendered logic
                if (self.hasRendered(uop.id)) continue;
                // NOTE: This old render path needs updating for ptr_map etc.
                // It likely won't produce correct pointer-based code for GEP/LOAD/STORE.
                // Let's call the new render_uop, but need a ptr_map...
                // For now, this old function is likely broken without a proper ptr_map.
                // Consider removing or updating it if needed.
                // Example placeholder call (will error without ptr_map):
                // var dummy_ptr_map = std.AutoHashMap(usize, []const u8).init(self.allocator);
                // defer dummy_ptr_map.deinit();
                // try self.render_uop(uop, &dummy_ptr_map);
                //-- OLD RENDER FUNCTION NEEDS UPDATE FOR PTR_MAP" --
                return error.NotImplemented;
                // Original switch removed as logic moves to render_uop
                // try self.rendered_ids.append(uop.id); // Use put for HashSet
            }
        }

        pub fn reset(self: *Self) void {
            self.buffer_map.clearRetainingCapacity();
            self.rendered_ids.clearRetainingCapacity();
            self.indent_level = 0; // <<< RENAMED from loop_indent
            self.final_output_buffer_id = null; // Reset output buffer ID
        }

        // TODELETE ?? mirko: this function is not used anywere
        // // NEW: Method to render a full function
        // pub fn render_as_function(self: *Self, uops: []const UOp, input_ids: []const usize) !void {
        //     // Reset state
        //     self.reset();

        //     // 1. Identify buffers
        //     try self.identify_buffers(uops, input_ids);

        //     // 2. Get output buffer info
        //     const output_id = self.final_output_buffer_id orelse return error.OutputBufferNotFound;
        //     const output_info = self.buffer_map.get(output_id) orelse return error.OutputBufferNotFound;
        //     const output_type = DTypeInfo.asString(output_info.dtype);

        //     // --- Add std import ---
        //     try self.writer.print("const std = @import(\"std\");\n\n", .{});
        //     // --- End Add std import ---

        //     // 3. Generate function signature
        //     try self.write_function_signature(input_ids, output_type);

        //     // 4. Generate buffer allocations
        //     try self.write_buffer_allocations(output_id);

        //     // 5. Create ptr_map for kernel rendering
        //     var ptr_map = std.AutoHashMap(usize, []const u8).init(self.allocator);
        //     defer {
        //         self.free_ptr_map_values(&ptr_map);
        //         ptr_map.deinit();
        //     }

        //     // Populate ptr_map with buffer names
        //     var map_iter = self.buffer_map.iterator();
        //     while (map_iter.next()) |entry| {
        //         try ptr_map.put(entry.key_ptr.*, entry.value_ptr.name);
        //     }

        //     // 6. Render kernel body
        //     try self.render_kernel_body(uops, &ptr_map);

        //     // 7. Generate return statement
        //     try self.writer.print("    return {s};\n}}\n", .{output_info.name});
        // }

        pub fn render_body(self: *Self, uops: []const UOp, input_ids: []const usize) !void {
            // Reset state
            self.reset();

            // 1. Identify buffers
            try self.identify_buffers(uops, input_ids);

            // 2. Get output buffer info
            const output_id = self.final_output_buffer_id orelse return error.OutputBufferNotFound;

            // 4. Generate buffer allocations
            try self.write_buffer_allocations(output_id);

            // 5. Create ptr_map for kernel rendering
            var ptr_map = std.AutoHashMap(usize, []const u8).init(self.allocator);
            defer {
                self.free_ptr_map_values(&ptr_map);
                ptr_map.deinit();
            }

            // Populate ptr_map with buffer names
            var map_iter = self.buffer_map.iterator();
            while (map_iter.next()) |entry| {
                try ptr_map.put(entry.key_ptr.*, entry.value_ptr.name);
            }

            // 6. Render kernel body
            try self.render_kernel_body(uops, &ptr_map);
        }

        fn write_function_signature(self: *Self, input_ids: []const usize, output_type: []const u8) !void {
            try self.writer.print("pub fn generated_kernel(allocator: std.mem.Allocator", .{});

            var sorted_inputs_list: std.ArrayList(usize) = .empty;
            defer sorted_inputs_list.deinit(self.allocator);
            try sorted_inputs_list.appendSlice(self.allocator, input_ids);
            std.mem.sort(usize, sorted_inputs_list.items, {}, std.sort.asc(usize));

            for (sorted_inputs_list.items) |id| {
                const info = self.buffer_map.get(id) orelse return RendererError.InputBufferNotFound;
                // For function parameters, ALWAYS use "input_N" naming convention.
                // The actual UOp definition (if ID 0 is also a CONST) will use "const_0".
                var param_name_buf: [32]u8 = undefined;
                const param_name = try std.fmt.bufPrint(&param_name_buf, "input_{d}", .{id});

                try self.writer.print(", {s}: []const {s}", .{ param_name, DTypeInfo.asString(info.dtype) });
            }

            try self.writer.print(") ![]{s} {{\n", .{output_type});
        }

        fn write_buffer_allocations(self: *Self, output_id: usize) !void {
            var alloc_iter = self.buffer_map.iterator();
            while (alloc_iter.next()) |entry| {
                const id = entry.key_ptr.*;
                const info = entry.value_ptr.*;

                // Skip inputs, loop vars, addresses, accumulators, and intermediate buffers
                // Explicitly skip if is_input flag is true OR if the name starts with "input_"
                // This provides redundancy in case the is_input flag logic has a subtle bug.
                if (info.is_input or std.mem.startsWith(u8, info.name, "input_") or
                    std.mem.startsWith(u8, info.name, "idx_") or
                    std.mem.startsWith(u8, info.name, "addr_") or
                    std.mem.startsWith(u8, info.name, "acc_") or
                    std.mem.startsWith(u8, info.name, "const_") or // <<< ADDED: Skip const_ for allocations
                    std.mem.startsWith(u8, info.name, "buf_"))
                {
                    continue;
                }

                // Calculate buffer size
                var size: usize = 1;
                if (info.shape.len > 0) {
                    for (info.shape) |dim| size *= dim;
                } else if (info.len > 0) {
                    size = info.len;
                }

                // Allocate buffer
                try self.writer.print("    const {s} = try allocator.alloc({s}, {d});\n", .{ info.name, DTypeInfo.asString(info.dtype), size });

                // Add defer free if not the output buffer
                if (id != output_id) {
                    try self.writer.print("    defer allocator.free({s});\n", .{info.name});
                }
            }
        }

        fn free_ptr_map_values(self: *Self, ptr_map: *std.AutoHashMap(usize, []const u8)) void {
            var ptr_map_value_iter = ptr_map.valueIterator();
            while (ptr_map_value_iter.next()) |var_name_ptr| {
                const var_name = var_name_ptr.*;
                // Free only dynamically allocated names not managed by buffer_map
                if (std.mem.startsWith(u8, var_name, "acc_") or
                    std.mem.startsWith(u8, var_name, "view_") or
                    std.mem.startsWith(u8, var_name, "shape_"))
                {
                    self.allocator.free(var_name);
                }
            }
        }

        fn render_kernel_body(self: *Self, uops: []const UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            for (uops) |uop| {
                if (uop.op == .DEFINE_GLOBAL) continue;
                if (self.rendered_ids.contains(uop.id)) continue;
                try self.render_uop(uop, ptr_map);
            }
        }

        // Helper to extract ID from names like "buf_123"
        fn getIdFromName(name: []const u8) usize {
            var parts = std.mem.splitScalar(u8, name, '_');
            if (parts.next() == null) return std.math.maxInt(usize); // Error case
            const id_str = parts.next() orelse return std.math.maxInt(usize);
            return std.fmt.parseInt(usize, id_str, 10) catch std.math.maxInt(usize);
        }

        fn hasRendered(self: *Self, id: usize) bool {
            // Use HashMap contains method
            return self.rendered_ids.contains(id);
        }

        // Helper to check if an ID is in the input list (used by identify_buffers)
        fn is_input(self: *Self, id: usize, input_ids: []const usize) bool {
            _ = self; // unused self ptr
            for (input_ids) |input_id| {
                if (id == input_id) return true;
            }
            return false;
        }

        // NEW: Helper function to render a single UOp
        pub fn render_uop(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            // Apply indentation based on loop depth
            // --- Indentation applied BEFORE specific op rendering, EXCEPT for ENDRANGE/ENDIF ---
            if (uop.op != .ENDRANGE and uop.op != .ENDIF) {
                try self.apply_indentation();
            }

            switch (uop.op) {
                .DEFINE_GLOBAL, .LOAD, .STORE, .CONST => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),

                .GEP => try GepRender.render(self.allocator, self.writer, uop, &self.view_map, &self.buffer_map, ptr_map),

                .RANGE => {
                    try ControlFlowRender.render(self.allocator, self.writer, uop, ptr_map, &self.view_map);
                    self.indent_level += 1; // <<< Use indent_level
                },

                .ENDRANGE => {
                    if (self.indent_level > 0) self.indent_level -= 1; // <<< Decrement first
                    try self.apply_indentation(); // <<< Apply indent for closing brace
                    try ControlFlowRender.render(self.allocator, self.writer, uop, ptr_map, &self.view_map);
                },

                .ADD, .SUB, .MUL, .FDIV, .MAX, .MIN, .CMPLT => try ArithmeticRender.render(self.allocator, self.writer, uop, ptr_map),

                .EXP2, .NEG, .CAST, .CLIP => try UnaryRender.render(self.allocator, self.writer, uop, ptr_map),

                .VIEW => try self.render_view(uop, ptr_map),
                .RESHAPE => try ReshapeRender.render(self.allocator, self.writer, uop, &self.view_map, &self.buffer_map, ptr_map),
                .DEFINE_ACC => try self.render_define_acc(uop, ptr_map),
                .MULACC => try self.render_mulacc(uop, ptr_map),

                .SHAPE => {
                    // Get shape from source operand's buffer or view info
                    const src_id = uop.src[0];
                    var shape: ?[]const usize = null;

                    // Try getting info from view_map first
                    if (self.view_map.get(src_id)) |view_info| {
                        // Access shape through the stored 'arg' field
                        if (view_info.arg == .view_meta) {
                            shape = view_info.arg.view_meta.shape; // Correct access path
                        }
                        // Fallback to buffer_map if source wasn't a VIEW
                    } else if (self.buffer_map.get(src_id)) |buffer_info| {
                        shape = buffer_info.shape;
                    }
                    // If shape is still null here, there was an error finding it, but maybe we don't need to hard error?

                    // Generate a name for the shape constant (for potential use in ptr_map)
                    const shape_name = try std.fmt.allocPrint(self.allocator, "shape_{d}", .{uop.id});
                    // We might still put the *name* in ptr_map if other ops might symbolically refer to it,
                    // but we won't generate the actual const variable in the code.
                    // TODO: Decide if putting the name in ptr_map is useful without the declaration.
                    _ = try ptr_map.put(uop.id, shape_name); // Maybe put? Maybe ignore? Free shape_name?

                    // Add a comment instead to mark that SHAPE was processed
                    // Use separate prints to handle the optional shape type correctly for {any}
                    if (shape) |s| {
                        try self.writer.print("// SHAPE: id={d} src={d} -> shape {any}\n", .{ uop.id, src_id, s });
                    } else {
                        try self.writer.print("// SHAPE: id={d} src={d} -> shape <not found>\n", .{ uop.id, src_id });
                    }
                },

                // --- NEW IF/ENDIF ---
                .IF => {
                    if (uop.src.len != 1) return RendererError.InvalidArgs;
                    const cond_id = uop.src[0];
                    const cond_name = ptr_map.get(cond_id) orelse return RendererError.VariableNotFound;
                    // Condition name should already be a boolean result from CMPLT etc.
                    try self.writer.print("if ({s}) {{\n", .{cond_name});
                    self.indent_level += 1;
                },
                .ENDIF => {
                    if (self.indent_level > 0) self.indent_level -= 1;
                    try self.apply_indentation(); // Indent the closing brace
                    try self.writer.print("}}\n", .{}); // Print closing brace
                },
                // --- END NEW IF/ENDIF ---

                // --- NEW WHERE Handling ---
                .WHERE => {
                    if (uop.src.len != 3) return RendererError.InvalidArgs;
                    const cond_id = uop.src[0];
                    const then_id = uop.src[1];
                    const else_id = uop.src[2];

                    const cond_var = ptr_map.get(cond_id) orelse return RendererError.VariableNotFound;
                    const then_var = ptr_map.get(then_id) orelse return RendererError.VariableNotFound;
                    const else_var = ptr_map.get(else_id) orelse return RendererError.VariableNotFound;
                    const result_var = ptr_map.get(uop.id) orelse return RendererError.VariableNotFound;
                    const type_str = DTypeInfo.asString(uop.dtype);

                    // Generate the ternary expression
                    try self.writer.print("const {s}: {s} = if ({s}) {s} else {s}; // WHERE (uop {d})\n", .{ result_var, type_str, cond_var, then_var, else_var, uop.id });
                    // Add the necessary "_ = &result_var;" to prevent unused variable warnings if needed?
                    // Assuming ptr_map handles registration, but we might need this:
                    try self.writer.print("_ = &{s};\n", .{result_var});
                },
                // --- END NEW WHERE Handling ---

                else => {
                    std.log.err("Rendering not implemented for UOp type: {s} (id: {d})\n", .{ @tagName(uop.op), uop.id });
                    return RendererError.UnsupportedUOp;
                },
            }

            // Mark UOp as rendered
            try self.rendered_ids.put(uop.id, void{});
        }

        fn apply_indentation(self: *Self) !void {
            var i: usize = 0;
            while (i < self.indent_level) : (i += 1) { // <<< Use indent_level
                try self.writer.print("    ", .{});
            }
        }

        fn render_view(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            // Record view metadata
            try ViewManagerModule.manage(uop, &self.view_map);

            // Create symbolic name for SSA id
            const view_name = try std.fmt.allocPrint(self.allocator, "view_{d}", .{uop.id});
            try ptr_map.put(uop.id, view_name);
        }

        fn render_define_acc(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            const dtype_str = DTypeInfo.asString(uop.dtype);
            const acc_name = try std.fmt.allocPrint(self.allocator, "acc_{d}", .{uop.id});
            try ptr_map.put(uop.id, acc_name);

            // --- Handle both explicit and implicit initialization ---
            if (uop.src.len == 1) {
                // Case 1: Explicit initial value (like -inf for MaxPool)
                const init_val_id = uop.src[0];
                const init_val_name = ptr_map.get(init_val_id) orelse return RendererError.VariableNotFound;
                try self.writer.print("var {s}: {s} = {s};\n", .{ acc_name, dtype_str, init_val_name });
            } else if (uop.src.len == 0) {
                // Case 2: Implicit zero initialization (like for MatMul)
                // TODO: Confirm zero is always the correct implicit default for all dtypes?
                try self.writer.print("var {s}: {s} = 0;\n", .{ acc_name, dtype_str });
            } else {
                // Error: Invalid number of sources for DEFINE_ACC
                std.log.err("DEFINE_ACC (id={d}) has invalid src count: {d}", .{ uop.id, uop.src.len });
                return RendererError.InvalidArgs;
            }
        }

        fn render_mulacc(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            if (uop.src.len != 3) return error.InvalidArgs;

            const acc_id = uop.src[0];
            const a_id = uop.src[1];
            const b_id = uop.src[2];

            // Get variable names from pointer map
            const acc_var = ptr_map.get(acc_id) orelse return RendererError.VariableNotFound;
            const a_var = ptr_map.get(a_id) orelse return RendererError.VariableNotFound;
            const b_var = ptr_map.get(b_id) orelse return RendererError.VariableNotFound;

            // Generate accumulation statement
            try self.writer.print("{s} += {s} * {s};\n", .{ acc_var, a_var, b_var });
        }
    };
}
