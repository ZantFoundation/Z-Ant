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
const ViewInfo = ViewManagerModule.ViewInfo;

const Uops = zant.uops;
const DTypeInfo = Uops.DTypeInfo;
const DType = Uops.DType;

// NEW: Structure to hold buffer info (inputs/outputs)
const BufferInfo = struct {
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
        loop_indent: usize,
        final_output_buffer_id: ?usize = null, // Store ID of the presumed output buffer

        // Define local errors
        pub const RendererError = error{
            InvalidOperation,
            EmptyUOpList,
            InvalidLastUOp,
            OutputBufferNotFound,
            InputBufferNotFound,
            UnsupportedUOp, // Added this error
        };

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, writer: WriterType) *Self {
            const self = allocator.create(Self) catch unreachable;
            self.* = Self{
                .allocator = allocator,
                .writer = writer,
                .rendered_ids = std.AutoHashMap(usize, void).init(allocator), // Initialize HashMap
                .view_map = std.AutoHashMap(usize, ViewInfo).init(allocator),
                .buffer_map = std.AutoHashMap(usize, BufferInfo).init(allocator),
                .loop_indent = 0,
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
            // Clear existing map (deinit frees old entries)
            var iter = self.buffer_map.valueIterator();
            while (iter.next()) |info| {
                self.allocator.free(info.name);
                self.allocator.free(info.shape);
            }
            self.buffer_map.clearRetainingCapacity();

            // Add explicit entries for input IDs
            for (input_ids) |in_id| {
                // We don't know shape/dtype from ID alone, need to find first usage?
                // Or assume they are passed externally. For now, create placeholder.
                // TODO: Find a way to get dtype/shape for inputs if needed here.
                const name = try std.fmt.allocPrint(self.allocator, "input_{d}", .{in_id});
                // Use getOrPut in case an input ID is also defined later (e.g., as DEFINE_GLOBAL)
                var entry = try self.buffer_map.getOrPut(in_id);
                if (!entry.found_existing) {
                    entry.value_ptr.* = BufferInfo{
                        .id = in_id,
                        .name = name,
                        .is_input = true,
                        .dtype = .f32, // Placeholder!
                        .shape = &[_]usize{}, // Placeholder!
                        .len = 0, // Initialize len
                    };
                } else {
                    // Input ID already exists (maybe from a later DEFINE_GLOBAL)
                    // Keep existing entry but mark as input and potentially update name.
                    // If the existing name wasn't an input name, free it.
                    if (!entry.value_ptr.is_input and !std.mem.startsWith(u8, entry.value_ptr.name, "input_")) {
                        self.allocator.free(entry.value_ptr.name);
                        entry.value_ptr.name = name; // Use the new input name
                    } else {
                        self.allocator.free(name); // Free the newly allocated name
                    }
                    entry.value_ptr.is_input = true;
                }
            }

            // Add/overwrite entries for ALL defining UOps
            for (uops_list) |uop| {
                // Determine if this uop defines a buffer/value
                const defines_buffer = switch (uop.op) {
                    .DEFINE_GLOBAL,
                    .LOAD,
                    .CONST,
                    .CAST,
                    .RANGE,
                    .DEFINE_ACC,
                    .ADD,
                    .SUB,
                    .MUL,
                    .FDIV,
                    .POW,
                    .EXP2,
                    .NEG,
                    .MAX,
                    .MIN,
                    .CLIP,
                    .CMPLT,
                    .WHERE,
                    .MULACC,
                    .REDUCE_ADD,
                    .REDUCE_MAX,
                    .COPY,
                    .RESHAPE,
                    .PAD,
                    .PERMUTE,
                    .EXPAND,
                    .GEP, // GEP defines a pointer/address variable
                    => true,
                    // Ops that do not define a buffer/value
                    .STORE, .VIEW, .ENDRANGE, .IF, .ENDIF, .SHAPE, .TILE_M, .TILE_N, .VECTORIZE, .UNROLL_K, .FUSE => false,
                };

                if (!defines_buffer) continue;

                // Check if it's an input ID (already handled partially)
                var is_an_input = false;
                for (input_ids) |input_id| {
                    if (uop.id == input_id) {
                        is_an_input = true;
                        break;
                    }
                }

                // Generate name and shape based on op type
                var name: []const u8 = undefined;
                var shape_copy: []const usize = undefined;
                var needs_free_name = true; // Flag to track if generated name needs freeing
                var needs_free_shape = true; // Flag to track if generated shape needs freeing

                // Determine name prefix based on UOp type
                const name_prefix = switch (uop.op) {
                    .DEFINE_GLOBAL => if (is_an_input) "input_" else "output_",
                    .GEP => "addr_",
                    .RANGE => "idx_",
                    else => "buf_", // Default for value buffers
                };
                name = try std.fmt.allocPrint(self.allocator, "{s}{d}", .{ name_prefix, uop.id });

                if (uop.op == .DEFINE_GLOBAL) {
                    shape_copy = if (uop.arg) |arg| switch (arg) {
                        .shape => |s| try self.allocator.dupe(usize, s),
                        else => &[_]usize{}, // Placeholder
                    } else &[_]usize{}; // Placeholder
                    if (shape_copy.len == 0) needs_free_shape = false; // Don't free static empty slice
                } else {
                    // For other ops, generate a temporary buffer name
                    // Use placeholder shape unless specific logic is added (e.g., for CONST scalar)
                    shape_copy = &[_]usize{}; // Placeholder shape
                    needs_free_shape = false; // Don't free static empty slice
                    if (is_an_input) {
                        // If it's also an input, prefer the input name later
                        self.allocator.free(name);
                        needs_free_name = false;
                    }
                }

                // Use getOrPut to add/update the entry
                var entry = try self.buffer_map.getOrPut(uop.id);

                if (entry.found_existing) {
                    // Entry exists (could be an input placeholder or from DEFINE_GLOBAL)
                    // Free the newly generated name/shape if we're not using them
                    if (needs_free_name) self.allocator.free(name);
                    if (needs_free_shape) self.allocator.free(shape_copy);

                    // Update existing entry if needed (e.g., placeholder input getting real info)
                    if (entry.value_ptr.is_input and entry.value_ptr.shape.len == 0) {
                        // It was an input placeholder, update with potentially better info
                        entry.value_ptr.dtype = uop.dtype;
                        // Only update shape if the new one isn't placeholder
                        if (shape_copy.len > 0 and needs_free_shape) { // Check if it's a non-empty, allocated shape
                            self.allocator.free(entry.value_ptr.shape); // Free old placeholder shape
                            entry.value_ptr.shape = shape_copy; // Use the new shape
                        } else if (shape_copy.len > 0) {
                            // New shape is static but non-empty (shouldn't happen with current logic?)
                            entry.value_ptr.shape = shape_copy;
                        }
                        // Keep is_input = true and the input name
                    } else if (!entry.value_ptr.is_input and uop.op == .DEFINE_GLOBAL) {
                        // Existing non-input overwritten by DEFINE_GLOBAL, update everything
                        self.allocator.free(entry.value_ptr.name); // Free old name
                        self.allocator.free(entry.value_ptr.shape); // Free old shape
                        entry.value_ptr.name = name; // Use new name from DEFINE_GLOBAL
                        entry.value_ptr.shape = shape_copy; // Use new shape from DEFINE_GLOBAL
                        entry.value_ptr.dtype = uop.dtype;
                        entry.value_ptr.is_input = is_an_input; // Mark as input only if in input_ids
                        needs_free_name = false; // Name ownership transferred
                        needs_free_shape = false; // Shape ownership transferred
                        // If this is a non-input DEFINE_GLOBAL, store its ID
                        if (!is_an_input) {
                            self.final_output_buffer_id = uop.id;
                        }
                    }
                    // Add other update logic if necessary (e.g., non-DEFINE_GLOBAL updating a placeholder)
                } else {
                    // New entry
                    entry.value_ptr.* = BufferInfo{
                        .id = uop.id,
                        .name = name, // Use generated name
                        .is_input = is_an_input,
                        .dtype = uop.dtype,
                        .shape = shape_copy, // Use generated shape or placeholder
                        .len = 0, // Initialize len
                    };
                    // If this is a non-input DEFINE_GLOBAL, store its ID
                    if (uop.op == .DEFINE_GLOBAL and !is_an_input) {
                        self.final_output_buffer_id = uop.id;
                    }
                    // Prevent double-free if name/shape ownership wasn't transferred
                    // Correct logic: If ownership is NOT transferred (needs_free_* is false),
                    // the struct already has the correct pointer/slice from the initial assignment.
                    // If ownership IS transferred (needs_free_* is true), we don't need to do anything
                    // as the struct now owns the allocated memory.
                    // The needs_free_* flags are used above when deciding whether to free *newly* allocated memory
                    // when an *existing* entry is found.
                    // if (!needs_free_name) entry.value_ptr.name = name; else needs_free_name = true;
                    // if (!needs_free_shape) entry.value_ptr.shape = shape_copy; else needs_free_shape = true;
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
                @panic("Old render function needs update for ptr_map");
                // Original switch removed as logic moves to render_uop
                // try self.rendered_ids.append(uop.id); // Use put for HashSet
            }
        }

        // NEW: Method to render a full function
        pub fn render_as_function(self: *Self, uops: []const UOp, input_ids: []const usize) !void {
            // Reset state for new rendering
            self.buffer_map.clearRetainingCapacity();
            self.rendered_ids.clearRetainingCapacity();
            self.loop_indent = 0;
            self.final_output_buffer_id = null; // Reset output buffer ID
            var output_name_for_return: ?[]const u8 = null;
            var output_type_str: []const u8 = "void"; // Default to void

            // 1. Identify buffers and build buffer_map
            try self.identify_buffers(uops, input_ids);

            // --- NEW: Explicitly find the non-input DEFINE_GLOBAL for output ---
            // Get the output ID stored by identify_buffers
            const final_output_buffer_id = self.final_output_buffer_id orelse return error.OutputBufferNotFound; // Use stored ID

            // Get info for the identified output buffer
            const output_info = self.buffer_map.get(final_output_buffer_id) orelse return error.OutputBufferNotFound;
            output_name_for_return = output_info.name; // Assign name for return
            output_type_str = DTypeInfo.asString(output_info.dtype); // Assign type for signature

            // 2. Generate Function Signature using determined output type
            try self.writer.print("pub fn generated_kernel(allocator: std.mem.Allocator", .{});
            // Add input parameters to signature, sorted by ID for consistency
            var sorted_input_ids = std.ArrayList(usize).init(self.allocator);
            defer sorted_input_ids.deinit();
            try sorted_input_ids.appendSlice(input_ids);
            std.mem.sort(usize, sorted_input_ids.items, {}, std.sort.asc(usize));

            for (sorted_input_ids.items) |id| {
                const info = self.buffer_map.get(id) orelse return RendererError.InputBufferNotFound;
                // Assume inputs are const slices for now
                try self.writer.print(", {s}: []const {s}", .{ info.name, DTypeInfo.asString(info.dtype) });
            }
            // Return type is a slice of the output buffer's type
            try self.writer.print(") ![]{s} {{\n", .{output_type_str});

            // 3. Generate Allocations and Defers (excluding inputs and the final output)
            var alloc_iter = self.buffer_map.iterator();
            while (alloc_iter.next()) |entry| {
                const id = entry.key_ptr.*;
                const info = entry.value_ptr.*;
                // Skip inputs (already parameters)
                if (info.is_input) continue;
                // Skip the final output buffer (it's the return value, allocated separately if needed? No, kernel allocates it)
                // if (id == output_buffer_id) continue; // KEEP the final output buffer allocation for now!
                // Skip loop variables (RANGE UOps)
                if (std.mem.startsWith(u8, info.name, "idx_")) continue;
                // Skip address variables (GEP UOps)
                if (std.mem.startsWith(u8, info.name, "addr_")) continue;

                // Allocate buffer (including intermediates and the final output)
                const type_str = DTypeInfo.asString(info.dtype);
                var size: usize = 1; // Calculate size based on shape
                if (info.shape.len > 0) { // Check if shape is available
                    for (info.shape) |dim| size *= dim;
                } else {
                    // Default size or error? Need to decide based on how shape info is populated
                    // For now, assume a default size might be needed if shape isn't always set
                    size = 1; // Placeholder: Get size from UOp if possible?
                    std.debug.print("WARN: Buffer {s} (id {d}) has no shape, allocating size 1\n", .{ info.name, id });
                    // Maybe use info.len if populated? Check identify_buffers
                    if (info.len > 0) size = info.len;
                }

                try self.writer.print("    const {s} = try allocator.alloc({s}, {d});\n", .{ info.name, type_str, size });

                // Add defer free ONLY if it's NOT the final output buffer
                if (id != final_output_buffer_id) {
                    try self.writer.print("    defer allocator.free({s});\n", .{info.name});
                }
            }

            // 4. Create Pointer Map for Kernel Rendering (maps buffer ID to its variable name)
            var ptr_map = std.AutoHashMap(usize, []const u8).init(self.allocator);
            defer ptr_map.deinit();
            var map_iter = self.buffer_map.iterator();
            while (map_iter.next()) |entry| {
                // Access key via ptr, value via ptr, then value's name field
                try ptr_map.put(entry.key_ptr.*, entry.value_ptr.name);
            }

            // 5. Render Kernel Body (UOps)
            for (uops) |uop| {
                if (uop.op == .DEFINE_GLOBAL) continue; // Skip globals in function body
                // Check if already rendered (using HashMap now)
                if (self.rendered_ids.contains(uop.id)) continue;

                // Call the dedicated render function for the UOp
                try self.render_uop(uop, &ptr_map);
                // No need to mark rendered here, render_uop does it
            }

            // 6. Generate Return Statement using the identified output buffer name
            if (output_name_for_return) |name| {
                try self.writer.print("    return {s};\n", .{name});
            } else {
                // This case should ideally not be reached if output_buffer_id was found
                try self.writer.print("    // Error: No output buffer name found for return statement\n", .{});
            }

            // 7. Closing Brace
            try self.writer.print("}}\n", .{});
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
        fn render_uop(self: *Self, uop: UOp, ptr_map: *std.AutoHashMap(usize, []const u8)) !void {
            // Apply indentation based on loop depth
            var i: usize = 0;
            while (i < self.loop_indent) : (i += 1) {
                try self.writer.print("    ", .{});
            }

            // Render based on UOp type
            switch (uop.op) {
                .DEFINE_GLOBAL => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),
                .LOAD => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),
                .STORE => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),
                .GEP => try GepRender.render(self.allocator, self.writer, uop, self.view_map, ptr_map),
                .RANGE => {
                    try ControlFlowRender.render(self.allocator, self.writer, uop, ptr_map);
                    self.loop_indent += 1;
                },
                .ENDRANGE => {
                    // Dedent *before* printing the closing brace
                    if (self.loop_indent > 0) {
                        self.loop_indent -= 1;
                        // Re-apply indent for the closing brace itself
                        i = 0;
                        while (i < self.loop_indent) : (i += 1) {
                            try self.writer.print("    ", .{});
                        }
                    }
                    try ControlFlowRender.render(self.allocator, self.writer, uop, ptr_map);
                },
                .ADD, .SUB, .MUL, .FDIV, .MAX, .MIN, .CMPLT => try ArithmeticRender.render(self.allocator, self.writer, uop, ptr_map),
                .EXP2, .NEG, .CAST => try UnaryRender.render(self.allocator, self.writer, uop, ptr_map),
                .CONST => try MemoryRender.render(self.allocator, self.writer, uop, ptr_map),
                .VIEW => try ViewManagerModule.manage(uop, &self.view_map),
                // TODO: Add rendering for other UOpTypes (REDUCE, CONDITIONAL, etc.)
                // .REDUCE_ADD, .REDUCE_MAX => try ReduceRender.render(...),
                // .IF, .ENDIF, .WHERE => try ConditionalRender.render(...),
                // .DEFINE_ACC, .MULACC => ??? (Need specific renderers)
                else => {
                    std.log.err("Rendering not implemented for UOp type: {s} (id: {d})\n", .{ @tagName(uop.op), uop.id });
                    return RendererError.UnsupportedUOp;
                },
            }

            // Mark UOp as rendered *after* successful rendering
            // Need putNoClobber or similar if an op could somehow be rendered twice validly?
            // Using put for now, assuming each ID corresponds to one render action.
            try self.rendered_ids.put(uop.id, {});
        }
    };
}
