const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

// --- uops ---
const cg_v2 = @import("codegen").codegen_v2;
const Uops = cg_v2.uops;
const UOpBuilder = cg_v2.builder;
const DType = Uops.DType;
const Any = Uops.Any;

//https://onnx.ai/onnx/operators/onnx__TopK.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
//      - K (heterogeneous) - tensor(int64): Number of top elements to retrieve
// OUTPUTS:
//      - Values (heterogeneous) - T: Top K values from the input tensor
//      - Indices (heterogeneous) - tensor(int64): Indices of the top K values
pub const TopK = struct {
    input_X: *TensorZant,
    input_K: *TensorZant,
    output_values: *TensorZant,
    output_indices: *TensorZant,

    // Attributes
    axis: i64,
    largest: bool,
    sorted: bool,

    pub fn init(nodeProto: *NodeProto) !TopK {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_K = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_K_notFound;
        const output_values = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_values_notFound;
        const output_indices = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[1])) |ptr| ptr else return error.output_indices_notFound;

        // Get attributes with defaults
        var axis: i64 = -1; // Default to last axis
        var largest: bool = true; // Default to largest
        var sorted: bool = true; // Default to sorted

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                axis = attr.i;
            } else if (std.mem.eql(u8, attr.name, "largest")) {
                largest = attr.i != 0;
            } else if (std.mem.eql(u8, attr.name, "sorted")) {
                sorted = attr.i != 0;
            }
        }

        // Set output types
        if (output_values.ty == tensorZant_lib.TensorType.undefined) output_values.ty = input_X.ty;
        if (output_indices.ty == tensorZant_lib.TensorType.undefined) output_indices.ty = tensorZant_lib.TensorType.i64;

        return TopK{
            .input_X = input_X,
            .input_K = input_K,
            .output_values = output_values,
            .output_indices = output_indices,
            .axis = axis,
            .largest = largest,
            .sorted = sorted,
        };
    }

    pub fn get_output_shape(self: TopK) []usize {
        return self.compute_output_shape() catch {
            // Fallback to a default shape in case of error
            std.log.warn("[TOPK DEBUG] Failed to compute output shape, using fallback", .{});
            const fallback_shape = allocator.alloc(usize, 1) catch unreachable;
            fallback_shape[0] = 1;
            return fallback_shape;
        };
    }

    pub fn get_input_tensors(self: TopK) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_X);
        try inputs.append(self.input_K);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: TopK) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_values);
        try outputs.append(self.output_indices);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: TopK, writer: std.fs.File.Writer) !void {
        // Create tensor strings for inputs
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);
        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        var tensor_K_string: []u8 = undefined;
        defer allocator.free(tensor_K_string);
        if (self.input_K.tc == TensorCategory.INITIALIZER) {
            tensor_K_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_K.name),
                ".data[0]",
            });
        } else {
            tensor_K_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", try utils.getSanitizedName(self.input_K.name), ".data[0]" });
        }

        _ = try writer.print(
            \\
            \\    {{
            \\        const k_val = @as(usize, @intCast({s}));
            \\        const topk_result = tensMath.topk_lean(
            \\            {s},
            \\            {s},
            \\            &tensor_{s},
            \\            &tensor_{s},
            \\            k_val,
            \\            {d},
            \\            {s},
            \\            {s},
            \\        ) catch return -1;
            \\        _ = topk_result;
            \\    }}
        ,
            .{
                tensor_K_string,
                self.input_X.ty.toString(),
                tensor_X_string,
                try utils.getSanitizedName(self.output_values.name),
                try utils.getSanitizedName(self.output_indices.name),
                self.axis,
                if (self.largest) "true" else "false",
                if (self.sorted) "true" else "false",
            },
        );
    }

    pub fn compute_output_shape(self: TopK) ![]usize {
        // For compute_output_shape, we use a default k value since we can't access tensor data at compile time
        const k_val: usize = 1; // Use default for shape computation
        const output_shapes = try tensorMath.get_topk_output_shape(
            self.input_X.shape,
            k_val,
            self.axis,
        );

        self.output_values.shape = output_shapes.values_shape;
        self.output_indices.shape = output_shapes.indices_shape;

        return output_shapes.values_shape;
    }

    pub fn print(self: TopK) void {
        std.debug.print("\n TopK: {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *TopK, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.input_K == old_tensor) {
            self.input_K = new_tensor;
            return;
        }
        if (self.output_values == old_tensor) {
            self.output_values = new_tensor;
            return;
        }
        if (self.output_indices == old_tensor) {
            self.output_indices = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn render_lower(self: TopK, builder: *UOpBuilder) !void {
        // TopK is complex and typically implemented as a library call
        // For now, we'll create a placeholder that calls the lean function
        const X_id = self.input_X.get_tensorZantID();
        const K_id = self.input_K.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const out_dtype = utils.tensorTypeToDtype(self.output_values.ty);

        const result_ids = lowerTopK(
            builder,
            X_id,
            K_id,
            out_shape,
            out_dtype,
            self.axis,
            self.largest,
            self.sorted,
        );
        _ = result_ids;
    }

    /// Lower TopK to UOps (simplified version - actual implementation would be very complex)
    pub fn lowerTopK(
        b: *UOpBuilder,
        X_id: usize,
        K_id: usize,
        out_shape: []const usize,
        out_dtype: DType,
        axis: i64,
        largest: bool,
        sorted: bool,
    ) struct { values_id: usize, indices_id: usize } {
        // For complex operations like TopK, we typically create a library call
        // This is a simplified placeholder that creates the output buffers
        _ = X_id;
        _ = K_id;
        _ = axis;
        _ = largest;
        _ = sorted;

        const values_id = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });
        const indices_id = b.push(.DEFINE_GLOBAL, .i64, &.{}, Any{ .shape = out_shape });

        // In a real implementation, we would:
        // 1. Create complex loops over all dimensions except the specified axis
        // 2. Implement sorting along the axis
        // 3. Extract top-k elements and their indices
        // For now, this is a placeholder

        return .{ .values_id = values_id, .indices_id = indices_id };
    }
};
