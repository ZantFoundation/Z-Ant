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

//https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html
// INPUTS:
//      - boxes (heterogeneous) - tensor(float): Boxes with shape [num_batches, spatial_dimension, 4]
//      - scores (heterogeneous) - tensor(float): Scores with shape [num_batches, num_classes, spatial_dimension]
//      - max_output_boxes_per_class (optional) - tensor(int64): Maximum number of output boxes per class
//      - iou_threshold (optional) - tensor(float): IoU threshold
//      - score_threshold (optional) - tensor(float): Score threshold
// OUTPUTS:
//      - selected_indices (heterogeneous) - tensor(int64): Selected indices with shape [num_selected_indices, 3]
pub const NonMaxSuppression = struct {
    boxes: *TensorZant,
    scores: *TensorZant,
    max_output_boxes_per_class: ?*TensorZant,
    iou_threshold: ?*TensorZant,
    score_threshold: ?*TensorZant,
    output: *TensorZant,

    // Attributes
    center_point_box: i64,

    pub fn init(nodeProto: *NodeProto) !NonMaxSuppression {
        const boxes = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.boxes_notFound;
        const scores = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.scores_notFound;

        // Optional inputs
        const max_output_boxes_per_class = if (nodeProto.input.len > 2 and nodeProto.input[2].len > 0)
            tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])
        else
            null;

        const iou_threshold = if (nodeProto.input.len > 3 and nodeProto.input[3].len > 0)
            tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])
        else
            null;

        const score_threshold = if (nodeProto.input.len > 4 and nodeProto.input[4].len > 0)
            tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])
        else
            null;

        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        // Get attributes
        var center_point_box: i64 = 0; // Default value
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "center_point_box")) {
                center_point_box = attr.i;
            }
        }

        // Set output type
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = tensorZant_lib.TensorType.i64;

        return NonMaxSuppression{
            .boxes = boxes,
            .scores = scores,
            .max_output_boxes_per_class = max_output_boxes_per_class,
            .iou_threshold = iou_threshold,
            .score_threshold = score_threshold,
            .output = output,
            .center_point_box = center_point_box,
        };
    }

    pub fn get_output_shape(self: NonMaxSuppression) []usize {
        return self.compute_output_shape() catch {
            // Fallback to a default shape in case of error
            const fallback_shape = allocator.alloc(usize, 2) catch unreachable;
            fallback_shape[0] = 2048; // max boxes
            fallback_shape[1] = 3; // [batch_idx, class_idx, box_idx]
            return fallback_shape;
        };
    }

    pub fn get_input_tensors(self: NonMaxSuppression) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.boxes);
        try inputs.append(self.scores);
        if (self.max_output_boxes_per_class) |tensor| try inputs.append(tensor);
        if (self.iou_threshold) |tensor| try inputs.append(tensor);
        if (self.score_threshold) |tensor| try inputs.append(tensor);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: NonMaxSuppression) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: NonMaxSuppression, writer: std.fs.File.Writer) !void {
        // Create tensor strings for inputs
        var boxes_string: []u8 = undefined;
        defer allocator.free(boxes_string);
        if (self.boxes.tc == TensorCategory.INITIALIZER) {
            boxes_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.boxes.name),
                ")",
            });
        } else {
            boxes_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.boxes.name) });
        }

        var scores_string: []u8 = undefined;
        defer allocator.free(scores_string);
        if (self.scores.tc == TensorCategory.INITIALIZER) {
            scores_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.scores.name),
                ")",
            });
        } else {
            scores_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.scores.name) });
        }

        // Generate parameter access code or use defaults
        const max_output_boxes = if (self.max_output_boxes_per_class) |tensor| blk: {
            if (tensor.tc == TensorCategory.INITIALIZER) {
                break :blk try std.mem.concat(allocator, u8, &[_][]const u8{
                    "param_lib.tensor_",
                    try utils.getSanitizedName(tensor.name),
                    ".data[0]",
                });
            } else {
                break :blk try std.mem.concat(allocator, u8, &[_][]const u8{
                    "tensor_",
                    try utils.getSanitizedName(tensor.name),
                    ".data[0]",
                });
            }
        } else try std.mem.concat(allocator, u8, &[_][]const u8{"2048"});
        defer allocator.free(max_output_boxes);

        const iou_thresh = if (self.iou_threshold) |tensor| blk: {
            if (tensor.tc == TensorCategory.INITIALIZER) {
                break :blk try std.mem.concat(allocator, u8, &[_][]const u8{
                    "param_lib.tensor_",
                    try utils.getSanitizedName(tensor.name),
                    ".data[0]",
                });
            } else {
                break :blk try std.mem.concat(allocator, u8, &[_][]const u8{
                    "tensor_",
                    try utils.getSanitizedName(tensor.name),
                    ".data[0]",
                });
            }
        } else try std.mem.concat(allocator, u8, &[_][]const u8{"0.5"});
        defer allocator.free(iou_thresh);

        const score_thresh = if (self.score_threshold) |tensor| blk: {
            if (tensor.tc == TensorCategory.INITIALIZER) {
                break :blk try std.mem.concat(allocator, u8, &[_][]const u8{
                    "param_lib.tensor_",
                    try utils.getSanitizedName(tensor.name),
                    ".data[0]",
                });
            } else {
                break :blk try std.mem.concat(allocator, u8, &[_][]const u8{
                    "tensor_",
                    try utils.getSanitizedName(tensor.name),
                    ".data[0]",
                });
            }
        } else try std.mem.concat(allocator, u8, &[_][]const u8{"0.0"});
        defer allocator.free(score_thresh);

        _ = try writer.print(
            \\
            \\    _ = tensMath.nonmaxsuppression_lean(
            \\      {s},
            \\      {s},
            \\      {s},
            \\      &tensor_{s},
            \\      {s},
            \\      {s},
            \\      {s},
            \\      {d},
            \\    ) catch return -1;
        ,
            .{
                self.boxes.ty.toString(),
                boxes_string,
                scores_string,
                try utils.getSanitizedName(self.output.name),
                max_output_boxes,
                iou_thresh,
                score_thresh,
                self.center_point_box,
            },
        );
    }

    pub fn sobstitute_tensors(self: *NonMaxSuppression, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.boxes == old_tensor) {
            self.boxes = new_tensor;
            return;
        }
        if (self.scores == old_tensor) {
            self.scores = new_tensor;
            return;
        }
        if (self.max_output_boxes_per_class != null and self.max_output_boxes_per_class.? == old_tensor) {
            self.max_output_boxes_per_class = new_tensor;
            return;
        }
        if (self.iou_threshold != null and self.iou_threshold.? == old_tensor) {
            self.iou_threshold = new_tensor;
            return;
        }
        if (self.score_threshold != null and self.score_threshold.? == old_tensor) {
            self.score_threshold = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn compute_output_shape(self: NonMaxSuppression) ![]usize {
        // For compute_output_shape, we use a default max value since we can't access tensor data at compile time
        const max_output_boxes_val: i64 = 2048; // Use default for shape computation

        const output_shape = try tensorMath.get_nonmaxsuppression_output_shape(
            self.boxes.shape,
            self.scores.shape,
            max_output_boxes_val,
            self.center_point_box,
        );
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: NonMaxSuppression) void {
        std.debug.print("\n NonMaxSuppression: {any}", .{self});
    }

    pub fn render_lower(self: NonMaxSuppression, builder: *UOpBuilder) !void {
        // NonMaxSuppression is complex and typically implemented as a library call
        // For now, we'll create a placeholder that calls the lean function
        const boxes_id = self.boxes.get_tensorZantID();
        const scores_id = self.scores.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const out_dtype = utils.tensorTypeToDtype(self.output.ty);

        const out_buf_id = lowerNonMaxSuppression(
            builder,
            boxes_id,
            scores_id,
            out_shape,
            out_dtype,
        );
        _ = out_buf_id;
    }

    /// Lower NonMaxSuppression to UOps (simplified version - actual implementation would be very complex)
    pub fn lowerNonMaxSuppression(
        b: *UOpBuilder,
        boxes_id: usize,
        scores_id: usize,
        out_shape: []const usize,
        out_dtype: DType,
    ) usize {
        // For complex operations like NMS, we typically create a library call
        // This is a simplified placeholder that creates the output buffer
        _ = boxes_id;
        _ = scores_id;

        const id_Y = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = out_shape });

        // In a real implementation, we would:
        // 1. Create complex loops over batches and classes
        // 2. Implement IoU computation
        // 3. Implement sorting and suppression logic
        // For now, this is a placeholder

        return id_Y;
    }
};
