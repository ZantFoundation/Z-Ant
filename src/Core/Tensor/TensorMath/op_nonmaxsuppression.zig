const std = @import("std");
const zant = @import("../../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

// --------------------- NONMAXSUPPRESSION OPERATOR ---------------------

/// Box structure for NMS
const Box = struct {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: usize,
    batch_id: usize,
    original_index: usize,
};

/// Computes the output shape for the NonMaxSuppression operator.
/// Output shape: [num_selected_indices, 3] where 3 represents [batch_index, class_index, box_index]
pub fn get_nonmaxsuppression_output_shape(
    boxes_shape: []const usize,
    scores_shape: []const usize,
    max_output_boxes_per_class: i64,
    center_point_box: i64,
) ![]usize {
    _ = center_point_box;

    if (boxes_shape.len != 3 or scores_shape.len != 3) {
        return TensorMathError.InvalidInput;
    }

    const num_batches = boxes_shape[0];
    const num_classes = scores_shape[1];
    const max_boxes = @as(usize, @intCast(@max(0, max_output_boxes_per_class)));

    // Maximum possible output size
    const max_output_size = num_batches * num_classes * max_boxes;

    const output_shape = try pkg_allocator.alloc(usize, 2);
    output_shape[0] = max_output_size;
    output_shape[1] = 3;
    return output_shape;
}

/// Computes intersection over union (IoU) between two boxes
fn computeIoU(box1: Box, box2: Box) f32 {
    const x1 = @max(box1.x1, box2.x1);
    const y1 = @max(box1.y1, box2.y1);
    const x2 = @min(box1.x2, box2.x2);
    const y2 = @min(box1.y2, box2.y2);

    if (x2 <= x1 or y2 <= y1) {
        return 0.0;
    }

    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union_area = area1 + area2 - intersection;

    if (union_area <= 0.0) {
        return 0.0;
    }

    return intersection / union_area;
}

/// Applies Non-Maximum Suppression, allocating a new output tensor.
pub fn nonmaxsuppression(
    comptime T: type,
    boxes: *const Tensor(T),
    scores: *const Tensor(T),
    max_output_boxes_per_class: i64,
    iou_threshold: f32,
    score_threshold: f32,
    center_point_box: i64,
) !Tensor(i64) {
    // Validate inputs
    if (boxes.shape.len != 3 or scores.shape.len != 3) {
        return TensorMathError.InvalidInput;
    }

    // Compute output shape
    const output_shape = try get_nonmaxsuppression_output_shape(
        boxes.shape,
        scores.shape,
        max_output_boxes_per_class,
        center_point_box,
    );
    defer pkg_allocator.free(output_shape);

    // Allocate output tensor
    var output = try Tensor(i64).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    const actual_size = try nonmaxsuppression_lean(
        T,
        boxes,
        scores,
        &output,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        center_point_box,
    );

    // Resize output tensor to actual size
    if (actual_size < output.shape[0]) {
        const new_shape = try pkg_allocator.alloc(usize, 2);
        new_shape[0] = actual_size;
        new_shape[1] = 3;
        output.shape = new_shape;
        output.size = actual_size * 3;
    }

    return output;
}

/// Applies Non-Maximum Suppression on pre-allocated output tensor.
/// Returns the actual number of selected boxes.
pub fn nonmaxsuppression_lean(
    comptime T: type,
    boxes: *const Tensor(T),
    scores: *const Tensor(T),
    output: *Tensor(i64),
    max_output_boxes_per_class: i64,
    iou_threshold: f32,
    score_threshold: f32,
    center_point_box: i64,
) !usize {
    const num_batches = boxes.shape[0];
    const num_boxes = boxes.shape[1];
    const num_classes = scores.shape[1];
    const max_boxes_per_class = @as(usize, @intCast(@max(0, max_output_boxes_per_class)));

    var output_count: usize = 0;

    // Process each batch and class
    for (0..num_batches) |batch_idx| {
        for (0..num_classes) |class_idx| {
            // Collect boxes for this batch and class
            var candidate_boxes = std.ArrayList(Box).init(pkg_allocator);
            defer candidate_boxes.deinit();

            for (0..num_boxes) |box_idx| {
                const score_idx = batch_idx * num_classes * num_boxes + class_idx * num_boxes + box_idx;
                const score = @as(f32, @floatCast(scores.data[score_idx]));

                if (score > score_threshold) {
                    const box_base_idx = batch_idx * num_boxes * 4 + box_idx * 4;

                    var box: Box = undefined;
                    if (center_point_box == 1) {
                        // Center point format: [center_x, center_y, width, height]
                        const center_x = @as(f32, @floatCast(boxes.data[box_base_idx]));
                        const center_y = @as(f32, @floatCast(boxes.data[box_base_idx + 1]));
                        const width = @as(f32, @floatCast(boxes.data[box_base_idx + 2]));
                        const height = @as(f32, @floatCast(boxes.data[box_base_idx + 3]));

                        box.x1 = center_x - width / 2.0;
                        box.y1 = center_y - height / 2.0;
                        box.x2 = center_x + width / 2.0;
                        box.y2 = center_y + height / 2.0;
                    } else {
                        // Corner format: [x1, y1, x2, y2]
                        box.x1 = @as(f32, @floatCast(boxes.data[box_base_idx]));
                        box.y1 = @as(f32, @floatCast(boxes.data[box_base_idx + 1]));
                        box.x2 = @as(f32, @floatCast(boxes.data[box_base_idx + 2]));
                        box.y2 = @as(f32, @floatCast(boxes.data[box_base_idx + 3]));
                    }

                    box.score = score;
                    box.class_id = class_idx;
                    box.batch_id = batch_idx;
                    box.original_index = box_idx;

                    try candidate_boxes.append(box);
                }
            }

            // Sort boxes by score (descending)
            std.sort.block(Box, candidate_boxes.items, {}, struct {
                fn lessThan(_: void, lhs: Box, rhs: Box) bool {
                    return lhs.score > rhs.score;
                }
            }.lessThan);

            // Apply NMS
            var selected_boxes = std.ArrayList(Box).init(pkg_allocator);
            defer selected_boxes.deinit();

            for (candidate_boxes.items) |candidate| {
                if (selected_boxes.items.len >= max_boxes_per_class) {
                    break;
                }

                var is_suppressed = false;
                for (selected_boxes.items) |selected| {
                    const iou = computeIoU(candidate, selected);
                    if (iou > iou_threshold) {
                        is_suppressed = true;
                        break;
                    }
                }

                if (!is_suppressed) {
                    try selected_boxes.append(candidate);
                }
            }

            // Write selected boxes to output
            for (selected_boxes.items) |box| {
                if (output_count < output.shape[0]) {
                    const base_idx = output_count * 3;
                    output.data[base_idx] = @as(i64, @intCast(box.batch_id));
                    output.data[base_idx + 1] = @as(i64, @intCast(box.class_id));
                    output.data[base_idx + 2] = @as(i64, @intCast(box.original_index));
                    output_count += 1;
                }
            }
        }
    }

    return output_count;
}
