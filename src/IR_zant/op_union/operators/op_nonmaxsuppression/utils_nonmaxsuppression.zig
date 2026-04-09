const zant = @import("zant");

const pkg_allocator = zant.utils.allocator.allocator;

/// Computes the output shape for the NonMaxSuppression operator.
/// Output shape: [num_selected_indices, 3] where 3 represents [batch_index, class_index, box_index]
pub fn get_non_max_suppression_output_shape(
    boxes_shape: []const usize,
    scores_shape: []const usize,
    max_output_boxes_per_class: i64,
    center_point_box: i64,
) ![]usize {
    _ = center_point_box;

    if (boxes_shape.len != 3 or scores_shape.len != 3) {
        return error.InvalidInput;
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
