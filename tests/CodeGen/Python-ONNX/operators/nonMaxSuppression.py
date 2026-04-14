import numpy as np
import random
from onnx import helper, TensorProto


def generate_nonmaxsuppression_model(input_names, output_names):
    """Generate a NonMaxSuppression operator model.

    Inputs:
      boxes:           [num_batches, spatial_dim, 4]
      scores:          [num_batches, num_classes, spatial_dim]
      max_output_boxes_per_class: scalar int64
      iou_threshold:   scalar float
      score_threshold: scalar float
    Output:
      selected_indices: [num_selected, 3]  (dynamic) — declared as [-1, 3]
    """
    initializers = []

    num_batches = 1
    num_classes = 1
    spatial = random.randint(4, 8)

    # Boxes in [y1, x1, y2, x2] format with y2>y1 and x2>x1.
    boxes_data = np.zeros((num_batches, spatial, 4), dtype=np.float32)
    for i in range(spatial):
        y1 = random.uniform(0.0, 0.5)
        x1 = random.uniform(0.0, 0.5)
        boxes_data[0, i] = [y1, x1, y1 + random.uniform(0.1, 0.5), x1 + random.uniform(0.1, 0.5)]
    initializers.append(helper.make_tensor(input_names[0], TensorProto.FLOAT, [num_batches, spatial, 4], boxes_data.flatten().tolist()))

    scores_data = np.random.rand(num_batches, num_classes, spatial).astype(np.float32)
    initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [num_batches, num_classes, spatial], scores_data.flatten().tolist()))

    max_per_class = spatial
    initializers.append(helper.make_tensor(input_names[2], TensorProto.INT64, [], [max_per_class]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.FLOAT, [], [0.5]))
    initializers.append(helper.make_tensor(input_names[4], TensorProto.FLOAT, [], [0.0]))

    # Worst-case output rows = batches * classes * max_per_class. Use a fixed
    # 2D shape so the Z-Ant codegen allocates the right buffer; the runtime
    # caps actual writes to whatever NMS produces.
    max_rows = num_batches * num_classes * max_per_class
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, [num_batches, spatial, 4])
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.INT64, [max_rows, 3])

    node = helper.make_node(
        "NonMaxSuppression",
        inputs=input_names[:5],
        outputs=[output_names[0]],
        name="NonMaxSuppression_node",
    )

    metadata = {"input_shapes": [[num_batches, spatial, 4]], "output_shapes": [[max_rows, 3]]}
    return [input_info], output_info, [node], initializers, metadata
