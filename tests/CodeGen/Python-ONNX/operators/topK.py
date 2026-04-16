import numpy as np
import random
from onnx import helper, TensorProto


def generate_topk_model(input_names, output_names):
    """Generate a TopK operator model.

    Note: TopK has two outputs (values, indices). The Z-Ant codegen currently
    requires a single graph output, so we expose only `values` and leave the
    indices output dangling at the node. ORT still validates the node end-to-end.
    """
    initializers = []

    axis = 1
    rows = random.randint(2, 4)
    cols = random.randint(4, 10)
    k = random.randint(1, max(1, cols // 2))

    X = np.random.randn(rows, cols).astype(np.float32)
    initializers.append(helper.make_tensor(input_names[0], TensorProto.FLOAT, [rows, cols], X.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.INT64, [1], [k]))

    out_shape = [rows, k]
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, [rows, cols])
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)

    # We deliberately keep the indices output named (so ORT accepts it) but only
    # expose `values` (output_names[0]) as the graph output above.
    node = helper.make_node(
        "TopK",
        inputs=[input_names[0], input_names[1]],
        outputs=[output_names[0], output_names[1]],
        axis=axis,
        largest=1,
        sorted=1,
        name="TopK_node",
    )

    metadata = {"input_shapes": [[rows, cols]], "output_shapes": [out_shape]}
    return [input_info], output_info, [node], initializers, metadata
