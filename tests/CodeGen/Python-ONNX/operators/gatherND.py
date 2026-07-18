import numpy as np
import random
from onnx import helper, TensorProto


def generate_gathernd_model(input_names, output_names):
    """Generate a GatherND operator model (batch_dims=0, indices index full slices)."""
    initializers = []

    # 2D data tensor; indices select complete rows.
    shape = [random.randint(4, 8), random.randint(4, 8)]
    data = np.random.randn(*shape).astype(np.float32)
    init_data = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_data)

    # indices shape: [Q, 1] meaning Q indices each pointing to one row -> output shape [Q, shape[1]]
    q = random.randint(1, 4)
    idx = np.random.randint(0, shape[0], size=(q, 1)).astype(np.int64)
    init_idx = helper.make_tensor(input_names[1], TensorProto.INT64, [q, 1], idx.flatten().tolist())
    initializers.append(init_idx)

    out_shape = [q, shape[1]]
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)

    node = helper.make_node("GatherND", inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], name="GatherND_node")

    metadata = {"input_shapes": [shape, [q, 1]], "output_shapes": [out_shape]}
    return [input_info], output_info, [node], initializers, metadata
