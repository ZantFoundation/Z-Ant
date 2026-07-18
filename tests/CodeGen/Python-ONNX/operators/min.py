import numpy as np
import random
from onnx import helper, TensorProto


def generate_min_model(input_names, output_names):
    """Generate Min operator model (element-wise minimum of two tensors)."""
    initializers = []

    shape = [1, random.randint(1, 4), random.randint(10, 50), random.randint(10, 50)]
    data0 = np.random.randn(*shape).astype(np.float32)
    data1 = np.random.randn(*shape).astype(np.float32)

    init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
    init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape, data1.flatten().tolist())
    initializers.extend([init_tensor0, init_tensor1])

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

    node = helper.make_node("Min", inputs=[input_names[0], input_names[1]], outputs=[output_names[0]], name="Min_node")

    metadata = {"input_shapes": [shape, shape], "output_shapes": [shape]}
    return [input_info], output_info, [node], initializers, metadata
