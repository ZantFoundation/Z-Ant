import numpy as np
import random
from onnx import helper, TensorProto


def generate_exp_model(input_names, output_names):
    """Generate Exp operator model."""
    initializers = []

    shape = [1, random.randint(1, 4), random.randint(5, 20), random.randint(5, 20)]
    # Keep magnitudes small so exp() doesn't overflow during ORT validation.
    data = (np.random.randn(*shape) * 0.5).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

    node = helper.make_node("Exp", inputs=[input_names[0]], outputs=[output_names[0]], name="Exp_node")
    metadata = {"input_shapes": [shape], "output_shapes": [shape]}
    return [input_info], output_info, [node], initializers, metadata
