import numpy as np
import random
from onnx import helper, TensorProto


def generate_qlinearmul_model(input_names, output_names):
    """Generate a QLinearMul (com.microsoft) operator model."""
    initializers = []

    shape = [1, random.randint(1, 4), random.randint(4, 8), random.randint(4, 8)]
    A = np.random.randint(0, 255, size=shape).astype(np.uint8)
    B = np.random.randint(0, 255, size=shape).astype(np.uint8)

    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, shape, A.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [0.05]))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [128]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.UINT8, shape, B.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[4], TensorProto.FLOAT, [], [0.05]))
    initializers.append(helper.make_tensor(input_names[5], TensorProto.UINT8, [], [128]))
    initializers.append(helper.make_tensor(input_names[6], TensorProto.FLOAT, [], [0.1]))
    initializers.append(helper.make_tensor(input_names[7], TensorProto.UINT8, [], [128]))

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, shape)

    node = helper.make_node(
        "QLinearMul",
        inputs=input_names[:8],
        outputs=[output_names[0]],
        domain="com.microsoft",
        name="QLinearMul_node",
    )

    metadata = {"input_shapes": [shape, shape], "output_shapes": [shape]}
    return [input_info], output_info, [node], initializers, metadata
