import numpy as np
import random
from onnx import helper, TensorProto


def generate_qlinearsoftmax_model(input_names, output_names):
    """Generate a QLinearSoftmax (com.microsoft) operator model."""
    initializers = []

    shape = [1, random.randint(4, 16)]
    X = np.random.randint(0, 255, size=shape).astype(np.uint8)

    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, shape, X.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [0.05]))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [128]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.FLOAT, [], [1.0 / 256.0]))
    initializers.append(helper.make_tensor(input_names[4], TensorProto.UINT8, [], [0]))

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, shape)

    node = helper.make_node(
        "QLinearSoftmax",
        inputs=input_names[:5],
        outputs=[output_names[0]],
        domain="com.microsoft",
        axis=1,
        opset=13,
        name="QLinearSoftmax_node",
    )

    metadata = {"input_shapes": [shape], "output_shapes": [shape]}
    return [input_info], output_info, [node], initializers, metadata
