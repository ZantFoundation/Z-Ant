import numpy as np
import random
from onnx import helper, TensorProto


def generate_qlinearaveragepool_model(input_names, output_names):
    """Generate a QLinearAveragePool (com.microsoft) operator model."""
    initializers = []

    N, C, H, W = 1, random.randint(1, 4), random.randint(8, 16), random.randint(8, 16)
    kh = kw = 2
    sh = sw = 2

    X = np.random.randint(0, 255, size=(N, C, H, W)).astype(np.uint8)
    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, [N, C, H, W], X.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [0.05]))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [128]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.FLOAT, [], [0.05]))
    initializers.append(helper.make_tensor(input_names[4], TensorProto.UINT8, [], [128]))

    out_h = (H - kh) // sh + 1
    out_w = (W - kw) // sw + 1

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, [N, C, H, W])
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, [N, C, out_h, out_w])

    node = helper.make_node(
        "QLinearAveragePool",
        inputs=input_names[:5],
        outputs=[output_names[0]],
        domain="com.microsoft",
        kernel_shape=[kh, kw],
        strides=[sh, sw],
        name="QLinearAveragePool_node",
    )

    metadata = {"input_shapes": [[N, C, H, W]], "output_shapes": [[N, C, out_h, out_w]]}
    return [input_info], output_info, [node], initializers, metadata
