import numpy as np
import random
from onnx import helper, TensorProto


def generate_matmulinteger_model(input_names, output_names):
    """
    Generates a MatMulInteger operator model.
    """
    initializers = []

    m_val = random.randint(2, 8)
    k_val = random.randint(2, 8)
    n_val = random.randint(2, 8)

    a_shape = [m_val, k_val]
    b_shape = [k_val, n_val]
    output_shape = [m_val, n_val]

    a_data = np.random.randint(0, 256, size=a_shape, dtype=np.uint8)
    b_data = np.random.randint(-128, 128, size=b_shape, dtype=np.int8)
    a_zero_point = np.random.randint(0, 256, dtype=np.uint8)
    b_zero_point = np.int8(0)

    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, a_shape, a_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.INT8, b_shape, b_data.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [a_zero_point]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.INT8, [], [b_zero_point]))

    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.INT32, output_shape)
    node = helper.make_node(
        "MatMulInteger",
        inputs=[input_names[0], input_names[1], input_names[2], input_names[3]],
        outputs=[output_names[0]],
        name="MatMulInteger_node",
    )

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, a_shape)
    metadata = {
        "input_shapes": [a_shape, b_shape],
        "output_shapes": [output_shape],
        "a_zero_point": int(a_zero_point),
        "b_zero_point": int(b_zero_point),
        "output_type": "int32",
    }

    return [input_info], output_info, [node], initializers, metadata
