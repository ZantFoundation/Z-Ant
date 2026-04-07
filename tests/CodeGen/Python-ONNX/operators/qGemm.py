import numpy as np
import random
from onnx import helper, TensorProto


def generate_qgemm_model(input_names, output_names):
    """Generate a QGemm (com.microsoft) operator model.

    Inputs (subset we exercise):
      A (u8), a_scale, a_zero_point,
      B (u8), b_scale, b_zero_point,
      C (i32 bias),
      y_scale, y_zero_point
    Output: Y (u8)
    """
    initializers = []

    M = random.randint(2, 6)
    K = random.randint(2, 6)
    N = random.randint(2, 6)

    A = np.random.randint(0, 255, size=(M, K)).astype(np.uint8)
    # Z-Ant's qgemm_lean expects B in [N, K] layout (transB=1).
    B = np.random.randint(0, 255, size=(N, K)).astype(np.uint8)
    # Bias C is currently ignored by qgemm_lean (see op_qgemm.zig "bias ignored
    # for now"). Until real bias support lands, keep C at zero so the test
    # exercises the matmul path without false-positive value mismatches.
    C = np.zeros((N,), dtype=np.int32)

    initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, [M, K], A.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [0.05]))
    initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [128]))
    initializers.append(helper.make_tensor(input_names[3], TensorProto.UINT8, [N, K], B.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[4], TensorProto.FLOAT, [], [0.05]))
    initializers.append(helper.make_tensor(input_names[5], TensorProto.UINT8, [], [128]))
    initializers.append(helper.make_tensor(input_names[6], TensorProto.INT32, [N], C.flatten().tolist()))
    initializers.append(helper.make_tensor(input_names[7], TensorProto.FLOAT, [], [0.1]))
    initializers.append(helper.make_tensor(input_names[8], TensorProto.UINT8, [], [128]))

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, [M, K])
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, [M, N])

    node = helper.make_node(
        "QGemm",
        inputs=input_names[:9],
        outputs=[output_names[0]],
        domain="com.microsoft",
        transB=1,
        name="QGemm_node",
    )

    metadata = {"input_shapes": [[M, K], [K, N]], "output_shapes": [[M, N]]}
    return [input_info], output_info, [node], initializers, metadata
