import numpy as np
import random
from onnx import helper, TensorProto


def generate_log_model(input_names, output_names):
    """
    Generates a Log operator model.
    Notes:
      - Log requires strictly positive inputs; we sample data in (0.1, 3.0].
      - We keep the same return signature used for Elu: 
        ([input_value_infos], output_value_info, [nodes], initializers, metadata)
    """
    initializers = []

    # generate 1D tensor with strictly positive values
    shape = [random.randint(1, 10)]
    data = np.random.uniform(low=0.1, high=3.0, size=shape).astype(np.float32)

    # model uses the tensor as an initializer (constant input)
    init_tensor = helper.make_tensor(
        input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist()
    )
    initializers.append(init_tensor)

    # keep the dummy input info for consistency with existing generators
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

    # single Log node (no attributes)
    node = helper.make_node(
        "Log",
        inputs=[input_names[0]],
        outputs=[output_names[0]],
        name="Log_node",
    )

    # minimal metadata
    metadata = {
        "input_shapes": [shape],
        "output_shapes": [shape],
        "domain_constraints": {"min_input_value": 0.1},
        "dtype": "FLOAT",
        "op": "Log",
    }

    return [input_info], output_info, [node], initializers, metadata
