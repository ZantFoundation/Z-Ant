import numpy as np
import random
from onnx import helper, TensorProto


def generate_abs_model(input_names, output_names):
    """
    Generates a Abs operator model.
    """
    initializers = []
    
    # Operatori a singolo input con forma casuale (rank=4)
    shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
    data = np.random.randn(*shape).astype(np.float32)
    init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
    initializers.append(init_tensor)

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

    node = helper.make_node("Abs", inputs=[input_names[0]], outputs=[output_names[0]], 
                            name=f"Abs_node")
    metadata = {"input_shapes": [shape], "output_shapes": [shape]}
    return [input_info], output_info, [node], initializers, metadata