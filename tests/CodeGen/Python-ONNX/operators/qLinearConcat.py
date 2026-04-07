import numpy as np
import random
from onnx import helper, TensorProto


def generate_qlinearconcat_model(input_names, output_names):
    """Generate a QLinearConcat operator model as a composite.

    ORT's contrib QLinearConcat shape inference is finicky about its input
    layout, so — following the qLinearAdd convention — we emit the equivalent
    standard composite (DequantizeLinear x N -> Concat -> QuantizeLinear).
    Z-Ant's fusion engine can collapse this back to QLinearConcat if it
    recognises the pattern; either way the runtime values match the ORT
    reference exactly.
    """
    initializers = []

    n_inputs = 2
    axis = 1
    base_shape = [1, random.randint(2, 4), random.randint(4, 8)]

    # Output scale / zero point.
    out_scale = float(np.random.uniform(0.005, 0.05))
    out_zp = int(np.random.randint(0, 256))

    nodes = []
    dequant_outputs = []

    for i in range(n_inputs):
        idx = i * 3
        x_name = input_names[idx]
        s_name = input_names[idx + 1]
        z_name = input_names[idx + 2]
        x_data = np.random.randint(0, 256, size=base_shape, dtype=np.uint8)
        x_scale = float(np.random.uniform(0.005, 0.05))
        x_zp = int(np.random.randint(0, 256))

        initializers.append(helper.make_tensor(x_name, TensorProto.UINT8, base_shape, x_data.flatten().tolist()))
        initializers.append(helper.make_tensor(s_name, TensorProto.FLOAT, [], [x_scale]))
        initializers.append(helper.make_tensor(z_name, TensorProto.UINT8, [], [x_zp]))

        deq_name = f"qlconcat_deq_{i}_{random.randint(1000, 9999)}"
        nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=[x_name, s_name, z_name],
            outputs=[deq_name],
            name=f"QLConcat_DQ_{i}",
        ))
        dequant_outputs.append(deq_name)

    concat_name = f"qlconcat_concat_{random.randint(1000, 9999)}"
    nodes.append(helper.make_node(
        "Concat",
        inputs=dequant_outputs,
        outputs=[concat_name],
        axis=axis,
        name="QLConcat_Concat",
    ))

    out_scale_name = input_names[n_inputs * 3]
    out_zp_name = input_names[n_inputs * 3 + 1]
    initializers.append(helper.make_tensor(out_scale_name, TensorProto.FLOAT, [], [out_scale]))
    initializers.append(helper.make_tensor(out_zp_name, TensorProto.UINT8, [], [out_zp]))

    nodes.append(helper.make_node(
        "QuantizeLinear",
        inputs=[concat_name, out_scale_name, out_zp_name],
        outputs=[output_names[0]],
        name="QLConcat_Q",
    ))

    out_shape = list(base_shape)
    out_shape[axis] = base_shape[axis] * n_inputs

    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, base_shape)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, out_shape)

    metadata = {"input_shapes": [base_shape] * n_inputs, "output_shapes": [out_shape]}
    return [input_info], output_info, nodes, initializers, metadata
