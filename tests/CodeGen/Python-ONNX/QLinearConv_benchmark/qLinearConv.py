import random
from typing import Sequence, Tuple, Optional, Dict, Any

import numpy as np
from onnx import helper, TensorProto, ModelProto


def _compute_output_hw(
    in_h: int,
    in_w: int,
    k_h: int,
    k_w: int,
    s_h: int,
    s_w: int,
    p_h: int,
    p_w: int,
    d_h: int,
    d_w: int,
) -> Tuple[int, int]:
    def out_dim(in_dim: int, k: int, s: int, p: int, d: int) -> int:
        return ((in_dim + 2 * p - d * (k - 1) - 1) // s) + 1

    return out_dim(in_h, k_h, s_h, p_h, d_h), out_dim(in_w, k_w, s_w, p_w, d_w)


def generate_qlinearconv_nchw_model(
    input_name: str,
    output_name: str,
    # shape input: [N, C_in, H, W]
    batch_size: int,
    in_channels: int,
    input_height: int,
    input_width: int,
    # pesi: [C_out, C_in/group, kH, kW]
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    group: int = 1,
    # iperparametri conv
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: Optional[int] = None,  # se None → "same-ish"
    pad_w: Optional[int] = None,
    dilation_h: int = 1,
    dilation_w: int = 1,
    # seed opzionale per rendere ripetibile
    seed: Optional[int] = None,
) -> Tuple[ModelProto, Dict[str, Any]]:
    """
    Genera un modello ONNX con un singolo nodo QLinearConv in convenzione NCHW.

    Convenzioni:
      - input:  [N, C_in, H, W]
      - pesi:   [C_out, C_in/group, kH, kW]
      - output: [N, C_out, H_out, W_out]

    La quantizzazione è generata in modo random
    ma ripetibile usando un seed.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if in_channels % group != 0:
        raise ValueError(f"in_channels ({in_channels}) must be divisible by group ({group})")

    # shape base
    input_shape = [batch_size, in_channels, input_height, input_width]
    weight_shape = [out_channels, in_channels // group, kernel_h, kernel_w]
    bias_shape = [out_channels]

    # padding default "same-ish" se non fornito
    if pad_h is None:
        pad_h = kernel_h // 2
    if pad_w is None:
        pad_w = kernel_w // 2

    # calcolo output
    out_h, out_w = _compute_output_hw(
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
    )
    output_shape = [batch_size, out_channels, out_h, out_w]

    # -------------------------
    # Quantizzazione 
    # -------------------------
    x_scale = np.float32(np.random.uniform(0.001, 0.1))
    x_zero_point = np.uint8(random.randint(0, 255))

    w_scale = np.float32(np.random.uniform(0.001, 0.1))
    w_zero_point = np.uint8(128)  # scelta tipica per pesi

    y_scale = np.float32(np.random.uniform(0.001, 0.1))
    y_zero_point = np.uint8(random.randint(0, 255))

    # weight and bias
    w_data = np.random.randint(0, 256, size=weight_shape, dtype=np.uint8)
    bias_data = np.random.randint(-1000, 1000, size=bias_shape, dtype=np.int32)

    # nomi univoci
    param_id = random.randint(1000, 9999)
    x_scale_name = f"x_scale_{param_id}"
    x_zero_point_name = f"x_zero_point_{param_id}"
    w_name = f"weight_{param_id}"
    w_scale_name = f"w_scale_{param_id}"
    w_zero_point_name = f"w_zero_point_{param_id}"
    y_scale_name = f"y_scale_{param_id}"
    y_zero_point_name = f"y_zero_point_{param_id}"
    bias_name = f"bias_{param_id}"

    initializers = [
        helper.make_tensor(x_scale_name,      TensorProto.FLOAT, [], [x_scale]),
        helper.make_tensor(x_zero_point_name, TensorProto.UINT8, [], [x_zero_point]),
        helper.make_tensor(w_name,            TensorProto.UINT8, weight_shape, w_data.flatten().tolist()),
        helper.make_tensor(w_scale_name,      TensorProto.FLOAT, [], [w_scale]),
        helper.make_tensor(w_zero_point_name, TensorProto.UINT8, [], [w_zero_point]),
        helper.make_tensor(y_scale_name,      TensorProto.FLOAT, [], [y_scale]),
        helper.make_tensor(y_zero_point_name, TensorProto.UINT8, [], [y_zero_point]),
        helper.make_tensor(bias_name,         TensorProto.INT32, bias_shape, bias_data.flatten().tolist()),
    ]

    # value info I/O
    input_info = helper.make_tensor_value_info(input_name, TensorProto.UINT8, input_shape)
    output_info = helper.make_tensor_value_info(output_name, TensorProto.UINT8, output_shape)

    # nodo QLinearConv
    node = helper.make_node(
        "QLinearConv",
        inputs=[
            input_name,
            x_scale_name,
            x_zero_point_name,
            w_name,
            w_scale_name,
            w_zero_point_name,
            y_scale_name,
            y_zero_point_name,
            bias_name,
        ],
        outputs=[output_name],
        name=f"QLinearConv_node_{param_id}",
        dilations=[dilation_h, dilation_w],
        group=group,
        kernel_shape=[kernel_h, kernel_w],
        pads=[pad_h, pad_w, pad_h, pad_w],
        strides=[stride_h, stride_w],
    )

    graph = helper.make_graph(
        [node],
        name=f"QLinearConv_NCHW_bench_{param_id}",
        inputs=[input_info],
        outputs=[output_info],
        initializer=initializers,
    )

    opset_imports = [helper.make_opsetid("", 13)]
    model = helper.make_model(
        graph,
        opset_imports=opset_imports,
        producer_name="zant_qlinearconv_bench",
        ir_version=6,
    )

    metadata = {
        "input_shape": input_shape,
        "weight_shape": weight_shape,
        "bias_shape": bias_shape,
        "output_shape": output_shape,
        "kernel": [kernel_h, kernel_w],
        "stride": [stride_h, stride_w],
        "padding": [pad_h, pad_w],
        "dilation": [dilation_h, dilation_w],
        "group": group,
        "quantization": {
            "x_scale": float(x_scale),
            "x_zero_point": int(x_zero_point),
            "w_scale": float(w_scale),
            "w_zero_point": int(w_zero_point),
            "y_scale": float(y_scale),
            "y_zero_point": int(y_zero_point),
        },
        "param_names": {
            "x": input_name,
            "y": output_name,
            "x_scale": x_scale_name,
            "x_zero_point": x_zero_point_name,
            "w": w_name,
            "w_scale": w_scale_name,
            "w_zero_point": w_zero_point_name,
            "y_scale": y_scale_name,
            "y_zero_point": y_zero_point_name,
            "bias": bias_name,
        },
    }

    return model, metadata