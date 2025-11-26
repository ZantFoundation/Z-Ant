import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

from qLinearConv import generate_qlinearconv_nchw_model


def _random_input(shape, elem_type):
    """Genera un tensore random del tipo richiesto."""
    if elem_type == TensorProto.FLOAT:
        return np.random.randn(*shape).astype(np.float32)
    elif elem_type == TensorProto.INT64:
        return np.random.randint(0, 10, size=shape, dtype=np.int64)
    elif elem_type == TensorProto.UINT8:
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)
    elif elem_type == TensorProto.INT8:
        return np.random.randint(-128, 128, size=shape, dtype=np.int8)
    elif elem_type == TensorProto.INT32:
        return np.random.randint(-1000, 1000, size=shape, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported input type: {elem_type}")


def run_single_onnx(model_path: Path) -> Dict[str, Any]:
    """Carica il modello, genera input random per gli input non-initializer e fa inference."""
    model = onnx.load(model_path)
    graph = model.graph

    initializer_names = {init.name for init in graph.initializer}
    runtime_inputs = [inp for inp in graph.input if inp.name not in initializer_names]

    inputs_data = {}
    for inp in runtime_inputs:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        elem_type = inp.type.tensor_type.elem_type
        data = _random_input(shape, elem_type)
        inputs_data[inp.name] = data

    sess = ort.InferenceSession(str(model_path))
    output_names = [out.name for out in graph.output]
    outputs = sess.run(output_names, inputs_data)

    inputs_list = {k: v.tolist() for k, v in inputs_data.items()}
    outputs_list = {name: out.tolist() for name, out in zip(output_names, outputs)}

    return {
        "inputs": inputs_list,
        "outputs": outputs_list,
    }


def main():
    # ---------------------------------------------------
    # Parametri fissati nel codice
    # ---------------------------------------------------
    output_dir = Path("Models/")
    model_name = "QLinearConv_0"
    seed = 42

    batch_size = 1
    in_channels = 3
    out_channels = 6
    input_height = 100
    input_width = 100
    kernel_h = 5
    kernel_w = 5
    group = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"
    json_path = output_dir / f"{model_name}_metadata.json"

    # 1) genera modello QLinearConv NCHW
    model, meta = generate_qlinearconv_nchw_model(
        input_name=f"{model_name}_input",
        output_name=f"{model_name}_output",
        batch_size=batch_size,
        in_channels=in_channels,
        input_height=input_height,
        input_width=input_width,
        out_channels=out_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        group=group,
        stride_h=stride_h,
        stride_w=stride_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        seed=seed,
    )

    # 2) salva ONNX
    onnx.save(model, str(onnx_path))
    print(f"Modello salvato in: {onnx_path}")

    # 3) inference
    io_data = run_single_onnx(onnx_path)

    # 4) salva metadata + input/output
    record = {
        "operation": "QLinearConv",
        "model_name": model_name,
        "onnx_path": str(onnx_path),
        "inputs": io_data["inputs"],
        "outputs": io_data["outputs"],
        "metadata": meta,
    }

    with json_path.open("w") as f:
        json.dump(record, f, indent=2)

    print(f"Metadata + input/output salvati in: {json_path}")


if __name__ == "__main__":
    main()