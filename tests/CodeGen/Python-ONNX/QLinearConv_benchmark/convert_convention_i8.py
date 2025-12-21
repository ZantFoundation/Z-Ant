import json
from pathlib import Path
from typing import List

import numpy as np
import onnx
from onnx import ModelProto, numpy_helper


def _nchw_to_nhwc_shape(shape: List[int]) -> List[int]:
    """Converte una shape [N, C, H, W] in [N, H, W, C]."""
    if len(shape) != 4:
        return shape
    n, c, h, w = shape
    return [n, h, w, c]


def _nchw_to_nhwc_array(x: np.ndarray) -> np.ndarray:
    """
    Converte un tensore 4D da NCHW a NHWC.
    Se la rank != 4 ritorna invariato.
    """
    if x.ndim != 4:
        return x
    # N,C,H,W -> N,H,W,C
    return np.transpose(x, (0, 2, 3, 1))


def _oihw_to_ohwi_array(w: np.ndarray) -> np.ndarray:
    """
    Converte un tensore pesi da OIHW (out, in, h, w) a OHWI (out, h, w, in),
    che è il layout 'N,H,W,C' usato da CMSIS per i filtri.
    """
    if w.ndim != 4:
        return w
    # O,I,H,W -> O,H,W,I
    return np.transpose(w, (0, 2, 3, 1))


def convert_qlinearconv_nchw_to_nhwc(
    onnx_in: Path,
    meta_in: Path,
    onnx_out: Path,
    meta_out: Path,
) -> None:
    """
    Converte un modello QLinearConv e il suo metadata da NCHW a NHWC.
    """
    # --------------------
    # 1. Carica ONNX + metadata
    # --------------------
    model: ModelProto = onnx.load(onnx_in)
    with meta_in.open("r") as f:
        record = json.load(f)

    meta = record["metadata"]
    input_shape_nchw = meta["input_shape"]
    output_shape_nchw = meta["output_shape"]
    weight_shape_oihw = meta["weight_shape"]

    graph = model.graph
    x_name = meta["param_names"]["x"]
    y_name = meta["param_names"]["y"]
    w_name = meta["param_names"]["w"]

    # --------------------
    # 2. Aggiorna shape I/O nel grafo (NCHW -> NHWC)
    # --------------------
    # Input
    for inp in graph.input:
        if inp.name == x_name:
            dims = inp.type.tensor_type.shape.dim
            old = [d.dim_value for d in dims]
            new = _nchw_to_nhwc_shape(old)
            for d, v in zip(dims, new):
                d.dim_value = v

    # Output
    for out in graph.output:
        if out.name == y_name:
            dims = out.type.tensor_type.shape.dim
            old = [d.dim_value for d in dims]
            new = _nchw_to_nhwc_shape(old)
            for d, v in zip(dims, new):
                d.dim_value = v

    # --------------------
    # 3. Converte i dati input/output nel metadata
    # --------------------
    x_json_name = x_name
    y_json_name = y_name

    # Input
    if x_json_name in record["inputs"]:
        # CORREZIONE QUI: dtype=np.int8 per supportare valori negativi
        x_array = np.array(record["inputs"][x_json_name], dtype=np.int8)
        x_nhwc = _nchw_to_nhwc_array(x_array)
        record["inputs"][x_json_name] = x_nhwc.tolist()

    # Output
    if y_json_name in record["outputs"]:
        # CORREZIONE QUI: dtype=np.int8 per supportare valori negativi
        y_array = np.array(record["outputs"][y_json_name], dtype=np.int8)
        y_nhwc = _nchw_to_nhwc_array(y_array)
        record["outputs"][y_json_name] = y_nhwc.tolist()

    # --------------------
    # 4. Converte i PESI in ONNX: OIHW -> OHWI
    # --------------------
    for i, init in enumerate(graph.initializer):
        if init.name == w_name:
            w_np = numpy_helper.to_array(init)
            w_ohwi = _oihw_to_ohwi_array(w_np)
            # sostituisci initializer con la nuova versione
            graph.initializer.remove(init)
            new_init = numpy_helper.from_array(w_ohwi, name=w_name)
            graph.initializer.insert(i, new_init)
            # aggiorna anche la shape nel value_info se esiste
            for vi in graph.value_info:
                if vi.name == w_name:
                    dims = vi.type.tensor_type.shape.dim
                    new_shape = list(w_ohwi.shape)
                    for d, v in zip(dims, new_shape):
                        d.dim_value = v
            break

    weight_shape_ohwi = [weight_shape_oihw[0], weight_shape_oihw[2], weight_shape_oihw[3], weight_shape_oihw[1]]

    # --------------------
    # 5. Aggiorna metadata shape + layout
    # --------------------
    input_shape_nhwc = _nchw_to_nhwc_shape(input_shape_nchw)
    output_shape_nhwc = _nchw_to_nhwc_shape(output_shape_nchw)

    # Sovrascrivi i campi con la nuova convenzione
    meta["input_shape"] = input_shape_nhwc       
    meta["output_shape"] = output_shape_nhwc        
    meta["weight_shape"] = weight_shape_ohwi        

    meta["layout"] = "NHWC"

    # --------------------
    # 6. Salva nuovo ONNX + nuovo metadata
    # --------------------
    onnx.save(model, str(onnx_out))

    record["metadata"] = meta
    record["onnx_path"] = str(onnx_out)


    with meta_out.open("w") as f:
        json.dump(record, f, indent=2)


def main():
    base_dir = Path("Models")
    # CORREZIONE QUI: Nome aggiornato per puntare al file generato dallo script precedente
    model_name = "QLinearConv_i8_kernel5x5"

    onnx_in = base_dir / f"{model_name}.onnx"
    meta_in = base_dir / f"{model_name}_metadata.json"

    onnx_out = base_dir / f"{model_name}_NHWC.onnx"
    meta_out = base_dir / f"{model_name}_NHWC_metadata.json"

    # Controllo di esistenza per evitare errori
    if not onnx_in.exists():
        print(f"ERRORE: Non trovo il file {onnx_in}. Hai eseguito lo script precedente?")
        return

    convert_qlinearconv_nchw_to_nhwc(onnx_in, meta_in, onnx_out, meta_out)
    print(f"Creati con successo:\n  {onnx_out}\n  {meta_out}")


if __name__ == "__main__":
    main()