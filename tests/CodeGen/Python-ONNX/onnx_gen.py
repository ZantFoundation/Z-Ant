import onnx
from onnx import helper, TensorProto
import random
import numpy as np
import argparse
import datetime
from onnx import StringStringEntryProto
import onnxruntime as ort  
import json  
import os  

def random_shape(rank, min_dim=1, max_dim=10):
    """Generates a random shape of length 'rank'."""
    return [random.randint(min_dim, max_dim) for _ in range(rank)]

def generate_fuzz_model(op_name):
    """
    Crea gli initializer, i nodi e gli output con parametri casuali per l'operatore op_name.
    In questo caso tutti gli input vengono inseriti come initializer, per cui la lista degli input
    del grafo viene restituita vuota.
    """
    initializers = []
    
    # Pre-generazione di nomi per input e output
    input_names = [f"{op_name}_param_in_{i}" for i in range(5)]
    output_names = [f"{op_name}_param_out_{i}" for i in range(5)]
    metadata = {}

    if op_name in ["Relu", "Sigmoid", "Ceil", "Tanh", "Identity", "Neg", "Shape"]:
        # Operatori a singolo input con forma casuale (rank=4)
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        # Crea dati casuali e li inserisce come initializer
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "LeakyRelu":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        alpha = round(random.uniform(0.001, 0.2), 3)
        data = np.random.randn(*shape).astype(np.float32)

        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                alpha=alpha, name=f"{op_name}node_alpha{alpha}")
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "alpha": alpha}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Softmax":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(-rank, rank-1)
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                axis=axis, name=f"{op_name}node_axis{axis}")
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "axis": axis}
        return [], output_info, [node], initializers, metadata

    elif op_name in ["Add", "Sub", "Mul", "Div", "Mean"]:
        # Operatori binari: due input della stessa forma
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data0 = np.random.randn(*shape).astype(np.float32)
        data1 = np.random.randn(*shape).astype(np.float32)

        init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
        init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape, data1.flatten().tolist())
        initializers.extend([init_tensor0, init_tensor1])

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, 
                                inputs=[input_names[0], 
                                input_names[1]], 
                                outputs=[output_names[0]],
                                name=f"{op_name}_node")
        
        metadata = {"input_shapes": [shape, shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Concat":
        # Due input con forma identica eccetto per la dimensione lungo l'asse di concatenazione
        shape = [1, random.randint(2,5), random.randint(10,50), random.randint(10,50)]
        rank = len(shape)
        axis = random.randint(0, rank-1)
        shape2 = shape.copy()
        shape2[axis] = shape[axis] + random.randint(1,3)
        data0 = np.random.randn(*shape).astype(np.float32)
        data1 = np.random.randn(*shape2).astype(np.float32)
        init_tensor0 = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data0.flatten().tolist())
        init_tensor1 = helper.make_tensor(input_names[1], TensorProto.FLOAT, shape2, data1.flatten().tolist())
        initializers.extend([init_tensor0, init_tensor1])
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] + shape2[axis]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                axis=axis, name=f"{op_name}node_axis{axis}")
        metadata = {"input_shapes": [shape, shape2], "output_shapes": [out_shape]}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Gather":
        # Primo input: dati; secondo input: indici (inserito come initializer)
        shape = [5, random.randint(5,10)]
        data = np.random.randn(*shape).astype(np.float32)
        init_data = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_data)
        indices_shape = [random.randint(1,3)]
        indices_data = np.random.randint(0, shape[random.randint(0, len(shape)-1)], size=indices_shape).astype(np.int64)
        init_indices = helper.make_tensor(input_names[1], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
        initializers.append(init_indices)
        axis = random.randint(0, len(shape)-1)
        out_shape = list(shape)
        out_shape[axis] = indices_shape[0]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                axis=axis, name=f"{op_name}node_axis{axis}")
        metadata = {"input_shapes": [shape, indices_shape], "output_shapes": [out_shape]}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Pad":
        # Operatore Pad: genera dati, pads e constant_value come initializer
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        pads = [random.randint(0,2) for _ in range(2*rank)]
        out_shape = [shape[i] + pads[i] + pads[i+rank] for i in range(rank)]
        pads_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(pads)], pads)
        initializers.append(pads_tensor)
        constant_value = 0.0
        constant_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [], [constant_value])
        initializers.append(constant_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]],
                                name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "pads": pads, "constant_value": constant_value}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Reshape":
        # Primo input: dati; secondo input: nuovo shape (initializer)
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        new_shape = shape.copy()
        random.shuffle(new_shape)
        shape_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(new_shape)], new_shape)
        initializers.append(shape_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, new_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}_node")
        metadata = {"input_shapes": [shape, new_shape], "output_shapes": [new_shape]}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Resize":
        # Quattro input: X, roi, scales, sizes
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        roi = []
        roi_tensor = helper.make_tensor(input_names[1], TensorProto.FLOAT, [0], roi)
        initializers.append(roi_tensor)
        scales = [round(random.uniform(0.5, 2.0), 2) for _ in shape]
        scales_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [len(scales)], scales)
        initializers.append(scales_tensor)
        sizes = [int(round(s * dim)) for s, dim in zip(scales, shape)]
        sizes_tensor = helper.make_tensor(input_names[3], TensorProto.INT64, [len(sizes)], sizes)
        initializers.append(sizes_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, sizes)
        mode = random.choice(["nearest", "linear"])
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2], input_names[3]],
                                outputs=[output_names[0]], mode=mode, name=f"{op_name}node_mode{mode}")
        metadata = {"input_shapes": [shape], "output_shapes": [sizes], "mode": mode, "scales": scales, "sizes": sizes}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Slice":
        # Primo input: dati; gli altri due (starts ed ends) come initializer
        shape = [random.randint(5,10) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        starts, ends = [], []
        for d in shape:
            start = random.randint(0, d-1)
            end = random.randint(start+1, d)
            starts.append(start)
            ends.append(end)
        starts_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(starts)], starts)
        ends_tensor = helper.make_tensor(input_names[2], TensorProto.INT64, [len(ends)], ends)
        initializers.extend([starts_tensor, ends_tensor])
        out_shape = [ends[i] - starts[i] for i in range(rank)]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]],
                                outputs=[output_names[0]], name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "starts": starts, "ends": ends}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Split":
        # Split in 2 parti lungo un asse casuale
        shape = [1, random.randint(4,10), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        axis = random.randint(0, rank-1)
        if shape[axis] < 2:
            shape[axis] = 2
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] // 2
        output_info = [
            helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info(output_names[1], TensorProto.FLOAT, out_shape)
        ]
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0], output_names[1]],
                                axis=axis, name=f"{op_name}node_axis{axis}")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape, out_shape]}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Transpose":
        # Genera una permutazione casuale per Transpose
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        perm = list(range(rank))
        random.shuffle(perm)
        out_shape = [shape[i] for i in perm]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                perm=perm, name=f"{op_name}node_perm{perm}")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "perm": perm}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Unsqueeze":
        # Inserisce una dimensione in un asse casuale
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        axis = random.randint(0, rank)
        axes = [axis]
        axes_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(axes)], axes)
        initializers.append(axes_tensor)
        out_shape = shape.copy()
        out_shape.insert(axis, 1)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}node_axis{axis}")
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "axes": axes}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Conv":
        # Operatore Conv: genera input e pesi come initializer
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        data = np.random.randn(*input_shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        kH = random.randint(2, max(2, H//2))
        kW = random.randint(2, max(2, W//2))
        kernel_shape = [kH, kW]
        M = random.randint(1,4)
        weight_shape = [M, C, kH, kW]
        weight_data = np.random.randn(*weight_shape).astype(np.float32)
        init_weight = helper.make_tensor(input_names[1], TensorProto.FLOAT, weight_shape, weight_data.flatten().tolist())
        initializers.append(init_weight)
        H_out = H - kH + 1
        W_out = W - kW + 1
        output_shape = [N, M, H_out, W_out]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                kernel_shape=kernel_shape, name=f"{op_name}node_kernel{kernel_shape}")
        metadata = {"input_shapes": [input_shape, weight_shape], "output_shapes": [output_shape], "kernel_shape": kernel_shape}
        return [], output_info, [node], initializers, metadata

    elif op_name == "MatMul":
        # Genera due matrici 2D compatibili
        M_val = random.randint(2,10)
        K_val = random.randint(2,10)
        N_val = random.randint(2,10)
        A_shape = [M_val, K_val]
        B_shape = [K_val, N_val]
        A_data = np.random.randn(*A_shape).astype(np.float32)
        B_data = np.random.randn(*B_shape).astype(np.float32)
        init_tensor_A = helper.make_tensor(input_names[0], TensorProto.FLOAT, A_shape, A_data.flatten().tolist())
        init_tensor_B = helper.make_tensor(input_names[1], TensorProto.FLOAT, B_shape, B_data.flatten().tolist())
        initializers.extend([init_tensor_A, init_tensor_B])
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M_val, N_val])
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                name=f"{op_name}_node")
        metadata = {"input_shapes": [A_shape, B_shape], "output_shapes": [[M_val, N_val]]}
        return [], output_info, [node], initializers, metadata

    elif op_name == "Gemm":
        # Gemm: esegue A * B + C
        M_val = random.randint(2,10)
        K_val = random.randint(2,10)
        N_val = random.randint(2,10)
        A_shape = [M_val, K_val]
        B_shape = [K_val, N_val]
        C_shape = [M_val, N_val]
        A_data = np.random.randn(*A_shape).astype(np.float32)
        B_data = np.random.randn(*B_shape).astype(np.float32)
        C_data = np.random.randn(*C_shape).astype(np.float32)
        init_tensor_A = helper.make_tensor(input_names[0], TensorProto.FLOAT, A_shape, A_data.flatten().tolist())
        init_tensor_B = helper.make_tensor(input_names[1], TensorProto.FLOAT, B_shape, B_data.flatten().tolist())
        init_tensor_C = helper.make_tensor(input_names[2], TensorProto.FLOAT, C_shape, C_data.flatten().tolist())
        initializers.extend([init_tensor_A, init_tensor_B, init_tensor_C])
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, [M_val, N_val])
        alpha = round(random.uniform(0.5, 2.0), 2)
        beta = round(random.uniform(0.5, 2.0), 2)
        transA = random.choice([0, 1])
        transB = random.choice([0, 1])
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]],
                                alpha=alpha, beta=beta, transA=transA, transB=transB,
                                name=f"{op_name}node_alpha{alpha}beta{beta}transA{transA}transB{transB}")
        metadata = {"input_shapes": [A_shape, B_shape, C_shape], "output_shapes": [[M_val, N_val]],
                    "alpha": alpha, "beta": beta, "transA": transA, "transB": transB}
        return [], output_info, [node], initializers, metadata

    elif op_name == "MaxPool":
        # Pooling layer con kernel e stride casuali
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        input_shape = [N, C, H, W]
        data = np.random.randn(*input_shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        kernel_size = random.randint(2, max(2, min(H, W)//2))
        kernel_shape = [kernel_size, kernel_size]
        strides = [random.randint(1, kernel_size), random.randint(1, kernel_size)]
        H_out = (H - kernel_size) // strides[0] + 1
        W_out = (W - kernel_size) // strides[1] + 1
        output_shape = [N, C, H_out, W_out]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                kernel_shape=kernel_shape, strides=strides,
                                name=f"{op_name}node_kernel{kernel_shape}strides{strides}")
        metadata = {"input_shapes": [input_shape], "output_shapes": [output_shape],
                    "kernel_shape": kernel_shape, "strides": strides}
        return [], output_info, [node], initializers, metadata

    else:
        # Caso di fallback per operatori non gestiti esplicitamente
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                name=f"{op_name}_generic_node")
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return [], output_info, [node], initializers, metadata

def generate_model(op_name, filename, model_id=0):
    """Crea e salva un modello ONNX."""
    input_info, output_info, nodes, initializers, metadata = generate_fuzz_model(op_name)
    
    # Gli output vanno sempre definiti
    graph_outputs = output_info if isinstance(output_info, list) else [output_info]
    
    for node in nodes:
        node.doc_string = f"Test node for {op_name} operation with ID {model_id}"
    
    graph = helper.make_graph(
        nodes,
        name=f"{op_name}test_graph{model_id}",
        inputs=input_info,  # In questo caso vuota, poiché tutti gli input sono initializer
        outputs=graph_outputs,
        initializer=initializers,
        doc_string=f"Test graph for {op_name} operation with configuration: {metadata}"
    )
    
    opset_imports = [helper.make_opsetid("", 13)]
    
    model = helper.make_model(
        graph, 
        producer_name='zant_test_generator',
        producer_version='1.0',
        domain='ai.zant.test',
        model_version=model_id,
        doc_string=f"Test model for {op_name} operation. Generated on {datetime.datetime.now().isoformat()}",
        opset_imports=opset_imports
    )
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_metadata"
    meta_prop.value = str(metadata)
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_op"
    meta_prop.value = op_name
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "test_id"
    meta_prop.value = str(model_id)
    model.metadata_props.append(meta_prop)
    
    meta_prop = StringStringEntryProto()
    meta_prop.key = "generator_version"
    meta_prop.value = "1.0"
    model.metadata_props.append(meta_prop)
    
    onnx.checker.check_model(model)
    onnx.save(model, filename)
    print(f"Fuzzed model for {op_name} (ID: {model_id}) saved to: {filename}")
    return metadata

def run_model(filename):
    """
    Esegue il modello ONNX.
    Poiché tutti gli input sono definiti come initializer, il feed del runtime sarà vuoto.
    """
    model = onnx.load(filename)
    graph = model.graph
    # Ottieni i nomi degli initializer (gli input runtime saranno vuoti)
    initializer_names = [init.name for init in graph.initializer]
    runtime_inputs = [inp for inp in graph.input if inp.name not in initializer_names]
    
    input_data = {}
    for inp in runtime_inputs:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        elem_type = inp.type.tensor_type.elem_type
        if elem_type == TensorProto.FLOAT:
            data = np.random.randn(*shape).astype(np.float32)
        elif elem_type == TensorProto.INT64:
            data = np.random.randint(0, 10, size=shape, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported input type: {elem_type}")
        input_data[inp.name] = data.tolist()
    
    session = ort.InferenceSession(filename)
    output_names = [out.name for out in graph.output]
    outputs = session.run(output_names, {name: np.array(data) for name, data in input_data.items()})
    
    outputs_dict = {name: output.tolist() for name, output in zip(output_names, outputs)}
    
    return {"inputs": input_data, "outputs": outputs_dict}

def load_supported_ops(filename="tests/CodeGen/Python-ONNX/available_operations.txt"):
    """Carica le operazioni supportate da un file oppure restituisce una lista di default."""
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default operations.")
        return [
            "LeakyRelu", "Relu", "Sigmoid", "Softmax", "Add", "Ceil", "Div", "Mul", "Sub", "Tanh",
            "Concat", "Gather", "Identity", "Neg", "Reshape", "Resize", "Shape", "Slice", 
            "Split", "Transpose", "Unsqueeze", "Mean", "Conv", "MatMul", "Gemm", "MaxPool"
        ]


def main():
    print(f"\n __main__")
    parser = argparse.ArgumentParser(description="Generate fuzzed ONNX models and save execution data in JSON.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of models to generate for each operation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random generation (for reproducibility).")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save generated models.")
    parser.add_argument("--metadata-file", type=str, default="results.json",
                        help="File to save metadata and execution data.")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    output_dir = args.output_dir
    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'
    os.makedirs(output_dir, exist_ok=True)
    
    supported_ops = load_supported_ops()
    
    print(f"\n supported_ops : {supported_ops}")
    
    all_models = []
    
    for op in supported_ops:
        for i in range(args.iterations):
            filename = f"{output_dir}{op}_{i}.onnx"
            try:
                metadata = generate_model(op, filename, i)
                data = run_model(filename)
                model_info = {
                    "operation": op,
                    "model_id": i,
                    "inputs": data["inputs"],
                    "outputs": data["outputs"],
                    "metadata": metadata
                }
                all_models.append(model_info)
                print(f"Successfully generated and ran model for {op} (ID: {i})")
            except Exception as e:
                print(f"Error generating or running model for {op} (ID: {i}): {e}")
    
    with open(args.metadata_file, 'w') as f:
        json.dump(all_models, f, indent=2)
    print(f"Execution data saved to {args.metadata_file}")

if __name__ == "__main__":
    main()