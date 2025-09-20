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
import pprint
import traceback


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
    input_names = [f"{op_name}_param_in_{i}" for i in range(10)]
    output_names = [f"{op_name}_param_out_{i}" for i in range(10)]
    metadata = {}

    if op_name in ["Relu", "Sigmoid", "Ceil", "Tanh", "Identity", "Neg", "Shape", "Floor", "Sqrt"]:
        # Operatori a singolo input con forma casuale (rank=4)
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        # Crea dati casuali e li inserisce come initializer
        if op_name == "Sqrt":
            data = np.abs(np.random.randn(*shape)).astype(np.float32)
        else:
            data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info( "useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]], 
                                name=f"{op_name}_node")
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Clip":
        # Generate input tensor with predictable shape and values
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        # Generate min value tensor (optional)
        min_value = round(random.uniform(-2.0, 0.0), 2)
        min_tensor = helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [min_value])
        initializers.append(min_tensor)

        # Generate max value tensor (optional)
        max_value = round(random.uniform(0.0, 2.0), 2)
        max_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [], [max_value])
        initializers.append(max_tensor)

        # Output shape is same as input shape
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        # Create node with all three inputs
        node = helper.make_node(
            op_name,
            inputs=[input_names[0], input_names[1], input_names[2]],
            outputs=[output_names[0]],
            name=f"{op_name}_node_min{min_value}_max{max_value}"
        )

        # Input info placeholder (not used in practice)
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)

        metadata = {
            "input_shapes": [shape],
            "output_shapes": [shape],
            "min_value": min_value,
            "max_value": max_value
        }

        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Gelu":
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        approximate = random.choice(["none", "tanh"])

        data = np.random.randn(*shape).astype(np.float32)

        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(
            op_name, 
            inputs=[input_names[0]], 
            outputs=[output_names[0]], 
            approximate=approximate, 
            name=f"{op_name}node_approx{approximate}",
            )

        metadata = {"input_shapes": [shape], "output_shapes": [shape], "approximate": approximate}
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
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape], "axis": axis}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name in ["Add", "Sub", "Div", "Mul"]:
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
    
    elif op_name == "ReduceMean":
        # Generate input tensor with 4D shape for typical use case
        shape = [random.randint(2, 6) for _ in range(4)]  # Ensure shape has values > 1 for meaningful reduction
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Define input_info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        # Choose random axes to reduce along (optional input)
        # Can be empty (reduce all), single axis, or multiple axes
        axes_options = [
            [],  # reduce all axes
            [random.randint(0, len(shape)-1)],  # reduce single random axis
            sorted(random.sample(range(len(shape)), k=random.randint(1, min(2, len(shape)))))  # reduce multiple axes
        ]
        axes = random.choice(axes_options)
        
        # ReduceMean attributes
        keepdims = random.choice([0, 1])  # whether to keep dimensions
        noop_with_empty_axes = random.choice([0, 1])  # behavior when axes is empty
        
        # Calculate output shape
        if len(axes) == 0 and noop_with_empty_axes == 1:
            # No reduction when axes is empty and noop_with_empty_axes is True
            out_shape = shape.copy()
        elif len(axes) == 0:
            # Reduce all axes
            out_shape = [1] * len(shape) if keepdims else []
        else:
            # Reduce specified axes
            out_shape = []
            for i, dim in enumerate(shape):
                if i in axes:
                    if keepdims:
                        out_shape.append(1)
                    # else: dimension is removed
                else:
                    out_shape.append(dim)
        
        # Create axes tensor as initializer if axes is not empty
        node_inputs = [input_names[0]]
        if len(axes) > 0:
            axes_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(axes)], axes)
            initializers.append(axes_tensor)
            node_inputs.append(input_names[1])
        
        # Create output info
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        
        # Create the ReduceMean node
        node = helper.make_node(
            op_name,
            inputs=node_inputs,
            outputs=[output_names[0]],
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
            name=f"{op_name}_node"
        )
        
        metadata = {
            "input_shapes": [shape],
            "output_shapes": [out_shape],
            "axes": axes,
            "keepdims": keepdims,
            "noop_with_empty_axes": noop_with_empty_axes
        }
        
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
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape, shape2], "output_shapes": [out_shape]}
        return [input_info], output_info, [node], initializers, metadata
   
    elif op_name == "Constant":
        # Constant operator: generates a tensor with constant values
        # Choose random data type
        data_types = [
            (TensorProto.FLOAT, np.float32),
            (TensorProto.INT32, np.int32),
            (TensorProto.INT64, np.int64),
            (TensorProto.BOOL, bool)
        ]
        tensor_proto_type, numpy_type = random.choice(data_types)

        # Choose random shape (can be scalar, 1D, 2D, etc.)
        shape_type = random.choice(["scalar", "1d", "2d", "3d"])

        if shape_type == "scalar":
            shape = []
            if tensor_proto_type == TensorProto.FLOAT:
                value = [random.uniform(-10.0, 10.0)]
            elif tensor_proto_type == TensorProto.INT32:
                value = [random.randint(-100, 100)]
            elif tensor_proto_type == TensorProto.INT64:
                value = [random.randint(-100, 100)]
            else:  # BOOL
                value = [random.choice([True, False])]
        elif shape_type == "1d":
            dim_size = random.randint(2, 8)
            shape = [dim_size]
            if tensor_proto_type == TensorProto.FLOAT:
                value = np.random.uniform(-10.0, 10.0, size=shape).astype(numpy_type).flatten().tolist()
            elif tensor_proto_type == TensorProto.INT32:
                value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
            elif tensor_proto_type == TensorProto.INT64:
                value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
            else:  # BOOL
                value = np.random.choice([True, False], size=shape).astype(numpy_type).flatten().tolist()
        elif shape_type == "2d":
            shape = [random.randint(2, 5), random.randint(2, 5)]
            if tensor_proto_type == TensorProto.FLOAT:
                value = np.random.uniform(-10.0, 10.0, size=shape).astype(numpy_type).flatten().tolist()
            elif tensor_proto_type == TensorProto.INT32:
                value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
            elif tensor_proto_type == TensorProto.INT64:
                value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
            else:  # BOOL
                value = np.random.choice([True, False], size=shape).astype(numpy_type).flatten().tolist()
        else:  # "3d"
            shape = [random.randint(1, 3), random.randint(2, 4), random.randint(2, 4)]
            if tensor_proto_type == TensorProto.FLOAT:
                value = np.random.uniform(-10.0, 10.0, size=shape).astype(numpy_type).flatten().tolist()
            elif tensor_proto_type == TensorProto.INT32:
                value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
            elif tensor_proto_type == TensorProto.INT64:
                value = np.random.randint(-100, 100, size=shape).astype(numpy_type).flatten().tolist()
            else:  # BOOL
                value = np.random.choice([True, False], size=shape).astype(numpy_type).flatten().tolist()

        # Create the constant tensor
        constant_tensor = helper.make_tensor(
            name="constant_value",
            data_type=tensor_proto_type,
            dims=shape,
            vals=value
        )

        # Output info
        output_info = helper.make_tensor_value_info(output_names[0], tensor_proto_type, shape)

        # Input info (dummy, since Constant has no inputs)
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, [1])

        # Metadata
        metadata = {
            "input_shapes": [],  # No inputs for Constant
            "output_shapes": [shape],
            "data_type": tensor_proto_type,
            "shape_type": shape_type,
            "value_sample": value[:5] if isinstance(value, list) and len(value) > 5 else value  # First 5 values for debug
        }

        # Create the Constant node with the value attribute
        node = helper.make_node(
            op_name,
            inputs=[],  # Constant has no inputs
            outputs=[output_names[0]],
            value=constant_tensor,
            name=f"{op_name}_node_{tensor_proto_type}_{len(shape)}d"
        )

        return [input_info], output_info, [node], [], metadata  # Empty initializers since value is in the node

    elif op_name == "OneHot":
        # Operatore OneHot: genera indices, depth e values come initializer
        # Genera un tensore indices con rango casuale (1 o 2)
        rank = random.randint(1, 2)
        indices_shape = random_shape(rank, min_dim=2, max_dim=3)
        max_index = 3  # Limite massimo per gli indici
        indices_data = np.random.randint(0, max_index, size=indices_shape).astype(np.int64)
        indices_tensor = helper.make_tensor(input_names[0], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
        initializers.append(indices_tensor)

        # Genera depth (scalare)
        depth_value = random.randint(3, max_index)  # Valore positivo per depth
        depth_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [], [depth_value])
        initializers.append(depth_tensor)

        # Genera values (tensore 1D di lunghezza 2, tipo float32)
        values_data = np.array([0.0, 1.0], dtype=np.float32)  # [off_value, on_value]
        values_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [2], values_data.tolist())
        initializers.append(values_tensor)

        # Scegli un axis valido
        output_rank = rank + 1
        # axis = random.randint(-output_rank, output_rank - 1)
        axis = random.randint(max(-output_rank, -1000), min(output_rank - 1, 1000))

        # Calcola la forma dell'output
        out_shape = indices_shape.copy()
        normalized_axis = axis if axis >= 0 else axis + output_rank
        out_shape.insert(normalized_axis, depth_value)

        # Crea il nodo OneHot
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(
            op_name,
            inputs=[input_names[0], input_names[1], input_names[2]],
            outputs=[output_names[0]],
            axis=axis,
            name=f"{op_name}_node_axis{axis}"
        )

        # Input info fittizio
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, indices_shape)

        # Metadati
        metadata = {
            "input_shapes": [indices_shape, [], [2]],  # Forme di indices, depth, values
            "output_shapes": [out_shape],
            "axis": axis,
            "depth": depth_value,
            "indices": indices_data.flatten().tolist()[:5],  # Solo i primi 5 per debug
            "values": values_data.tolist()
        }

        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Gather":
        # First input: data; second input: indices
        shape = [5, random.randint(5,10)]
        data = np.random.randn(*shape).astype(np.float32)
        init_data = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_data)
        
        # Pick a random axis
        axis = random.randint(0, len(shape)-1)
        
        # Ensure indices are within bounds of the chosen axis
        max_index = shape[axis] - 1
        indices_shape = [random.randint(1,3)]
        indices_data = np.random.randint(0, max_index + 1, size=indices_shape).astype(np.int64)
        init_indices = helper.make_tensor(input_names[1], TensorProto.INT64, indices_shape, indices_data.flatten().tolist())
        initializers.append(init_indices)
        
        # Calculate output shape
        out_shape = list(shape)
        out_shape[axis] = indices_shape[0]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                              axis=axis, name=f"{op_name}node_axis{axis}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {
            "input_shapes": [shape, indices_shape], 
            "output_shapes": [out_shape],
            "axis": axis,
            "indices": indices_data.tolist()
        }
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Elu":
        # generate 1D tensor 
        shape = [random.randint(1, 10)]  
        alpha = round(random.uniform(0.5, 2.0), 3)  
        data = np.random.randn(*shape).astype(np.float32)

        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(op_name,
                                inputs=[input_names[0]],
                                outputs=[output_names[0]], 
                                alpha=alpha,
                                name=f"{op_name}node_alpha{alpha}"
                                )
        metadata = {"input_shapes": [shape],
                    "output_shapes": [shape],
                    "alpha": alpha
                    }
        
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Flatten":
        # Generate casual rank and shape
        rank = random.randint(1, 4)
        if rank == 0:
            shape = []  
        else:
            shape = random_shape(rank, min_dim=1, max_dim=10)

        # Generate casual data
        total_size = 1 if rank == 0 else np.prod(shape)
        data = np.random.randn(total_size).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        # choosing right axis
        axis = random.randint(-rank, rank) if rank > 0 else 0  # for scalar, axis is 0

        # Calculate output shape
        if rank == 0:
            out_shape = [1, 1]  # Scalare -> [1, 1]
        else:
            outer_dim = 1
            normalized_axis = axis if axis >= 0 else axis + rank
            for i in range(normalized_axis):
                outer_dim *= shape[i]
            inner_dim = 1
            for i in range(normalized_axis, rank):
                inner_dim *= shape[i]
            out_shape = [outer_dim, inner_dim]

        # create input and output info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)

        # Create Flatten node
        node = helper.make_node(
            op_name,
            inputs=[input_names[0]],
            outputs=[output_names[0]],
            axis=axis,
            name=f"{op_name}node_axis{axis}"
        )

        # Metadati
        metadata = {
            "input_shapes": [shape],
            "output_shapes": [out_shape],
            "axis": axis
        }

        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Squeeze":
        # Generate input shape with at least one dimension of size 1
        shape = [1, random.randint(1, 3), 1, random.randint(5, 10)]
        data = np.random.randn(*shape).astype(np.float32)

        # Create initializer for the input tensor
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        # Randomly decide whether to include axes
        use_axes = random.choice([True, False])

        if use_axes:
            # Find which axes in the shape are 1
            squeezable_axes = [i for i, dim in enumerate(shape) if dim == 1]
            if not squeezable_axes:
                # No valid axes to squeeze; force at least one
                squeezable_axes = [0]
            # Pick a random subset of the squeezable axes
            num_axes = random.randint(1, len(squeezable_axes))
            selected_axes = random.sample(squeezable_axes, num_axes)

            # Create an INT64 tensor with selected_axes
            axes_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(selected_axes)], selected_axes)
            initializers.append(axes_tensor)

            # Node takes 2 inputs: data and axes
            node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]],
                                    outputs=[output_names[0]], name=f"{op_name}_node")

            # Compute output shape manually
            out_shape = [dim for i, dim in enumerate(shape) if i not in selected_axes]

        else:
            # No axes specified: remove all dims of size 1
            node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                    name=f"{op_name}_node")
            out_shape = [dim for dim in shape if dim != 1]

        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)

        metadata = {
            "input_shapes": [shape],
            "output_shapes": [out_shape],
            "axes": selected_axes if use_axes else None
        }

        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Pad":
        # Pad operator: pads tensor with specified padding modes
        # Supports constant, reflect, edge, and wrap modes according to ONNX spec
        
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        rank = len(shape)
        
        # Generate pads tensor: [begin_pad_0, begin_pad_1, ..., end_pad_0, end_pad_1, ...]
        pads = [random.randint(0, 2) for _ in range(2 * rank)]
        out_shape = [shape[i] + pads[i] + pads[i + rank] for i in range(rank)]
        pads_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(pads)], pads)
        initializers.append(pads_tensor)
        
        # Choose padding mode
        mode = random.choice(["constant", "reflect", "edge"])
        node_inputs = [input_names[0], input_names[1]]
        constant_value = None
        
        # Add constant value if mode is constant
        if mode == "constant":
            constant_value = round(random.uniform(-1.0, 1.0), 2)
            constant_tensor = helper.make_tensor(input_names[2], TensorProto.FLOAT, [], [constant_value])
            initializers.append(constant_tensor)
            node_inputs.append(input_names[2])
        
        # Note: axes parameter is not commonly used in basic Pad, keeping it simple
        axes = None
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        
        node = helper.make_node(
            "Pad",
            inputs=node_inputs,
            outputs=[output_names[0]],
            mode=mode,
            name=f"{op_name}_node_mode_{mode}"
        )
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        metadata = {
            "input_shapes": [shape],
            "output_shapes": [out_shape],
            "pads": pads,
            "mode": mode,
            "axes": axes,
            "constant_value": constant_value
        }
        
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Reshape":
        
        # First input: data; second input: new shape (initializer)
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        new_shape = shape.copy()
        random.shuffle(new_shape)
        shape_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(new_shape)], new_shape)
        initializers.append(shape_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, new_shape)
        
        # Remove the 'shape' attribute - it's not valid for ONNX Reshape
        node = helper.make_node(op_name, 
                                inputs=[input_names[0], input_names[1]], 
                                outputs=[output_names[0]],
                                name=f"{op_name}_node")
        
        metadata = {"input_shapes": [shape, new_shape], "output_shapes": [new_shape]}
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Resize":

        # Generate a random input tensor shape: (N=1, C=random, H=random, W=random)
        shape = [1, random.randint(1, 4), random.randint(10, 50), random.randint(10, 50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.ravel())
        initializers.append(init_tensor)

        # Empty ROI tensor (optional in ONNX, but included for compatibility)
        roi_name = input_names[1] + "_roi"
        roi_tensor = helper.make_tensor(roi_name, TensorProto.FLOAT, [0], [])
        initializers.append(roi_tensor)

        # Empty Scales tensor (using Sizes instead)
        scales_name = input_names[2] + "_scales"
        scales_tensor = helper.make_tensor(scales_name, TensorProto.FLOAT, [0], [])
        initializers.append(scales_tensor)

        # Compute new sizes (scaled spatial dimensions)
        scale_factor = round(random.uniform(0.5, 2.0), 2)
        new_height = int(shape[2] * scale_factor)
        new_width = int(shape[3] * scale_factor)

        sizes = [shape[0], shape[1], new_height, new_width]
        sizes_name = input_names[3] + "_sizes"
        sizes_tensor = helper.make_tensor(sizes_name, TensorProto.INT64, [len(sizes)], sizes)
        initializers.append(sizes_tensor)

        # Output tensor info
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, sizes)

        # Choose a resize mode (explicit selection)
        mode = random.choice(["nearest", "linear"])

        # Create ONNX Resize node
        node = helper.make_node(
            op_name,
            inputs=[input_names[0], roi_name, scales_name, sizes_name],
            outputs=[output_names[0]],
            mode=mode,
            name=f"{op_name}_mode_{mode}"
        )

        # Input info placeholder (not actually used)
        input_info = helper.make_tensor_value_info("unused_input", TensorProto.FLOAT, shape)

        # Metadata dictionary
        metadata = {
            "input_shapes": [shape],
            "output_shapes": [sizes],
            "mode": mode,
            "scale_factor": scale_factor,
        }

        return [input_info], output_info, [node], initializers, metadata

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
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "starts": starts, "ends": ends}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Split":
        # Create a more realistic neural network test for Split
        shape = [2, 7, 28, 26]  # Fixed shape to match API checks
        axis = 0  # Split along the first dimension
        
        # Create input data
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Calculate split output shapes
        out_shape = shape.copy()
        out_shape[axis] = shape[axis] // 2
        
        # Create intermediate output names
        split_output1 = "split_output1"
        split_output2 = "split_output2"
        processed_output1 = "processed_output1"
        processed_output2 = "processed_output2"
        
        # Create the Split node
        split_node = helper.make_node(
            "Split", 
            inputs=[input_names[0]], 
            outputs=[split_output1, split_output2],
            axis=axis, 
            name=f"{op_name}_split_node",
            num_outputs=2  # Add this line
        )
        
        # Process the first split part with Relu
        relu_node = helper.make_node(
            "Relu",
            inputs=[split_output1],
            outputs=[processed_output1],
            name="Relu_after_split"
        )
        
        # Process the second split part with Sigmoid
        sigmoid_node = helper.make_node(
            "Sigmoid",
            inputs=[split_output2],
            outputs=[processed_output2],
            name="Sigmoid_after_split"
        )
        
        # Combine the processed outputs with Add
        add_node = helper.make_node(
            "Add",
            inputs=[processed_output1, processed_output2],
            outputs=[output_names[0]],
            name="Add_after_processing"
        )
        
        # All nodes needed for this model
        node = [split_node, relu_node, sigmoid_node, add_node]
        
        # Create the input tensor info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        metadata = {
            "input_shapes": [shape], 
            "output_shapes": [out_shape],
            "axis": axis,
            "note": "This model splits the input, applies Relu to first part and Sigmoid to second part, then adds them together"
        }
        
        return [input_info], helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape), node, initializers, metadata

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
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "perm": perm}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Unsqueeze":
        # Insert a dimension at a random axis
        shape = [random.randint(1,4) for _ in range(4)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        rank = len(shape)
        # Axis can be in range [-rank-1, rank] for Unsqueeze
        axis = random.randint(-rank-1, rank)
        axes = [axis]
        axes_tensor = helper.make_tensor(input_names[1], TensorProto.INT64, [len(axes)], axes)
        initializers.append(axes_tensor)
        
        # Convert negative axis to positive for output shape calculation
        actual_axis = axis if axis >= 0 else rank + 1 + axis
        out_shape = shape.copy()
        out_shape.insert(actual_axis, 1)
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, out_shape)
        node = helper.make_node(op_name, 
                            inputs=[input_names[0], input_names[1]], 
                            outputs=[output_names[0]],
                            name=f"{op_name}node_axis{axis}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [out_shape], "axes": axes}
        return [input_info], output_info, [node], initializers, metadata

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
        
        # Add strides parameter
        strides = [random.randint(1, 3), random.randint(1, 3)]
        
        # Add dilations parameter
        dilations = [random.randint(1, 2), random.randint(1, 2)]
        
        # Add padding parameter (padding for begin and end of each spatial axis)
        # Format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        pad_h = random.randint(0, 2)
        pad_w = random.randint(0, 2)
        pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]  # Padding for 4D tensor [N,C,H,W]
        
        # Calculate output dimensions with strides, dilations and padding
        # Formula: output_size = (input_size + pad_begin + pad_end - (kernel_size-1)*dilation - 1) / stride + 1
        H_out = (H + 2*pad_h - (kH-1)*dilations[0] - 1) // strides[0] + 1
        W_out = (W + 2*pad_w - (kW-1)*dilations[1] - 1) // strides[1] + 1
        
        # Ensure valid output dimensions
        if H_out <= 0 or W_out <= 0:
            # Fallback to simpler case if dimensions don't work out
            dilations = [1, 1]
            pad_h = pad_w = 1
            pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
            H_out = (H + 2*pad_h - kH) // strides[0] + 1
            W_out = (W + 2*pad_w - kW) // strides[1] + 1
        
        output_shape = [N, M, H_out, W_out]
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        
        # Use pads attribute in the node (simplified to just the 4 values for H,W dimensions)
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
                                kernel_shape=kernel_shape, strides=strides, dilations=dilations,
                                pads=[pad_h, pad_w, pad_h, pad_w],
                                name=f"{op_name}node_k{kernel_shape}_s{strides}_d{dilations}_p{[pad_h, pad_w]}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        metadata = {"input_shapes": [input_shape, weight_shape], "output_shapes": [output_shape],
                    "kernel_shape": kernel_shape, "strides": strides, "dilations": dilations, 
                    "pads": [pad_h, pad_w, pad_h, pad_w]}
        return [input_info], output_info, [node], initializers, metadata

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
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, A_shape)
        metadata = {"input_shapes": [A_shape, B_shape], "output_shapes": [[M_val, N_val]]}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Gemm":
        # Gemm: esegue A * B + C
        M_val = random.randint(2,10)
        K_val = random.randint(2,10)
        N_val = random.randint(2,10)
        A_shape = [M_val, K_val]
        B_shape = [K_val, N_val]
        C_shape = [M_val, N_val]  # C must be broadcastable to (M,N)
        
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
        
        # Fix: Don't use transA/transB to avoid dimension mismatches
        node = helper.make_node(op_name, inputs=[input_names[0], input_names[1], input_names[2]], outputs=[output_names[0]],
                                alpha=alpha, beta=beta, 
                                name=f"{op_name}node_alpha{alpha}beta{beta}")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, A_shape)
        metadata = {"input_shapes": [A_shape, B_shape, C_shape], "output_shapes": [[M_val, N_val]],
                    "alpha": alpha, "beta": beta}
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "MaxPool":
        # MaxPool operator: applies max pooling with configurable parameters
        # Supports various spatial dimensions and parameter combinations
        
        # Random values for batch and channel dimensions
        N = random.randint(1, 3)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        
        # Support both 2D and 3D pooling
        spatial_type = random.choice(["2D", "3D"])
        
        if spatial_type == "2D":
            H = random.randint(4, 12)  # Height
            W = random.randint(4, 12)  # Width
            input_shape = [N, C, H, W]
            spatial_dims = [H, W]
            
            # Kernel shape for 2D pooling
            kernel_h = random.randint(2, min(4, H))
            kernel_w = random.randint(2, min(4, W))
            kernel_shape = [kernel_h, kernel_w]
            
        else:  # 3D pooling
            D = random.randint(3, 6)   # Depth
            H = random.randint(3, 8)   # Height  
            W = random.randint(3, 8)   # Width
            input_shape = [N, C, D, H, W]
            spatial_dims = [D, H, W]
            
            # Kernel shape for 3D pooling
            kernel_d = random.randint(2, min(3, D))
            kernel_h = random.randint(2, min(3, H))
            kernel_w = random.randint(2, min(3, W))
            kernel_shape = [kernel_d, kernel_h, kernel_w]
        
        # Create input data with predictable patterns
        total_elements = np.prod(input_shape)
        data = np.arange(1, total_elements + 1, dtype=np.float32).reshape(input_shape)
        
        # Add some randomness while keeping it deterministic for testing
        if random.choice([True, False]):
            # Add structured pattern
            data = data + np.random.uniform(-0.5, 0.5, input_shape).astype(np.float32)
        else:
            # Use sequential values for predictable max pooling results
            data = data / total_elements * 100  # Scale to reasonable range
        
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Random stride parameters (must be positive)
        if spatial_type == "2D":
            strides = [random.randint(1, min(3, kernel_h)), random.randint(1, min(3, kernel_w))]
        else:
            strides = [random.randint(1, min(2, kernel_d)), 
                      random.randint(1, min(2, kernel_h)), 
                      random.randint(1, min(2, kernel_w))]
        
        # Randomly choose auto_pad strategy
        auto_pad_options = ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"]
        auto_pad = random.choice(auto_pad_options)
        
        # Calculate padding and output dimensions based on auto_pad
        if auto_pad == "NOTSET":
            # Explicit padding - random but reasonable values
            if spatial_type == "2D":
                max_pad_h = min(2, kernel_h // 2)
                max_pad_w = min(2, kernel_w // 2)
                pads = [
                    random.randint(0, max_pad_h),  # top
                    random.randint(0, max_pad_w),  # left
                    random.randint(0, max_pad_h),  # bottom
                    random.randint(0, max_pad_w)   # right
                ]
            else:  # 3D
                max_pad_d = min(1, kernel_d // 2)
                max_pad_h = min(1, kernel_h // 2)
                max_pad_w = min(1, kernel_w // 2)
                pads = [
                    random.randint(0, max_pad_d), random.randint(0, max_pad_h), random.randint(0, max_pad_w),  # front, top, left
                    random.randint(0, max_pad_d), random.randint(0, max_pad_h), random.randint(0, max_pad_w)   # back, bottom, right
                ]
        else:
            # Auto padding - let ONNX calculate
            pads = [0] * (len(spatial_dims) * 2)
        
        # Optional parameters (need to be before output shape calculation)
        ceil_mode = random.choice([0, 1])  # Random ceil mode
        storage_order = random.choice([0, 1])  # 0 = row major, 1 = column major
        dilations = [1] * len(kernel_shape)  # Currently only support dilations = 1

        # Calculate output dimensions
        if auto_pad == "NOTSET":
            # Manual calculation for explicit padding
            if spatial_type == "2D":
                if ceil_mode:
                    H_out = ((H + pads[0] + pads[2] - kernel_shape[0] + strides[0] - 1) // strides[0]) + 1
                    W_out = ((W + pads[1] + pads[3] - kernel_shape[1] + strides[1] - 1) // strides[1]) + 1
                else:
                    H_out = ((H + pads[0] + pads[2] - kernel_shape[0]) // strides[0]) + 1
                    W_out = ((W + pads[1] + pads[3] - kernel_shape[1]) // strides[1]) + 1
                output_shape = [N, C, H_out, W_out]
            else:  # 3D
                if ceil_mode:
                    D_out = ((D + pads[0] + pads[3] - kernel_shape[0] + strides[0] - 1) // strides[0]) + 1
                    H_out = ((H + pads[1] + pads[4] - kernel_shape[1] + strides[1] - 1) // strides[1]) + 1
                    W_out = ((W + pads[2] + pads[5] - kernel_shape[2] + strides[2] - 1) // strides[2]) + 1
                else:
                    D_out = ((D + pads[0] + pads[3] - kernel_shape[0]) // strides[0]) + 1
                    H_out = ((H + pads[1] + pads[4] - kernel_shape[1]) // strides[1]) + 1
                    W_out = ((W + pads[2] + pads[5] - kernel_shape[2]) // strides[2]) + 1
                output_shape = [N, C, D_out, H_out, W_out]
        elif auto_pad == "VALID":
            # No padding
            if spatial_type == "2D":
                H_out = ((H - kernel_shape[0]) // strides[0]) + 1
                W_out = ((W - kernel_shape[1]) // strides[1]) + 1
                output_shape = [N, C, H_out, W_out]
            else:  # 3D
                D_out = ((D - kernel_shape[0]) // strides[0]) + 1
                H_out = ((H - kernel_shape[1]) // strides[1]) + 1
                W_out = ((W - kernel_shape[2]) // strides[2]) + 1
                output_shape = [N, C, D_out, H_out, W_out]
        else:  # SAME_UPPER or SAME_LOWER
            # Output size preserves input size divided by stride
            if spatial_type == "2D":
                H_out = int(np.ceil(H / strides[0]))
                W_out = int(np.ceil(W / strides[1]))
                output_shape = [N, C, H_out, W_out]
            else:  # 3D
                D_out = int(np.ceil(D / strides[0]))
                H_out = int(np.ceil(H / strides[1]))
                W_out = int(np.ceil(W / strides[2]))
                output_shape = [N, C, D_out, H_out, W_out]
        
        # Validate output dimensions are positive
        if any(dim <= 0 for dim in output_shape[2:]):  # Check spatial dimensions
            # Fallback to simpler, guaranteed-working configuration
            if spatial_type == "2D":
                kernel_shape = [2, 2]
                strides = [2, 2]
                auto_pad = "VALID"
                pads = [0, 0, 0, 0]
                H_out = ((H - 2) // 2) + 1
                W_out = ((W - 2) // 2) + 1
                output_shape = [N, C, max(1, H_out), max(1, W_out)]
            else:  # 3D
                kernel_shape = [2, 2, 2]
                strides = [2, 2, 2]
                auto_pad = "VALID"
                pads = [0, 0, 0, 0, 0, 0]
                D_out = ((D - 2) // 2) + 1
                H_out = ((H - 2) // 2) + 1
                W_out = ((W - 2) // 2) + 1
                output_shape = [N, C, max(1, D_out), max(1, H_out), max(1, W_out)]
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        
        # dilations parameter
        dilations = [1] * len(kernel_shape)  # Currently only support dilations = 1
        
        # Create the MaxPool node with all parameters
        node_attrs = {
            "kernel_shape": [int(k) for k in kernel_shape],
            "strides": [int(s) for s in strides],
            "pads": [int(p) for p in pads],
            "auto_pad": auto_pad,
            "ceil_mode": int(ceil_mode),
            "storage_order": int(storage_order)
        }
        
        # Only add dilations if not all 1s (some ONNX versions don't like explicit [1,1])
        if any(d != 1 for d in dilations):
            node_attrs["dilations"] = [int(d) for d in dilations]
        
        node = helper.make_node(
            op_name,
            inputs=[input_names[0]],
            outputs=[output_names[0]],
            name=f"{op_name}_node_{spatial_type}_N{N}_C{C}_k{kernel_shape}_s{strides}",
            **node_attrs
        )
        
        metadata = {
            "input_shapes": [input_shape],
            "output_shapes": [output_shape],
            "spatial_type": spatial_type,
            "batch_size": int(N),
            "channels": int(C),
            "spatial_dimensions": [int(d) for d in spatial_dims],
            "kernel_shape": [int(k) for k in kernel_shape],
            "strides": [int(s) for s in strides],
            "pads": [int(p) for p in pads],
            "auto_pad": auto_pad,
            "ceil_mode": int(ceil_mode),
            "storage_order": int(storage_order),
            "dilations": [int(d) for d in dilations]
        }
        
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "AveragePool":
        # AveragePool operator following ONNX v22 specification exactly
        # https://onnx.ai/onnx/operators/onnx__AveragePool.html
        
        # Random values for batch and channel dimensions
        N = random.randint(1, 2)  # Keep small for testing
        C = random.randint(1, 3)  # Few channels
        
        # Support 2D pooling (most common case)
        spatial_type = "2D"  # Focus on 2D for now
        
        H = random.randint(4, 8)  # Height
        W = random.randint(4, 8)  # Width
        input_shape = [N, C, H, W]
        
        # Create input data with predictable sequential values
        total_elements = np.prod(input_shape)
        data = np.arange(1, total_elements + 1, dtype=np.float32).reshape(input_shape)
        
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Conservative kernel and stride parameters
        kernel_h = random.randint(2, min(3, H))
        kernel_w = random.randint(2, min(3, W))
        kernel_shape = [kernel_h, kernel_w]
        
        stride_h = random.randint(1, min(2, kernel_h))
        stride_w = random.randint(1, min(2, kernel_w))
        strides = [stride_h, stride_w]
        
        # Dilations: INTS - defaults to 1 along each spatial axis if not present
        # Must match spatial dimensions (2D = 2 values)
        use_dilation = random.choice([True, False])
        if use_dilation:
            # Use small dilations for stability
            dilations = [random.choice([1, 2]), random.choice([1, 2])]
        else:
            dilations = [1, 1]  # Default
        
        # Random auto_pad mode
        auto_pad_options = ["NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"]
        auto_pad = random.choice(auto_pad_options)
        
        # Random ceil_mode and count_include_pad
        ceil_mode = random.choice([0, 1])
        count_include_pad = random.choice([0, 1])
        
        # Calculate padding and output dimensions using exact ONNX v22 formulas
        if auto_pad == "NOTSET":
            # Explicit padding - use small values to avoid issues
            # ONNX pads format: [x1_begin, x2_begin, x1_end, x2_end] for 2D
            max_pad = 1
            pad_top = random.randint(0, max_pad)
            pad_left = random.randint(0, max_pad)
            pad_bottom = random.randint(0, max_pad)
            pad_right = random.randint(0, max_pad)
            pads = [pad_top, pad_left, pad_bottom, pad_right]
            
            # ONNX v22 explicit padding formula:
            # output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides[i] + 1)
            # or ceil(...) if ceil_mode is enabled
            
            # For height (i=0)
            pad_shape_h = pad_top + pad_bottom
            effective_kernel_h = dilations[0] * (kernel_shape[0] - 1) + 1
            if ceil_mode:
                H_out = int(np.ceil((H + pad_shape_h - effective_kernel_h) / strides[0])) + 1
            else:
                H_out = int(np.floor((H + pad_shape_h - effective_kernel_h) / strides[0])) + 1
            
            # For width (i=1)
            pad_shape_w = pad_left + pad_right
            effective_kernel_w = dilations[1] * (kernel_shape[1] - 1) + 1
            if ceil_mode:
                W_out = int(np.ceil((W + pad_shape_w - effective_kernel_w) / strides[1])) + 1
            else:
                W_out = int(np.floor((W + pad_shape_w - effective_kernel_w) / strides[1])) + 1
                
        elif auto_pad == "VALID":
            pads = [0, 0, 0, 0]
            
            # ONNX v22 VALID padding formulas:
            if ceil_mode:
                # VALID with ceil_mode: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides[i])
                effective_kernel_h = (kernel_shape[0] - 1) * dilations[0] + 1
                effective_kernel_w = (kernel_shape[1] - 1) * dilations[1] + 1
                H_out = int(np.ceil((H - effective_kernel_h + 1) / strides[0]))
                W_out = int(np.ceil((W - effective_kernel_w + 1) / strides[1]))
            else:
                # VALID without ceil_mode: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides[i]) + 1
                effective_kernel_h = (kernel_shape[0] - 1) * dilations[0] + 1
                effective_kernel_w = (kernel_shape[1] - 1) * dilations[1] + 1
                H_out = int(np.floor((H - effective_kernel_h) / strides[0])) + 1
                W_out = int(np.floor((W - effective_kernel_w) / strides[1])) + 1
                
        else:  # SAME_UPPER or SAME_LOWER
            # ONNX v22 SAME padding formulas:
            if ceil_mode:
                # SAME with ceil_mode: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
                H_out = int(np.ceil(H / strides[0]))
                W_out = int(np.ceil(W / strides[1]))
            else:
                # SAME without ceil_mode: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides[i]) + 1
                H_out = int(np.floor((H - 1) / strides[0])) + 1
                W_out = int(np.floor((W - 1) / strides[1])) + 1
            
            # Calculate padding using ONNX v22 formula:
            # pad_shape[i] = (output_spatial_shape[i] - 1) * strides[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
            effective_kernel_h = (kernel_shape[0] - 1) * dilations[0] + 1
            effective_kernel_w = (kernel_shape[1] - 1) * dilations[1] + 1
            
            pad_shape_h = max(0, (H_out - 1) * strides[0] + effective_kernel_h - H)
            pad_shape_w = max(0, (W_out - 1) * strides[1] + effective_kernel_w - W)
            
            if auto_pad == "SAME_UPPER":
                # Extra padding at the end for SAME_UPPER
                pad_top = pad_shape_h // 2
                pad_bottom = pad_shape_h - pad_top
                pad_left = pad_shape_w // 2
                pad_right = pad_shape_w - pad_left
            else:  # SAME_LOWER
                # Extra padding at the beginning for SAME_LOWER
                pad_bottom = pad_shape_h // 2
                pad_top = pad_shape_h - pad_bottom
                pad_right = pad_shape_w // 2
                pad_left = pad_shape_w - pad_right
            
            pads = [pad_top, pad_left, pad_bottom, pad_right]
        
        # Ensure output dimensions are positive
        H_out = max(1, H_out)
        W_out = max(1, W_out)
        
        # Validate that output makes sense
        if H_out <= 0 or W_out <= 0 or H_out > H * 2 or W_out > W * 2:
            # Fallback to guaranteed working configuration
            kernel_shape = [2, 2]
            strides = [1, 1]
            dilations = [1, 1]
            auto_pad = "VALID"
            pads = [0, 0, 0, 0]
            ceil_mode = 0
            count_include_pad = 0
            
            # Recalculate with safe parameters
            H_out = ((H - 2) // 1) + 1
            W_out = ((W - 2) // 1) + 1
        
        output_shape = [N, C, H_out, W_out]
        
        # Validate final output shape
        assert all(dim > 0 for dim in output_shape), f"Invalid output shape: {output_shape}"
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        
        # Create AveragePool node with ONNX v22 attributes
        node_attrs = {
            "kernel_shape": [int(k) for k in kernel_shape],
            "auto_pad": auto_pad,
            "ceil_mode": int(ceil_mode),
            "count_include_pad": int(count_include_pad)
        }
        
        # Add optional attributes only if they differ from defaults
        # strides: defaults to 1 along each spatial axis if not present
        if any(s != 1 for s in strides):
            node_attrs["strides"] = [int(s) for s in strides]
        
        # dilations: defaults to 1 along each spatial axis if not present  
        if any(d != 1 for d in dilations):
            node_attrs["dilations"] = [int(d) for d in dilations]
        
        # pads: defaults to 0 along start and end of each spatial axis if not present
        # Cannot be used simultaneously with auto_pad attribute (except NOTSET)
        if auto_pad == "NOTSET" and any(p != 0 for p in pads):
            node_attrs["pads"] = [int(p) for p in pads]
        
        node = helper.make_node(
            op_name,
            inputs=[input_names[0]],
            outputs=[output_names[0]],
            name=f"{op_name}_node_N{N}_C{C}_H{H}_W{W}_k{kernel_shape}_s{strides}_d{dilations}",
            **node_attrs
        )
        
        metadata = {
            "input_shapes": [input_shape],
            "output_shapes": [output_shape],
            "spatial_type": spatial_type,
            "batch_size": int(N),
            "channels": int(C),
            "spatial_dimensions": [int(H), int(W)],
            "kernel_shape": [int(k) for k in kernel_shape],
            "strides": [int(s) for s in strides],
            "dilations": [int(d) for d in dilations],
            "pads": [int(p) for p in pads],
            "auto_pad": auto_pad,
            "ceil_mode": int(ceil_mode),
            "count_include_pad": int(count_include_pad),
            "onnx_version": "v22"
        }
        
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "GlobalAveragePool":
        # GlobalAveragePool operator: applies average pooling across all spatial dimensions
        # Input shape: (N, C, H, W, ...) where N=batch, C=channels, H,W,...=spatial dims
        # Output shape: (N, C, 1, 1, ...) where all spatial dimensions become 1
        
        N = random.randint(1, 2)  # Batch size (smaller for compatibility)
        C = random.randint(1, 3)  # Number of channels (smaller for compatibility)
        
        # Most ONNX Runtime implementations work well with 2D spatial dimensions (H, W)
        # Let's focus on 2D case which is most common and well-supported
        spatial_config = random.choice([
            "2D_small",   # Small 2D: typical for testing
            "2D_medium",  # Medium 2D: more realistic
            "3D_small"    # Small 3D: for advanced testing
        ])
        
        if spatial_config == "2D_small":
            spatial_dims = [random.randint(2, 4), random.randint(2, 4)]  # H, W
        elif spatial_config == "2D_medium":
            spatial_dims = [random.randint(3, 6), random.randint(3, 6)]  # H, W
        else:  # 3D_small
            spatial_dims = [random.randint(2, 3), random.randint(2, 3), random.randint(2, 3)]  # H, W, D
        
        # Complete input shape: [N, C, spatial_dims...]
        input_shape = [N, C] + spatial_dims
        
        # Create input data with predictable values for testing
        # Use smaller range to avoid numerical issues
        total_elements = np.prod(input_shape)
        if total_elements > 1000:  # Avoid very large tensors
            # Fallback to smaller dimensions
            spatial_dims = [3, 3]  # Simple 3x3 spatial
            input_shape = [N, C] + spatial_dims
            total_elements = np.prod(input_shape)
        
        # Generate data in a reasonable range
        data = np.random.uniform(0.1, 10.0, input_shape).astype(np.float32)
        
        # Alternative: use sequential data for predictable testing
        if random.choice([True, False]):
            data = np.arange(1, total_elements + 1, dtype=np.float32).reshape(input_shape)
            data = data / total_elements  # Normalize to avoid large numbers
        
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, input_shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Output shape: same as input but all spatial dimensions become 1
        output_shape = [N, C] + [1] * len(spatial_dims)
        
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        
        # GlobalAveragePool has no attributes according to ONNX specification
        node = helper.make_node(
            op_name, 
            inputs=[input_names[0]], 
            outputs=[output_names[0]],
            name=f"{op_name}_node_N{N}_C{C}_config{spatial_config}"
        )
        
        metadata = {
            "input_shapes": [input_shape],
            "output_shapes": [output_shape],
            "batch_size": int(N),  # Convert to Python int
            "channels": int(C),    # Convert to Python int
            "spatial_dimensions": [int(dim) for dim in spatial_dims],  # Convert list elements
            "spatial_config": spatial_config,
            "total_elements": int(total_elements)  # Convert to Python int
        }
        
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Mean":
        num_inputs = random.randint(1, 5)
        max_dims = 3 
        
        # 1. Generate a potential "output" shape first
        output_shape = [random.randint(1, 4) for _ in range(max_dims)]

        shapes = []
        initializers = [] # Ensure initializers is defined here
        
        for i in range(num_inputs):
            # 2. Derive compatible input shape from the output shape
            current_shape = []
            for dim_size in output_shape:
                # Each dimension is either the same as output_shape or 1
                current_shape.append(random.choice([1, dim_size]))
            shapes.append(current_shape)

            # data generation for each input tensor
            data = np.random.randn(*current_shape).astype(np.float32)
            tensor_name = input_names[i]
            init_tensor = helper.make_tensor(tensor_name, TensorProto.FLOAT, current_shape, data.flatten().tolist())
            initializers.append(init_tensor) # Now append to the locally defined list
        
        # The actual output shape is already determined by output_shape list
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
        node = helper.make_node(
            op_name,
            inputs=[input_names[i] for i in range(num_inputs)],
            outputs=[output_names[0]],
            name=f"{op_name}_node_{num_inputs}_inputs"
        )
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shapes[0])
        metadata = {
            "input_shapes": shapes,
            "output_shapes": [output_shape]
        }
        
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "Clip":
        # Clip operator: clips values between min and max
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32) * 10 # Scale data to have values outside typical clip range
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)

        node_inputs = [input_names[0]]
        min_val = None
        max_val = None
        clip_metadata = {}

        # Randomly include min value
        if random.choice([True, False]):
            min_val = round(random.uniform(-5.0, 0.0), 2)
            min_tensor = helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [min_val])
            initializers.append(min_tensor)
            node_inputs.append(input_names[1])
            clip_metadata["min_value"] = min_val
            
        # Randomly include max value
        if random.choice([True, False]):
            # Ensure max_val is greater than min_val if min_val exists
            lower_bound_for_max = min_val if min_val is not None else 0.1
            max_val = round(random.uniform(lower_bound_for_max, 5.0), 2) 
            max_tensor_input_index = len(node_inputs) # Determine correct input index for max
            max_tensor = helper.make_tensor(input_names[max_tensor_input_index], TensorProto.FLOAT, [], [max_val])
            initializers.append(max_tensor)
            node_inputs.append(input_names[max_tensor_input_index])
            clip_metadata["max_value"] = max_val
            
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=node_inputs, outputs=[output_names[0]], 
                                name=f"{op_name}_node")

        # Define input_info before using it
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape], **clip_metadata} # Merge clip specific metadata
        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "BatchNormalization":

        # BatchNorm has 5 inputs: X, scale, B, mean, var
        # Output shape is the same as input
        shape = [1, random.randint(1, 4), random.randint(10, 50), random.randint(10, 50)]
        C = shape[1]  # Number of channels (second dimension)

        # Create input tensor
        data_X = np.random.randn(*shape).astype(np.float32)
        init_X = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data_X.flatten().tolist())

        # scale (gamma), bias (beta), mean, var — all of shape [C]
        scale = np.random.randn(C).astype(np.float32)
        B = np.random.randn(C).astype(np.float32)
        mean = np.random.randn(C).astype(np.float32)
        var = np.abs(np.random.randn(C)).astype(np.float32)  # ensure variance is non-negative

        init_scale = helper.make_tensor(input_names[1], TensorProto.FLOAT, [C], scale.tolist())
        init_B = helper.make_tensor(input_names[2], TensorProto.FLOAT, [C], B.tolist())
        init_mean = helper.make_tensor(input_names[3], TensorProto.FLOAT, [C], mean.tolist())
        init_var = helper.make_tensor(input_names[4], TensorProto.FLOAT, [C], var.tolist())

        initializers.extend([init_X, init_scale, init_B, init_mean, init_var])

        epsilon = round(random.uniform(1e-5, 1e-2), 6)
        momentum = round(random.uniform(0.8, 0.99), 3)

        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(
            "BatchNormalization",
            inputs=input_names[:5],
            outputs=[output_names[0]],
            epsilon=epsilon,
            momentum=momentum,
            name=f"{op_name}_node"
        )

        metadata = {
            "input_shapes": [shape, [C], [C], [C], [C]],
            "output_shapes": [shape],
            "epsilon": epsilon,
            "momentum": momentum
        }

        return [input_info], output_info, [node], initializers, metadata
    
    elif op_name == "QuantizeLinear":
        # Randomly pick input shape
        N = 1
        C = random.randint(1,4)
        H = random.randint(10,50)
        W = random.randint(10,50)
        shape = [N, C, H, W]
        data = np.random.randn(*shape).astype(np.float32) # Scale data to have values outside typical clip range
        
        # Randomly choose quantization mode
        mode_choice = random.choice(["per_tensor", 
                                     "per_axis", 
                                    #  "per_block"
                                     ])

        if mode_choice == "per_tensor":
            # Per-tensor quantization
            mode = "per_tensor"
            y_scale = np.array([round(random.uniform(0.01, 1.0), 4)], dtype=np.float32)
            axis = 0
            bl_size = 0
            
        elif mode_choice == "per_axis":
            # Per-axis quantization
            mode = "per_axis"
            axis = random.randint(0, len(shape) - 1)  # Pick a random axis
            length = shape[axis]
            y_scale = np.random.rand(length).astype(np.float32) * 0.5 + 0.1
            bl_size = 0
            
        else:  # per_block
            # Per-block quantization
            mode = "per_block"
            axis = random.randint(0, len(shape) - 1)  # Pick a random axis
            axis_size = shape[axis]
            
            # Block size should be a divisor of the axis size or smaller
            # For simplicity, we'll use powers of 2 that are <= axis_size
            possible_block_sizes = [2**i for i in range(1, int(np.log2(axis_size)) + 1) if 2**i <= axis_size]
            if not possible_block_sizes:
                possible_block_sizes = [1]  # Fallback to block size 1
            
            bl_size = random.choice(possible_block_sizes)
            
            # Calculate the number of blocks
            num_blocks = (axis_size + bl_size - 1) // bl_size  # Ceiling division
            
            # Scale shape: broadcast-compatible with input, with axis dimension = num_blocks
            scale_shape = [1] * len(shape)
            scale_shape[axis] = num_blocks
            y_scale = np.random.rand(*scale_shape).astype(np.float32) * 0.5 + 0.1
        
        # Pick a valid ONNX-compatible dtype for y_zero_point
        valid_dtypes = [
            TensorProto.UINT8,
            # TensorProto.INT8,
            # TensorProto.UINT16,
            # TensorProto.INT16,
        ]
        dtype = random.choice(valid_dtypes)
        
        # Get zero point shape (same as scale shape)
        zp_shape = y_scale.shape
        
        # Map ONNX dtype to numpy dtype
        dtype_np = {
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
            TensorProto.UINT16: np.uint16,
            TensorProto.INT16: np.int16,
        }[dtype]
        
        # Generate y_zero_point with matching shape and dtype
        info = np.iinfo(dtype_np)
        scale_zp = np.random.randint(info.min, info.max + 1, size=zp_shape, dtype=dtype_np)
        
        # Assign names
        scale_name = input_names[1]
        zp_name = input_names[2]
        
        # Add scale and zero point initializers
        initializers.append(helper.make_tensor(
            scale_name, TensorProto.FLOAT, list(zp_shape), y_scale.flatten().tolist()))
        initializers.append(helper.make_tensor(
            zp_name, dtype, list(zp_shape), scale_zp.flatten().tolist()))
        
        # Prepare main input tensor (x)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
         
        # Create output metadata
        output_info = helper.make_tensor_value_info(output_names[0], dtype, shape)
        
        # Create the QuantizeLinear node
        node_kwargs = {
            "inputs": [input_names[0], scale_name, zp_name],
            "outputs": [output_names[0]],
            "name": f"QuantizeLinear_node_dtype{dtype}_mode{mode}_axis{axis if axis is not None else 'None'}"
        }
        
        # Add attributes based on mode
        node_kwargs["axis"] = axis
        node_kwargs["block_size"] = bl_size
        
        # node = helper.make_node("QuantizeLinear", **node_kwargs)


        # node = helper.make_node(op_name, inputs=[input_names[0], input_names[1]], outputs=[output_names[0]],
        #                         kernel_shape=kernel_shape, strides=strides, dilations=dilations,
        #                         pads=[pad_h, pad_w, pad_h, pad_w],
        #                         name=f"{op_name}node_k{kernel_shape}_s{strides}_d{dilations}_p{[pad_h, pad_w]}")

        node = helper.make_node(op_name, inputs=[input_names[0], scale_name, zp_name], outputs=[output_names[0]],
                                axis=axis, 
                                # block_size = bl_size,  // dky but it is not supported, also if the standard onnx doc allows it
                                name=f"{op_name}node_ax{axis}_bl{bl_size}")
        
        # Dummy input_info, useful for the rest of the pipeline
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        # Metadata for codegen
        metadata = {
            "input_shapes": [shape],
            "scale_shape": list(zp_shape),
            "dtype": dtype,
            "axis": axis,
            "block_size": bl_size,
            "mode": mode
        }
        
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "DequantizeLinear":
        mode = "per_tensor"
        # Randomly pick input shape
        shape = [ random.randint(1, 4) for _ in range(3)]  # 3D tensor

        # Select quantized input type
        dtype = random.choice([
            TensorProto.UINT8, TensorProto.INT8,
            # TensorProto.UINT16, TensorProto.INT16,
            # TensorProto.UINT4, TensorProto.INT4,  # Uncomment if supported
        ])

        # Determine scale shape (per-tensor or per-axis)
        if random.choice([True, False]):
            # Per-tensor
            y_scale = np.array([round(random.uniform(0.01, 1.0), 4)], dtype=np.float32)
            zp_shape = y_scale.shape
            axis = None
        else:
            # Per-axis
            axis = 1  # Fixed axis for example
            length = shape[axis]
            y_scale = (np.random.rand(length) * 0.5 + 0.1).astype(np.float32)
            zp_shape = y_scale.shape

        # Generate quantized input `x` and zero point
        if dtype in (TensorProto.UINT4, TensorProto.INT4):  # Optional 4-bit support
            max_val = 2**4 - 1 if dtype == TensorProto.UINT4 else 2**3 - 1
            min_val = 0 if dtype == TensorProto.UINT4 else -2**3
            x_data = np.random.randint(min_val, max_val + 1, size=shape, dtype=np.int32)
            y_zero_point = np.random.randint(min_val, max_val + 1, size=zp_shape, dtype=np.int32)
        else:
            info = np.iinfo({
                TensorProto.UINT8: np.uint8, TensorProto.INT8: np.int8,
                TensorProto.UINT16: np.uint16, TensorProto.INT16: np.int16
            }[dtype])
            x_data = np.random.randint(info.min, info.max + 1, size=shape, dtype=info.dtype)
            y_zero_point = np.random.randint(info.min, info.max + 1, size=zp_shape, dtype=info.dtype)

        # Names
        scale_name = input_names[1]
        zp_name = input_names[2]

        # Initializers
        initializers.append(helper.make_tensor(scale_name, TensorProto.FLOAT, list(zp_shape), y_scale.flatten().tolist()))
        initializers.append(helper.make_tensor(zp_name, dtype, list(zp_shape), y_zero_point.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[0], dtype, shape, x_data.flatten().tolist()))

        # Create output metadata (always float32)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)

        node = helper.make_node(
            "DequantizeLinear",
            inputs=[input_names[0], scale_name, zp_name],
            outputs=[output_names[0]],
            axis=axis if axis is not None else None,
            name=f"DequantizeLinear_node_dtype{dtype}_axis{axis}"
        )

        # Dummy input info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)

        metadata = {
            "input_shapes": [shape],
            "scale_shape": list(zp_shape),
            "dtype": dtype,
            "axis": axis
        }

        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "QLinearConv":
        # QLinearConv requires 8 inputs: x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point
        # Plus optional bias as 9th input
        
        # Input tensor dimensions [N, C, H, W]
        batch_size = 1
        in_channels = random.randint(1, 4)
        input_height = random.randint(10, 20)
        input_width = random.randint(10, 20)
        
        # Filter dimensions [out_channels, in_channels/group, kernel_h, kernel_w]
        out_channels = random.randint(1, 4)
        kernel_size = random.randint(3, 5)
        group = 1  # Keep simple for now
        
        input_shape = [batch_size, in_channels, input_height, input_width]
        weight_shape = [out_channels, in_channels // group, kernel_size, kernel_size]
        bias_shape = [out_channels]
        
        # Calculate output dimensions
        pad = kernel_size // 2  # Same padding
        output_height = input_height  # Same padding
        output_width = input_width
        output_shape = [batch_size, out_channels, output_height, output_width]
        
        # Generate quantized input data (uint8)
        x_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        x_scale = np.random.uniform(0.001, 0.1)
        x_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Generate quantized weight data (uint8)
        w_data = np.random.randint(0, 256, size=weight_shape, dtype=np.uint8)
        w_scale = np.random.uniform(0.001, 0.1)
        w_zero_point = np.uint8(128)  # Zero point for uint8 weights
        
        # Generate output quantization parameters
        y_scale = np.random.uniform(0.001, 0.1)
        y_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Generate bias (int32)
        bias_data = np.random.randint(-1000, 1000, size=bias_shape, dtype=np.int32)
        
        # Create initializers
        initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, input_shape, x_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [x_scale]))
        initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [x_zero_point]))
        initializers.append(helper.make_tensor(input_names[3], TensorProto.UINT8, weight_shape, w_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[4], TensorProto.FLOAT, [], [w_scale]))
        initializers.append(helper.make_tensor(input_names[5], TensorProto.UINT8, [], [w_zero_point]))
        initializers.append(helper.make_tensor(input_names[6], TensorProto.FLOAT, [], [y_scale]))
        initializers.append(helper.make_tensor(input_names[7], TensorProto.UINT8, [], [y_zero_point]))
        initializers.append(helper.make_tensor(input_names[8], TensorProto.INT32, bias_shape, bias_data.flatten().tolist()))
        
        # Create output info (quantized uint8)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, output_shape)
        
        # Create QLinearConv node with all required inputs including bias
        node = helper.make_node(
            "QLinearConv",
            inputs=[input_names[0], input_names[1], input_names[2],  # x, x_scale, x_zero_point
                   input_names[3], input_names[4], input_names[5],   # w, w_scale, w_zero_point
                   input_names[6], input_names[7],                   # y_scale, y_zero_point
                   input_names[8]],                                  # bias
            outputs=[output_names[0]],
            name=f"{op_name}_generic_node",
            dilations=[1, 1],
            group=group,
            kernel_shape=[kernel_size, kernel_size],
            pads=[pad, pad, pad, pad],
            strides=[1, 1]
        )
        
        # Dummy input info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        
        metadata = {
            "input_shapes": [input_shape, weight_shape, bias_shape],
            "output_shapes": [output_shape],
            "x_scale": x_scale,
            "w_scale": w_scale,
            "y_scale": y_scale,
            "kernel_size": kernel_size,
            "group": group
        }
        
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "QLinearGlobalAveragePool":
        # QLinearGlobalAveragePool implemented as: DequantizeLinear -> GlobalAveragePool -> QuantizeLinear
        
        # Input tensor dimensions [N, C, H, W]
        batch_size = 1
        channels = random.randint(1, 8)
        input_height = random.randint(8, 16)
        input_width = random.randint(8, 16)
        
        input_shape = [batch_size, channels, input_height, input_width]
        output_shape = [batch_size, channels, 1, 1]  # Global pooling reduces spatial dims to 1x1
        
        # Generate quantized input data (uint8)
        x_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        x_scale = np.random.uniform(0.001, 0.1)
        x_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Generate output quantization parameters
        y_scale = np.random.uniform(0.001, 0.1)
        y_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Create initializers for all quantization parameters
        initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, input_shape, x_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [x_scale]))
        initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [x_zero_point]))
        initializers.append(helper.make_tensor(input_names[3], TensorProto.FLOAT, [], [y_scale]))
        initializers.append(helper.make_tensor(input_names[4], TensorProto.UINT8, [], [y_zero_point]))
        
        # Create intermediate tensor names
        dequant_output = f"dequant_{output_names[0]}"
        gap_output = f"gap_{output_names[0]}"
        
        # Create three nodes: DequantizeLinear -> GlobalAveragePool -> QuantizeLinear
        nodes = []
        
        # 1. DequantizeLinear node
        dequant_node = helper.make_node(
            "DequantizeLinear",
            inputs=[input_names[0], input_names[1], input_names[2]],  # x, x_scale, x_zero_point
            outputs=[dequant_output],
            name=f"Dequant_{op_name}_node"
        )
        nodes.append(dequant_node)
        
        # 2. GlobalAveragePool node
        gap_node = helper.make_node(
            "GlobalAveragePool",
            inputs=[dequant_output],
            outputs=[gap_output],
            name=f"GAP_{op_name}_node"
        )
        nodes.append(gap_node)
        
        # 3. QuantizeLinear node
        quant_node = helper.make_node(
            "QuantizeLinear",
            inputs=[gap_output, input_names[3], input_names[4]],  # gap_output, y_scale, y_zero_point
            outputs=[output_names[0]],
            name=f"Quant_{op_name}_node"
        )
        nodes.append(quant_node)
        
        # Create output info (quantized uint8)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, output_shape)
        
        # Dummy input info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        
        metadata = {
            "input_shapes": [input_shape],
            "output_shapes": [output_shape],
            "x_scale": x_scale,
            "y_scale": y_scale,
            "channels": channels,
            "spatial_size": input_height * input_width,
            "implementation": "dequant_gap_quant"
        }
        
        return [input_info], output_info, nodes, initializers, metadata

    elif op_name == "QLinearAdd":
        # QLinearAdd implemented as composite: DequantizeLinear -> Add -> QuantizeLinear
        
        # Input tensor dimensions [N, C, H, W] (same for both inputs)
        batch_size = 1
        channels = random.randint(1, 8)
        height = random.randint(8, 16)
        width = random.randint(8, 16)
        
        input_shape = [batch_size, channels, height, width]
        output_shape = input_shape  # QLinearAdd preserves shape
        
        # Generate quantized input data A (uint8)
        a_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        a_scale = np.random.uniform(0.001, 0.1)
        a_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Generate quantized input data B (uint8)
        b_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        b_scale = np.random.uniform(0.001, 0.1)
        b_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Generate output quantization parameters
        c_scale = np.random.uniform(0.001, 0.1)
        c_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Create intermediate tensor names
        dequant_a_name = f"dequant_a_{random.randint(1000, 9999)}"
        dequant_b_name = f"dequant_b_{random.randint(1000, 9999)}"
        add_result_name = f"add_result_{random.randint(1000, 9999)}"
        
        # Create initializers for quantized inputs and quantization parameters
        initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, input_shape, a_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [a_scale]))
        initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [a_zero_point]))
        initializers.append(helper.make_tensor(input_names[3], TensorProto.UINT8, input_shape, b_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[4], TensorProto.FLOAT, [], [b_scale]))
        initializers.append(helper.make_tensor(input_names[5], TensorProto.UINT8, [], [b_zero_point]))
        initializers.append(helper.make_tensor(input_names[6], TensorProto.FLOAT, [], [c_scale]))
        initializers.append(helper.make_tensor(input_names[7], TensorProto.UINT8, [], [c_zero_point]))
        
        # Create three nodes: DequantizeLinear A, DequantizeLinear B, Add, QuantizeLinear
        nodes = []
        
        # 1. DequantizeLinear for input A
        dequant_a_node = helper.make_node(
            "DequantizeLinear",
            inputs=[input_names[0], input_names[1], input_names[2]],  # A, A_scale, A_zero_point
            outputs=[dequant_a_name],
            name=f"DequantizeLinear_A_{random.randint(1000, 9999)}"
        )
        nodes.append(dequant_a_node)
        
        # 2. DequantizeLinear for input B
        dequant_b_node = helper.make_node(
            "DequantizeLinear",
            inputs=[input_names[3], input_names[4], input_names[5]],  # B, B_scale, B_zero_point
            outputs=[dequant_b_name],
            name=f"DequantizeLinear_B_{random.randint(1000, 9999)}"
        )
        nodes.append(dequant_b_node)
        
        # 3. Add the dequantized values
        add_node = helper.make_node(
            "Add",
            inputs=[dequant_a_name, dequant_b_name],
            outputs=[add_result_name],
            name=f"Add_{random.randint(1000, 9999)}"
        )
        nodes.append(add_node)
        
        # 4. QuantizeLinear to output
        quant_node = helper.make_node(
            "QuantizeLinear",
            inputs=[add_result_name, input_names[6], input_names[7]],  # add_result, C_scale, C_zero_point
            outputs=[output_names[0]],
            name=f"QuantizeLinear_{random.randint(1000, 9999)}"
        )
        nodes.append(quant_node)
        
        # Create output info (quantized uint8)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, output_shape)
        
        # Dummy input info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        
        metadata = {
            "input_shapes": [input_shape, input_shape],
            "output_shapes": [output_shape],
            "a_scale": a_scale,
            "b_scale": b_scale,
            "c_scale": c_scale,
            "operation": "quantized_addition_composite"
        }
        
        return [input_info], output_info, nodes, initializers, metadata

    elif op_name == "QLinearMatMul":
        # QLinearMatMul implemented as composite: DequantizeLinear -> MatMul -> QuantizeLinear
        
        # Input tensor dimensions for matrix multiplication
        batch_size = 1
        m = random.randint(4, 16)  # rows of first matrix
        k = random.randint(4, 16)  # cols of first matrix / rows of second matrix  
        n = random.randint(4, 16)  # cols of second matrix
        
        a_shape = [batch_size, m, k]  # First matrix
        b_shape = [batch_size, k, n]  # Second matrix
        output_shape = [batch_size, m, n]  # Result matrix
        
        # Generate quantized input data A (uint8)
        a_data = np.random.randint(0, 256, size=a_shape, dtype=np.uint8)
        a_scale = np.random.uniform(0.001, 0.1)
        a_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Generate quantized input data B (uint8)
        b_data = np.random.randint(0, 256, size=b_shape, dtype=np.uint8)
        b_scale = np.random.uniform(0.001, 0.1)
        b_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Generate output quantization parameters
        c_scale = np.random.uniform(0.001, 0.1)
        c_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Create intermediate tensor names
        dequant_a_name = f"dequant_a_{random.randint(1000, 9999)}"
        dequant_b_name = f"dequant_b_{random.randint(1000, 9999)}"
        matmul_result_name = f"matmul_result_{random.randint(1000, 9999)}"
        
        # Create initializers for quantized inputs and quantization parameters
        initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, a_shape, a_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[1], TensorProto.FLOAT, [], [a_scale]))
        initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [a_zero_point]))
        initializers.append(helper.make_tensor(input_names[3], TensorProto.UINT8, b_shape, b_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[4], TensorProto.FLOAT, [], [b_scale]))
        initializers.append(helper.make_tensor(input_names[5], TensorProto.UINT8, [], [b_zero_point]))
        initializers.append(helper.make_tensor(input_names[6], TensorProto.FLOAT, [], [c_scale]))
        initializers.append(helper.make_tensor(input_names[7], TensorProto.UINT8, [], [c_zero_point]))
        
        # Create four nodes: DequantizeLinear A, DequantizeLinear B, MatMul, QuantizeLinear
        nodes = []
        
        # 1. DequantizeLinear for input A
        dequant_a_node = helper.make_node(
            "DequantizeLinear",
            inputs=[input_names[0], input_names[1], input_names[2]],  # A, A_scale, A_zero_point
            outputs=[dequant_a_name],
            name=f"DequantizeLinear_A_{random.randint(1000, 9999)}"
        )
        nodes.append(dequant_a_node)
        
        # 2. DequantizeLinear for input B
        dequant_b_node = helper.make_node(
            "DequantizeLinear",
            inputs=[input_names[3], input_names[4], input_names[5]],  # B, B_scale, B_zero_point
            outputs=[dequant_b_name],
            name=f"DequantizeLinear_B_{random.randint(1000, 9999)}"
        )
        nodes.append(dequant_b_node)
        
        # 3. MatMul the dequantized values
        matmul_node = helper.make_node(
            "MatMul",
            inputs=[dequant_a_name, dequant_b_name],
            outputs=[matmul_result_name],
            name=f"MatMul_{random.randint(1000, 9999)}"
        )
        nodes.append(matmul_node)
        
        # 4. QuantizeLinear to output
        quant_node = helper.make_node(
            "QuantizeLinear",
            inputs=[matmul_result_name, input_names[6], input_names[7]],  # matmul_result, C_scale, C_zero_point
            outputs=[output_names[0]],
            name=f"QuantizeLinear_{random.randint(1000, 9999)}"
        )
        nodes.append(quant_node)
        
        # Create output info (quantized uint8)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, output_shape)
        
        # Dummy input info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, a_shape)
        
        metadata = {
            "input_shapes": [a_shape, b_shape],
            "output_shapes": [output_shape],
            "a_scale": a_scale,
            "b_scale": b_scale,
            "c_scale": c_scale,
            "operation": "quantized_matmul_composite"
        }
        
        return [input_info], output_info, nodes, initializers, metadata

    elif op_name == "ConvInteger":
        # ConvInteger: convolution with integer arithmetic on quantized tensors
        # Inputs: x (quantized), w (quantized), x_zero_point (optional), w_zero_point (optional)
        
        # Input tensor dimensions [N, C, H, W]
        batch_size = 1
        in_channels = random.randint(1, 4)
        input_height = random.randint(10, 20)
        input_width = random.randint(10, 20)
        
        # Filter dimensions [out_channels, in_channels/group, kernel_h, kernel_w]
        out_channels = random.randint(1, 4)
        kernel_size = random.randint(3, 5)
        group = 1  # Keep simple for now
        
        input_shape = [batch_size, in_channels, input_height, input_width]
        weight_shape = [out_channels, in_channels // group, kernel_size, kernel_size]
        
        # Calculate output dimensions
        pad = kernel_size // 2  # Same padding
        output_height = input_height  # Same padding
        output_width = input_width
        output_shape = [batch_size, out_channels, output_height, output_width]
        
        # Generate quantized input data (uint8)
        x_data = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
        x_zero_point = np.random.randint(0, 256, dtype=np.uint8)
        
        # Generate quantized weight data (int8)
        w_data = np.random.randint(-128, 128, size=weight_shape, dtype=np.int8)
        w_zero_point = np.int8(0)  # Zero point for int8 weights
        
        # Create initializers for quantized inputs
        initializers.append(helper.make_tensor(input_names[0], TensorProto.UINT8, input_shape, x_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[1], TensorProto.INT8, weight_shape, w_data.flatten().tolist()))
        initializers.append(helper.make_tensor(input_names[2], TensorProto.UINT8, [], [x_zero_point]))
        initializers.append(helper.make_tensor(input_names[3], TensorProto.INT8, [], [w_zero_point]))
        
        # Create output info (int32 - ConvInteger outputs accumulated results)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.INT32, output_shape)
        
        # Create ConvInteger node
        node = helper.make_node(
            "ConvInteger",
            inputs=[input_names[0], input_names[1], input_names[2], input_names[3]],  # x, w, x_zero_point, w_zero_point
            outputs=[output_names[0]],
            name=f"{op_name}_node",
            dilations=[1, 1],
            group=group,
            kernel_shape=[kernel_size, kernel_size],
            pads=[pad, pad, pad, pad],
            strides=[1, 1]
        )
        
        # Dummy input info
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
        
        metadata = {
            "input_shapes": [input_shape, weight_shape],
            "output_shapes": [output_shape],
            "x_zero_point": int(x_zero_point),
            "w_zero_point": int(w_zero_point),
            "kernel_size": kernel_size,
            "group": group,
            "output_type": "int32"
        }
        
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "Cast":
        # Cast operator: converts tensor from one type to another
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        
        # Choose input and output types
        input_types = [
            (TensorProto.FLOAT, np.float32),
            (TensorProto.INT32, np.int32),
            (TensorProto.INT64, np.int64),
            (TensorProto.UINT8, np.uint8),
            (TensorProto.INT8, np.int8)
        ]
        
        output_types = [
            TensorProto.FLOAT,
            TensorProto.INT32,
            TensorProto.INT64,
            TensorProto.UINT8,
            TensorProto.INT8
        ]
        
        input_proto_type, input_np_type = random.choice(input_types)
        output_proto_type = random.choice(output_types)
        
        # Generate input data based on input type
        if input_proto_type == TensorProto.FLOAT:
            data = np.random.randn(*shape).astype(input_np_type)
        elif input_proto_type in [TensorProto.INT32, TensorProto.INT64]:
            data = np.random.randint(-100, 100, size=shape).astype(input_np_type)
        else:  # UINT8, INT8
            if input_proto_type == TensorProto.UINT8:
                data = np.random.randint(0, 256, size=shape).astype(input_np_type)
            else:  # INT8
                data = np.random.randint(-128, 128, size=shape).astype(input_np_type)
        
        init_tensor = helper.make_tensor(input_names[0], input_proto_type, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        output_info = helper.make_tensor_value_info(output_names[0], output_proto_type, shape)
        
        node = helper.make_node(
            "Cast",
            inputs=[input_names[0]],
            outputs=[output_names[0]],
            to=output_proto_type,
            name=f"{op_name}_node_{input_proto_type}_to_{output_proto_type}"
        )
        
        metadata = {
            "input_shapes": [shape],
            "output_shapes": [shape],
            "input_type": input_proto_type,
            "output_type": output_proto_type
        }
        
        return [input_info], output_info, [node], initializers, metadata

    elif op_name == "DynamicQuantizeLinear":
        # DynamicQuantizeLinear: quantizes a tensor dynamically
        # Input: x (float) -> Outputs: y (quantized), y_scale (float), y_zero_point (uint8)
        
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        
        # Generate input data (float32)
        data = np.random.randn(*shape).astype(np.float32) * 10  # Scale for better quantization
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        
        # Create output infos
        # y: quantized tensor (uint8)
        y_info = helper.make_tensor_value_info(output_names[0], TensorProto.UINT8, shape)
        # y_scale: scale factor (float, scalar)
        y_scale_info = helper.make_tensor_value_info(output_names[1], TensorProto.FLOAT, [])
        # y_zero_point: zero point (uint8, scalar)
        y_zero_point_info = helper.make_tensor_value_info(output_names[2], TensorProto.UINT8, [])
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        
        node = helper.make_node(
            "DynamicQuantizeLinear",
            inputs=[input_names[0]],
            outputs=[output_names[0], output_names[1], output_names[2]],  # y, y_scale, y_zero_point
            name=f"{op_name}_node"
        )
        
        metadata = {
            "input_shapes": [shape],
            "output_shapes": [shape, [], []],  # y_shape, scalar, scalar
            "operation": "dynamic_quantization"
        }
        
        return [input_info], [y_info, y_scale_info, y_zero_point_info], [node], initializers, metadata

    else:
        # Caso di fallback per operatori non gestiti esplicitamente
        shape = [1, random.randint(1,4), random.randint(10,50), random.randint(10,50)]
        data = np.random.randn(*shape).astype(np.float32)
        init_tensor = helper.make_tensor(input_names[0], TensorProto.FLOAT, shape, data.flatten().tolist())
        initializers.append(init_tensor)
        output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, shape)
        node = helper.make_node(op_name, inputs=[input_names[0]], outputs=[output_names[0]],
                                name=f"{op_name}_generic_node")
        
        input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, shape)
        metadata = {"input_shapes": [shape], "output_shapes": [shape]}
        return [input_info], output_info, [node], initializers, metadata

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
    
    opset_imports = [
        helper.make_opsetid("", 13),  # Standard ONNX opset
        helper.make_opsetid("", 20)
    ]
    
    model = helper.make_model(
        graph, 
        producer_name='zant_test_generator',
        producer_version='1.0',
        domain='ai.zant.test',
        model_version=model_id,
        doc_string=f"Test model for {op_name} operation. Generated on {datetime.datetime.now().isoformat()}",
        opset_imports=opset_imports,
        ir_version=6  # Explicitly set IR version 6 which corresponds to opset 10
    )
    model = onnx.shape_inference.infer_shapes(model)
    
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
            # Fix: Use astype(np.float32) instead of default np.float64
            data = np.random.randn(*shape).astype(np.float32)
        elif elem_type == TensorProto.INT64:
            data = np.random.randint(0, 10, size=shape, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported input type: {elem_type}")
        input_data[inp.name] = data
    
    session = ort.InferenceSession(filename)
    output_names = [out.name for out in graph.output]
    outputs = session.run(output_names, input_data)
    
    # Convert numpy arrays to lists for JSON serialization
    outputs_dict = {name: output.tolist() for name, output in zip(output_names, outputs)}
    input_data_dict = {name: data.tolist() for name, data in input_data.items()}
    
    return {"inputs": input_data_dict, "outputs": outputs_dict}

def load_supported_ops(filename="tests/CodeGen/Python-ONNX/available_operations.txt"):
    """Carica le operazioni supportate da un file oppure restituisce una lista di default."""
    try:
        with open(filename, "r") as file:
            ops = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]
            return ops
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default operations.")
        return [
            "LeakyRelu", "Relu", "Sigmoid", "Softmax", "Add", "Ceil", "Div", "Mul", "Sub", "Tanh",
            "Concat", "Gather", "Identity", "Neg", "Reshape", "Resize", "Slice", 
            "Split", "Transpose", "Unsqueeze", "ReduceMean", "Conv", "MatMul", "Gemm", "MaxPool",
            "Clip", "Cast", "ConvInteger", "DynamicQuantizeLinear", "Pad"
            # "Shape" removed from the list
        ]


def main():
    print(f"\n __main__")
    parser = argparse.ArgumentParser(description="Generate fuzzed ONNX models and save execution data in JSON.")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of models to generate for each operation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for random generation (for reproducibility).")
    parser.add_argument("--output-dir", type=str, default="datasets/oneOpModels",
                        help="Directory to save generated models.")
    parser.add_argument("--metadata-file", type=str, default="datasets/oneOpModels/results.json",
                        help="File to save metadata and execution data.")
    parser.add_argument("--op", type=str, default="all",
                        help="Name of the operation you want to generate and test")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    output_dir = args.output_dir
    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.op == "all" : supported_ops = load_supported_ops() 
    else: supported_ops = [args.op] 
    
    print(f"\n supported_ops : {supported_ops}")
    
    all_models = []
    
    for op in supported_ops:
        for i in range(args.iterations):

            print("Saving model to " + output_dir) 
            filename = f"{output_dir}{op}_{i}.onnx"
            try: 
                metadata = generate_model(op, filename, i)
                print(f"Successfully generated model for {op} (ID: {i})")
            except Exception as e:
                print(f"Error generating model for {op} (ID: {i}): {e}")
                traceback.print_exc()

            try:
                data = run_model(filename)
                model_info = {
                    "operation": op,
                    "model_id": i,
                    "inputs": data["inputs"],
                    "outputs": data["outputs"],
                    "metadata": metadata
                }

                # FOR DEBUG
                # print( "model info:")
                # pprint.pprint(model_info)
                
                test_file_name = f"{output_dir}{op}_{i}_user_tests.json"
                print(f"Saving user tests to {test_file_name}")
                
                user_tests = []

                for (in_key, out_key) in zip(data["inputs"].keys(), data["outputs"].keys()):
                    in_array = np.array(data["inputs"][in_key]).flatten().tolist()
                    out_array = np.array(data["outputs"][out_key]).flatten().tolist()
                    
                    test_model_info = {
                        "name": op,
                        "type": "exact",
                        "input": in_array,
                        "output": out_array,
                        "expected_class": 0
                    }
                    user_tests.append(test_model_info)

                with open(test_file_name, 'w') as f:
                    json.dump(user_tests, f, indent=2)
                print(f"Execution data saved to {test_file_name}")
                    
                
                all_models.append(model_info)
                print(f"Successfully ran model for {op} (ID: {i})")
            except Exception as e:
                print(f"Error running model for {op} (ID: {i}): {e} ")

    
    with open(args.metadata_file, 'w') as f:
        json.dump(all_models, f, indent=2)
    print(f"Execution data saved to {args.metadata_file}")

if __name__ == "__main__":
    main()