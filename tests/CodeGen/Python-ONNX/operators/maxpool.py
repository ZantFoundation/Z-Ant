import numpy as np
import random
from onnx import helper, TensorProto

import numpy as np
import random
from onnx import helper, TensorProto


def calculate_output_size(input_size, kernel_size, stride, pad_before, pad_after, ceil_mode=0):
    """Calculate output size for pooling operation."""
    if ceil_mode:
        return int(np.ceil((input_size + pad_before + pad_after - kernel_size) / stride)) + 1
    else:
        return int(np.floor((input_size + pad_before + pad_after - kernel_size) / stride)) + 1


def generate_maxpool_model(input_names, output_names):
    """
    Generates a MaxPool operator model.
    
    Args:
        input_names: List of input names (expects at least 1 for the input data)
        output_names: List of output names (expects at least 1)
    
    Returns:
        tuple: (input_infos, output_info, nodes, initializers, metadata)
    """
    if len(input_names) < 1:
        raise ValueError("input_names must contain at least 1 element: [data_input]")
    if len(output_names) < 1:
        raise ValueError("output_names must contain at least 1 element")
    
    initializers = []
     # MaxPool operator: applies max pooling with configurable parameters
    # Supports various spatial dimensions and parameter combinations
    
    # Random values for batch and channel dimensions
    N = random.randint(1, 3)  # Batch size
    C = random.randint(1, 4)  # Number of channels
    

    H = random.randint(4, 12)  # Height
    W = random.randint(4, 12)  # Width
    input_shape = [N, C, H, W]
    spatial_dims = [H, W]
    
    # Kernel shape for 2D pooling
    kernel_h = random.randint(2, min(4, H))
    kernel_w = random.randint(2, min(4, W))
    kernel_shape = [kernel_h, kernel_w]
        
    
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
    strides = [random.randint(1, min(3, kernel_h)), random.randint(1, min(3, kernel_w))]
   
    
    # Random auto_pad mode
    #"NOTSET" working
    #"VALID" working
    #"SAME_UPPER" broken
    #"SAME_LOWER" broken
    auto_pad_options = ["NOTSET", "VALID" ] #"SAME_LOWER" AND "SAME_UPPER" are broken
    auto_pad = random.choice(auto_pad_options)
    
    # Calculate padding and output dimensions based on auto_pad
    if auto_pad == "NOTSET":
        # Explicit padding - random but reasonable values
       
        max_pad_h = min(2, kernel_h // 2)
        max_pad_w = min(2, kernel_w // 2)
        pads = [
            random.randint(0, max_pad_h),  # top
            random.randint(0, max_pad_w),  # left
            random.randint(0, max_pad_h),  # bottom
            random.randint(0, max_pad_w)   # right
        ]
       
    else:
        # Auto padding - let ONNX calculate
        pads = [0] * (len(spatial_dims) * 2)

    # ceil_mode must be decided BEFORE we compute the output shape so the
    # formula matches what ORT will report at load time.
    ceil_mode = random.choice([0, 1])

    # Calculate output dimensions
    if auto_pad == "NOTSET":
        # Manual calculation for explicit padding (honour ceil_mode).
        if ceil_mode:
            H_out = int(np.ceil((H + pads[0] + pads[2] - kernel_shape[0]) / strides[0])) + 1
            W_out = int(np.ceil((W + pads[1] + pads[3] - kernel_shape[1]) / strides[1])) + 1
        else:
            H_out = ((H + pads[0] + pads[2] - kernel_shape[0]) // strides[0]) + 1
            W_out = ((W + pads[1] + pads[3] - kernel_shape[1]) // strides[1]) + 1
        output_shape = [N, C, H_out, W_out]
        
    elif auto_pad == "VALID":
        # No padding. Honour ceil_mode so the declared shape matches what ORT
        # produces (otherwise ORT writes more outputs than the codegen-allocated
        # buffer holds and the user_test comparison drifts out of sync).
        if ceil_mode:
            H_out = int(np.ceil((H - kernel_shape[0]) / strides[0])) + 1
            W_out = int(np.ceil((W - kernel_shape[1]) / strides[1])) + 1
        else:
            H_out = ((H - kernel_shape[0]) // strides[0]) + 1
            W_out = ((W - kernel_shape[1]) // strides[1]) + 1
        output_shape = [N, C, H_out, W_out]
        
    else:  # SAME_UPPER or SAME_LOWER
        # Output size preserves input size divided by stride
        H_out = int(np.ceil(H / strides[0]))
        W_out = int(np.ceil(W / strides[1]))
        output_shape = [N, C, H_out, W_out]
        
    
    # Validate output dimensions are positive
    if any(dim <= 0 for dim in output_shape[2:]):  # Check spatial dimensions
        # Fallback to simpler, guaranteed-working configuration
                        
        kernel_shape = [2, 2]
        strides = [2, 2]
        auto_pad = "VALID"
        pads = [0, 0, 0, 0]
        H_out = ((H - 2) // 2) + 1
        W_out = ((W - 2) // 2) + 1
        output_shape = [N, C, max(1, H_out), max(1, W_out)]

       
    
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
    
    # Optional parameters (ceil_mode already chosen above)
    storage_order = random.choice([0, 1])  # 0 = row major, 1 = column major
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
        "MaxPool",
        inputs=[input_names[0]],
        outputs=[output_names[0]],
        name=f"MaxPool_node_N{N}_C{C}_k{kernel_shape}_s{strides}",
        **node_attrs
    )
    
    metadata = {
        "input_shapes": [input_shape],
        "output_shapes": [output_shape],
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
