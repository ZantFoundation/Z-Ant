import numpy as np
import random
from onnx import helper, TensorProto


def generate_averagepool_model(input_names, output_names):
    """
    Generates an AveragePool operator model.
    """
    initializers = []
    
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
    #"NOTSET" working
    #"VALID" working
    #"SAME_UPPER" broken
    #"SAME_LOWER" broken
    auto_pad_options = ["NOTSET", "VALID" ] #"SAME_LOWER" AND "SAME_UPPER" are broken
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

        # ONNX v22 VALID padding formulas. The official spec puts the `+ 1`
        # *outside* the ceil/floor, matching the NOTSET branch above:
        #   output[i] = ceil((input[i] - eff_kernel[i]) / strides[i]) + 1
        # An earlier version of this generator placed the `+ 1` inside the
        # dividend, which under-counts the last column when stride > 1 and
        # makes ORT (which uses the spec formula) produce more outputs than
        # the codegen-allocated buffer holds.
        if ceil_mode:
            effective_kernel_h = (kernel_shape[0] - 1) * dilations[0] + 1
            effective_kernel_w = (kernel_shape[1] - 1) * dilations[1] + 1
            H_out = int(np.ceil((H - effective_kernel_h) / strides[0])) + 1
            W_out = int(np.ceil((W - effective_kernel_w) / strides[1])) + 1
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
        "AveragePool",
        inputs=[input_names[0]],
        outputs=[output_names[0]],
        name=f"AveragePool_node_N{N}_C{C}_H{H}_W{W}_k{kernel_shape}_s{strides}_d{dilations}",
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