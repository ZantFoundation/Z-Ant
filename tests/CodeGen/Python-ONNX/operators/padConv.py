import numpy as np
import random
from onnx import helper, TensorProto


def generate_padconv_model(input_names, output_names):
    """
    Generates a multi-node model: Pad -> Conv
    
    This demonstrates how to create a sequential graph where:
    1. Input is padded using Pad operator
    2. Padded output is fed into Conv operator
    3. Conv produces final output
    """
    initializers = []
    nodes = []
    
    # ==================== STEP 1: Generate Input Tensor ====================
    # Create input dimensions [N, C, H, W]
    N = 1
    C = random.randint(1, 4)
    H = random.randint(10, 20)
    W = random.randint(10, 20)
    input_shape = [N, C, H, W]
    
    # Generate random input data
    input_data = np.random.randn(*input_shape).astype(np.float32)
    input_tensor = helper.make_tensor(
        input_names[0], 
        TensorProto.FLOAT, 
        input_shape, 
        input_data.flatten().tolist()
    )
    initializers.append(input_tensor)
    
    # ==================== STEP 2: Configure Pad Operation ====================
    # Define padding: [pad_top, pad_left, pad_bottom, pad_right]
    # ONNX format: [x1_begin, x2_begin, x3_begin, x4_begin, x1_end, x2_end, x3_end, x4_end]
    pad_h = random.randint(1, 3)
    pad_w = random.randint(1, 3)
    
    # For 4D tensor [N, C, H, W], we only pad H and W dimensions
    # Format: [pad_N_begin, pad_C_begin, pad_H_begin, pad_W_begin, 
    #          pad_N_end, pad_C_end, pad_H_end, pad_W_end]
    pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
    
    # Create pads tensor (must be INT64)
    pads_tensor = helper.make_tensor(
        input_names[1], 
        TensorProto.INT64, 
        [len(pads)], 
        pads
    )
    initializers.append(pads_tensor)
    
    # Calculate padded dimensions
    padded_H = H + 2 * pad_h
    padded_W = W + 2 * pad_w
    padded_shape = [N, C, padded_H, padded_W]
    
    # Choose padding mode
    pad_mode = random.choice(["constant", "edge"])
    
    # Create intermediate tensor name for Pad output
    pad_output_name = f"pad_output_{random.randint(1000, 9999)}"
    
    # Handle constant value if mode is "constant"
    node_inputs = [input_names[0], input_names[1]]
    constant_value = 0.0
    
    if pad_mode == "constant":
        constant_value = round(random.uniform(-1.0, 1.0), 2)
        constant_tensor = helper.make_tensor(
            input_names[2], 
            TensorProto.FLOAT, 
            [],  # Scalar
            [constant_value]
        )
        initializers.append(constant_tensor)
        node_inputs.append(input_names[2])
    
    # Create Pad node
    pad_node = helper.make_node(
        "Pad",
        inputs=node_inputs,
        outputs=[pad_output_name],
        mode=pad_mode,
        name=f"Pad_node_{pad_mode}_p{pad_h}x{pad_w}"
    )
    nodes.append(pad_node)
    
    # ==================== STEP 3: Configure Conv Operation ====================
    # Define convolution kernel
    kH = random.randint(3, min(7, padded_H // 2))
    kW = random.randint(3, min(7, padded_W // 2))
    kernel_shape = [kH, kW]
    
    # Define output channels
    M = random.randint(1, 4)
    
    # Generate weight tensor [out_channels, in_channels, kernel_h, kernel_w]
    weight_shape = [M, C, kH, kW]
    weight_data = np.random.randn(*weight_shape).astype(np.float32)
    weight_tensor = helper.make_tensor(
        input_names[3] if pad_mode == "constant" else input_names[2], 
        TensorProto.FLOAT, 
        weight_shape, 
        weight_data.flatten().tolist()
    )
    initializers.append(weight_tensor)
    
    # Define convolution parameters
    strides = [random.randint(1, 2), random.randint(1, 2)]
    dilations = [1, 1]  # Keep simple
    conv_pads = [0, 0, 0, 0]  # No additional padding (already padded!)
    
    # Calculate Conv output dimensions
    # Formula: output_size = (input_size + 2*pad - (kernel_size-1)*dilation - 1) / stride + 1
    H_out = (padded_H + 2*conv_pads[0] - (kH-1)*dilations[0] - 1) // strides[0] + 1
    W_out = (padded_W + 2*conv_pads[1] - (kW-1)*dilations[1] - 1) // strides[1] + 1
    
    # Ensure valid output dimensions
    if H_out <= 0 or W_out <= 0:
        strides = [1, 1]
        H_out = (padded_H + 2*conv_pads[0] - (kH-1)*dilations[0] - 1) // strides[0] + 1
        W_out = (padded_W + 2*conv_pads[1] - (kW-1)*dilations[1] - 1) // strides[1] + 1
    
    output_shape = [N, M, H_out, W_out]
    
    # Create Conv node - input is the Pad output
    conv_node = helper.make_node(
        "Conv",
        inputs=[
            pad_output_name,  
            input_names[3] if pad_mode == "constant" else input_names[2]
        ],
        outputs=[output_names[0]],
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations,
        pads=conv_pads,
        name=f"Conv_node_k{kernel_shape}_s{strides}"
    )
    nodes.append(conv_node)
    
    # ==================== STEP 4: Create Graph Metadata ====================
    # Create dummy input info
    input_info = helper.make_tensor_value_info(
        "useless_input", 
        TensorProto.FLOAT, 
        input_shape
    )
    
    # Create output info
    output_info = helper.make_tensor_value_info(
        output_names[0], 
        TensorProto.FLOAT, 
        output_shape
    )
    
    # Metadata for debugging
    metadata = {
        "operation": "Pad -> Conv",
        "input_shapes": [input_shape],
        "intermediate_shapes": [padded_shape],
        "output_shapes": [output_shape],
        "pad": {
            "mode": pad_mode,
            "pads": pads,
            "constant_value": constant_value if pad_mode == "constant" else None
        },
        "conv": {
            "kernel_shape": kernel_shape,
            "strides": strides,
            "dilations": dilations,
            "pads": conv_pads,
            "weight_shape": weight_shape
        }
    }
    
    # Return: input_info, output_info, nodes_list, initializers, metadata
    return [input_info], output_info, nodes, initializers, metadata
