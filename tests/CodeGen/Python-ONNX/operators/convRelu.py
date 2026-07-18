import numpy as np
import random
from onnx import helper, TensorProto

def generate_conv_relu_model(input_names, output_names):
    """
    Generates a fused Conv+ReLU operator model.
    """
    initializers = []
   
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
   
    # Add padding parameter
    pad_h = random.randint(0, 2)
    pad_w = random.randint(0, 2)
    pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
   
    # Calculate output dimensions
    H_out = (H + 2*pad_h - (kH-1)*dilations[0] - 1) // strides[0] + 1
    W_out = (W + 2*pad_w - (kW-1)*dilations[1] - 1) // strides[1] + 1
   
    # Ensure valid output dimensions
    if H_out <= 0 or W_out <= 0:
        dilations = [1, 1]
        pad_h = pad_w = 1
        pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
        H_out = (H + 2*pad_h - kH) // strides[0] + 1
        W_out = (W + 2*pad_w - kW) // strides[1] + 1
   
    output_shape = [N, M, H_out, W_out]
    
    # Output intermedio della Conv
    conv_output_name = f"{input_names[0]}_conv_intermediate"
    
    # Nodo Conv
    conv_node = helper.make_node(
        "Conv", 
        inputs=[input_names[0], input_names[1]], 
        outputs=[conv_output_name],  # Output intermedio
        kernel_shape=kernel_shape, 
        strides=strides, 
        dilations=dilations,
        pads=[pad_h, pad_w, pad_h, pad_w],
        name=f"Conv_node_k{kernel_shape}_s{strides}_d{dilations}_p{[pad_h, pad_w]}"
    )
    
    # Nodo ReLU che prende l'output della Conv
    relu_node = helper.make_node(
        "Relu",
        inputs=[conv_output_name],
        outputs=[output_names[0]],  # Output finale del grafo
        name="Relu_node"
    )
   
    # Output finale (dopo ReLU)
    output_info = helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, output_shape)
   
    input_info = helper.make_tensor_value_info("useless_input", TensorProto.FLOAT, input_shape)
    
    metadata = {
        "input_shapes": [input_shape, weight_shape], 
        "output_shapes": [output_shape],
        "kernel_shape": kernel_shape, 
        "strides": strides, 
        "dilations": dilations,
        "pads": [pad_h, pad_w, pad_h, pad_w],
        "fused_op": "Conv+ReLU"  # Indicazione che è un operatore fuso
    }
    
    # Restituisce entrambi i nodi: prima Conv, poi ReLU
    return [input_info], output_info, [conv_node, relu_node], initializers, metadata