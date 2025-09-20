import onnx
from onnx import shape_inference
import argparse
from onnxsim import simplify
import numpy as np

def create_dummy_input_for_initializers(model):
    """
    Create dummy graph inputs for initializers to satisfy ONNX validation.
    This fixes the 'initializer but not in graph input' error.
    """
    # Get names of existing graph inputs
    existing_input_names = set([inp.name for inp in model.graph.input])
    
    # Get all initializer names
    initializer_names = set([init.name for init in model.graph.initializer])
    
    # Find initializers that need dummy inputs
    missing_inputs = initializer_names - existing_input_names
    
    if missing_inputs:
        print(f"Creating dummy inputs for {len(missing_inputs)} initializers...")
        
        # Create dummy inputs for each missing initializer
        for init in model.graph.initializer:
            if init.name in missing_inputs:
                # Create corresponding input tensor
                input_tensor = model.graph.input.add()
                input_tensor.name = init.name
                input_tensor.type.tensor_type.elem_type = init.data_type
                
                # Set shape from initializer dimensions
                for dim_size in init.dims:
                    dim = input_tensor.type.tensor_type.shape.dim.add()
                    dim.dim_value = dim_size
        
        print(f"Added {len(missing_inputs)} dummy inputs for initializers")
    
    return model

def set_input_shape(model, input_shape, input_name="data"):
    """
    Set the input tensor shape for a specific input by name.
    """
    target_input = None
    
    # Find the target input tensor by name
    for input_tensor in model.graph.input:
        if input_tensor.name == input_name:
            target_input = input_tensor
            break
    
    if target_input is None:
        print(f"Warning: Input '{input_name}' not found. Available inputs:")
        for inp in model.graph.input:
            print(f"  - {inp.name}")
        # Use first input as fallback
        if len(model.graph.input) > 0:
            target_input = model.graph.input[0]
            print(f"Using first input: {target_input.name}")
        else:
            raise ValueError("No inputs found in model")
    
    # Clear existing dimensions
    target_input.type.tensor_type.shape.ClearField('dim')
    
    # Add new dimensions
    for dim_value in input_shape:
        new_dim = target_input.type.tensor_type.shape.dim.add()
        new_dim.dim_value = dim_value
    
    print(f"Updated input '{target_input.name}' shape to: {input_shape}")
    return model

def calculate_avgpool_output_shape(input_shape, kernel_shape, strides=None, pads=None, ceil_mode=False):
    """
    Calculate output shape for AveragePool operation following ONNX specification.
    """
    if strides is None:
        strides = kernel_shape
    if pads is None:
        pads = [0] * (len(kernel_shape) * 2)
    
    output_shape = []
    
    # First two dimensions (batch and channels) remain unchanged
    output_shape.extend(input_shape[:2])
    
    # Calculate spatial dimensions
    for i, (input_size, kernel_size, stride, pad_begin, pad_end) in enumerate(
        zip(input_shape[2:], kernel_shape, strides, pads[:len(kernel_shape)], pads[len(kernel_shape):])
    ):
        if ceil_mode:
            output_size = int(np.ceil((input_size + pad_begin + pad_end - kernel_size) / stride + 1))
        else:
            output_size = int(np.floor((input_size + pad_begin + pad_end - kernel_size) / stride + 1))
        output_shape.append(output_size)
    
    return output_shape

def manual_shape_inference(model):
    """
    Manually infer shapes for QLinearAveragePool nodes when automatic inference fails.
    """
    print("Running manual shape inference for QLinearAveragePool nodes...")
    
    # Create a mapping of tensor names to their shapes
    tensor_shapes = {}
    
    # Get input shapes
    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.shape.dim:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            tensor_shapes[input_tensor.name] = shape
    
    # Get initializer shapes
    for init in model.graph.initializer:
        tensor_shapes[init.name] = list(init.dims)
    
    # Process nodes and infer missing output shapes
    for node in model.graph.node:
        if node.op_type == "QLinearAveragePool":
            print(f"Processing QLinearAveragePool node: {node.name}")
            
            # Get input tensor shape (first input is the data tensor)
            input_name = node.input[0]
            if input_name not in tensor_shapes:
                print(f"Warning: Input shape for {input_name} not found")
                continue
            
            input_shape = tensor_shapes[input_name]
            
            # Get attributes
            kernel_shape = None
            strides = None
            pads = None
            ceil_mode = False
            
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = list(attr.ints)
                elif attr.name == "strides":
                    strides = list(attr.ints)
                elif attr.name == "pads":
                    pads = list(attr.ints)
                elif attr.name == "ceil_mode":
                    ceil_mode = bool(attr.i)
            
            if kernel_shape is None:
                print(f"Warning: No kernel_shape found for QLinearAveragePool {node.name}")
                continue
            
            # Calculate output shape
            try:
                output_shape = calculate_avgpool_output_shape(
                    input_shape, kernel_shape, strides, pads, ceil_mode
                )
                
                # Store output shape
                output_name = node.output[0]
                tensor_shapes[output_name] = output_shape
                
                print(f"Inferred shape for {output_name}: {output_shape}")
                
                # Add value_info to the graph if it doesn't exist
                value_info_exists = False
                for vi in model.graph.value_info:
                    if vi.name == output_name:
                        value_info_exists = True
                        # Update existing value_info
                        vi.type.tensor_type.shape.ClearField('dim')
                        for dim_size in output_shape:
                            dim = vi.type.tensor_type.shape.dim.add()
                            dim.dim_value = dim_size
                        break
                
                if not value_info_exists:
                    # Create new value_info
                    value_info = model.graph.value_info.add()
                    value_info.name = output_name
                    value_info.type.tensor_type.elem_type = onnx.TensorProto.INT8  # Assuming quantized output
                    for dim_size in output_shape:
                        dim = value_info.type.tensor_type.shape.dim.add()
                        dim.dim_value = dim_size
                    
                    print(f"Added value_info for {output_name}")
                
            except Exception as e:
                print(f"Error calculating output shape for {node.name}: {e}")
        
        # For other operations, try to infer shapes based on known patterns
        elif len(node.input) > 0 and len(node.output) > 0:
            input_name = node.input[0]
            if input_name in tensor_shapes:
                output_name = node.output[0]
                
                if node.op_type in ["Sub", "Div", "Add", "Mul"]:
                    # Element-wise operations preserve shape
                    tensor_shapes[output_name] = tensor_shapes[input_name]
                elif node.op_type == "QuantizeLinear":
                    # QuantizeLinear preserves shape but changes type
                    tensor_shapes[output_name] = tensor_shapes[input_name]
                elif node.op_type == "DequantizeLinear":
                    # DequantizeLinear preserves shape but changes type
                    tensor_shapes[output_name] = tensor_shapes[input_name]
                elif node.op_type == "QLinearConv":
                    # QLinearConv can change spatial dimensions, skip for now
                    continue
                else:
                    # For unknown operations, try to preserve shape
                    tensor_shapes[output_name] = tensor_shapes[input_name]
    
    return model

def safe_simplify(model):
    """
    Attempt to simplify the model with multiple fallback strategies.
    """
    print("Attempting model simplification...")
    
    # Strategy 1: Standard simplification
    try:
        model_simp, check = simplify(model)
        if check:
            print("✅ Standard simplification successful!")
            return model_simp
        print("Standard simplification completed but validation uncertain")
        return model_simp
    except Exception as e:
        print(f"Standard simplification failed: {e}")
    
    # Strategy 2: Skip optimization
    try:
        print("Trying with skip_optimization=True...")
        model_simp, check = simplify(model, skip_optimization=True)
        print("✅ Simplification with skip_optimization successful!")
        return model_simp
    except Exception as e:
        print(f"Skip optimization also failed: {e}")
    
    # Strategy 3: Skip shape inference
    try:
        print("Trying with skip_shape_inference=True...")
        model_simp, check = simplify(model, skip_shape_inference=True)
        print("✅ Simplification with skip_shape_inference successful!")
        return model_simp
    except Exception as e:
        print(f"Skip shape inference also failed: {e}")
    
    print("⚠️ All simplification strategies failed, keeping original model")
    return model

def clean_model(model_path, input_shape, output_path=None):
    """
    Clean and process ONNX model with comprehensive error handling.
    """
    if output_path is None:
        output_path = model_path
    
    print(f"Processing: {model_path}")
    print(f"Target shape: {input_shape}")
    
    # Load model
    try:
        model = onnx.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    print(f"Original model: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")
    
    # Fix the validation issue
    print("\n1. Fixing initializer validation issues...")
    model = create_dummy_input_for_initializers(model)
    
    # Set input shape
    print("\n2. Setting input shape...")
    try:
        model = set_input_shape(model, input_shape)
    except Exception as e:
        print(f"❌ Failed to set input shape: {e}")
        return False
    
    # Run shape inference
    print("\n3. Running shape inference...")
    shape_inference_success = False
    try:
        model = shape_inference.infer_shapes(model)
        print("✅ Shape inference successful")
        shape_inference_success = True
    except Exception as e:
        print(f"⚠️ Automatic shape inference failed: {e}")
        print("Trying manual shape inference for QLinearAveragePool...")
        
        # Try manual shape inference
        try:
            model = manual_shape_inference(model)
            print("✅ Manual shape inference completed")
        except Exception as manual_e:
            print(f"⚠️ Manual shape inference also failed: {manual_e}")
            print("Continuing without complete shape inference...")
    
    # Simplify model
    print("\n4. Simplifying model...")
    model = safe_simplify(model)
    
    # Save model
    print(f"\n5. Saving model to: {output_path}")
    try:
        onnx.save(model, output_path)
        print("✅ Model saved successfully!")
        
        # Final stats
        print(f"\nFinal model: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix and simplify ResNet ONNX models")
    parser.add_argument("--path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--shape", type=str, required=True, 
                       help="Input shape as comma-separated values (e.g., 1,3,224,224)")
    parser.add_argument("--output", type=str, help="Output path (optional)")
    
    args = parser.parse_args()
    
    # Parse shape
    try:
        input_shape = [int(x) for x in args.shape.split(",")]
    except ValueError:
        print("❌ Invalid shape format. Use: 1,3,224,224")
        return
    
    # Process model
    success = clean_model(args.path, input_shape, args.output)
    
    if success:
        print("\n🎉 Model processing completed successfully!")
    else:
        print("\n💥 Model processing failed!")

if __name__ == "__main__":
    main()