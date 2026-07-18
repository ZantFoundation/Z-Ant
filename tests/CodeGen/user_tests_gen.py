import onnx
import onnxruntime as ort
import argparse
import json
import numpy as np
import os
import traceback

def get_runtime_input_shapes_and_types(model_path):
    """Get runtime inputs with their data types"""
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model_path, sess_options)
    except Exception as e:
        print(f"Warning: Failed to create session with disabled optimizations: {e}")
        try:
            sess = ort.InferenceSession(model_path)
        except Exception as e2:
            print(f"Error: Failed to create any session: {e2}")
            return {}
    
    input_info = {}
    
    for input_meta in sess.get_inputs():
        name = input_meta.name
        shape = input_meta.shape
        dtype_str = input_meta.type
        
        # Replace dynamic dimensions with 1
        dims = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
        
        # Map ONNX type strings to numpy dtypes
        if 'uint8' in dtype_str:
            np_dtype = np.uint8
        elif 'int8' in dtype_str:
            np_dtype = np.int8
        elif 'uint16' in dtype_str:
            np_dtype = np.uint16
        elif 'int16' in dtype_str:
            np_dtype = np.int16
        elif 'uint32' in dtype_str:
            np_dtype = np.uint32
        elif 'int32' in dtype_str:
            np_dtype = np.int32
        elif 'int64' in dtype_str:   #This type has been added to support loop nodes
            np_dtype = np.int64;     #Idem
        elif 'float16' in dtype_str:
            np_dtype = np.float16
        elif 'float' in dtype_str:
            np_dtype = np.float32
        elif 'double' in dtype_str:
            np_dtype = np.float64
        elif 'bool' in dtype_str:   #This type has been added to support loop nodes
            np_dtype = bool         #Idem
        else:
            print(f"Warning: Unknown dtype {dtype_str}, defaulting to float32")
            np_dtype = np.float32
        
        input_info[name] = {
            'shape': dims,
            'dtype': np_dtype,
            'dtype_str': dtype_str
        }      
        
        
        print(f"Input '{name}': shape={dims}, dtype={dtype_str} -> {np_dtype}")
        
    return input_info

def generate_random_input(shape, dtype, input_name="", normalize=False):
    """Generate random input data with correct dtype"""
    
    if dtype == np.uint8:
        # For uint8 quantized inputs, generate values in full range [0, 255]
        data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    
    elif dtype == np.int8:
        # For int8 quantized inputs, generate values in range [-128, 127]
        data = np.random.randint(-128, 128, size=shape, dtype=np.int8)
    
    elif dtype in [np.uint16, np.uint32]:
        # For unsigned integer types
        max_val = np.iinfo(dtype).max
        data = np.random.randint(0, min(max_val, 1000), size=shape, dtype=dtype)
    
    elif dtype in [np.int16, np.int32]:
        # For signed integer types
        max_val = min(np.iinfo(dtype).max, 1000)
        min_val = max(np.iinfo(dtype).min, -1000)
        data = np.random.randint(min_val, max_val, size=shape, dtype=dtype)
    
    elif dtype in [np.float16, np.float32, np.float64]:
        # For floating point types
        if "mnist" in input_name.lower() or any(dim == 28 for dim in shape):
            # For MNIST-like inputs, generate values in [0, 1] range
            data = np.random.rand(*shape).astype(dtype)
        elif "image" in input_name.lower() or len(shape) == 4:
            # For image inputs, generate normalized values [0, 1]
            data = np.random.rand(*shape).astype(dtype)
        else:
            # For other inputs, use standard normal distribution but clamp to reasonable range
            data = np.random.randn(*shape).astype(dtype)
            data = np.clip(data, -3.0, 3.0)

    else:
        print(f"Warning: Unsupported dtype {dtype}, using float32")
        data = np.random.rand(*shape).astype(np.float32)
    
    # Apply normalization if requested
    if normalize and dtype in [np.float16, np.float32, np.float64]:
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            data = (data - mean) / std
    
    return data

def generate_random_loop_input(dtype):

    if dtype == bool:
        return np.array(True, dtype=bool)
    else:
        data = np.array(np.random.randint(0,10), dtype=dtype)  #10 iteration is enought to test

    return data


def run_onnx_inference(model_path, input_info, normalize=False):
    """Run inference with the ONNX model"""
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model_path, sess_options)
    except Exception as e:
        print(f"Warning: Failed to create optimized session: {e}")
        try:
            sess = ort.InferenceSession(model_path)
        except Exception as e2:
            print(f"Error: Failed to create any inference session: {e2}")
            raise e2
    
    # Generate inputs with correct data types
    inputs = {}
    for name, info in input_info.items():
        shape = info['shape']
        dtype = info['dtype']
        #Handling loop inputs
        if name in ['M', 'cond', 'trip_count']:
            inputs[name] = generate_random_loop_input(dtype)
            continue  
        inputs[name] = generate_random_input(shape, dtype, name, normalize)
        print(f"Generated input '{name}': shape={inputs[name].shape}, dtype={inputs[name].dtype}")
    
    try:
        # Run inference
        outputs = sess.run(None, inputs)
        output_names = [output.name for output in sess.get_outputs()]
        return inputs, dict(zip(output_names, outputs))
    except Exception as e:
        print(f"Inference failed: {e}")
        
        # Try with zero input of correct dtype
        print("Retrying with zero input...")
        for name, info in input_info.items():
            shape = info['shape']
            dtype = info['dtype']
            inputs[name] = np.zeros(shape, dtype=dtype)
        
        try:
            outputs = sess.run(None, inputs)
            output_names = [output.name for output in sess.get_outputs()]
            return inputs, dict(zip(output_names, outputs))
        except Exception as e2:
            print(f"Zero input also failed: {e2}")
            raise e2

def main():
    parser = argparse.ArgumentParser(description="Run ONNX model multiple times with random inputs and save execution data.")
    parser.add_argument("--model", type=str, required=True, help="your ONNX model.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of randomized inference runs.")
    parser.add_argument("--normalize", action="store_true", help="Normalize the input fuzzing values.")
    args = parser.parse_args()
    
    model_name = args.model
    model_path = f"datasets/models/{model_name}/{model_name}.onnx"
    iterations = args.iterations
    normalize = args.normalize
    
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return
    
    try:
        model = onnx.load(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Get runtime input shapes and types
    try:
        input_info = get_runtime_input_shapes_and_types(model_path)
        print(f"Runtime input info: {input_info}")
    except Exception as e:
        print(f"Failed to get input info: {e}")
        return
    
    if not input_info:
        print("No runtime input infos found in the model")
        return
    
    user_tests = []
    
    for i in range(iterations):
        try:
            inputs, outputs = run_onnx_inference(model_path, input_info, normalize)
            
            # Create test case
            input_names = list(inputs.keys())
            output_names = list(outputs.keys())
            
            if len(input_names) > 0 and len(output_names) > 0:
                # Use the first input and first output
                in_key = input_names[0]
                out_key = output_names[0]
                
                # Convert to lists, handling different dtypes
                in_array = inputs[in_key].flatten()
                out_array = outputs[out_key].flatten()
                
                # Convert to Python native types for JSON serialization
                if in_array.dtype == np.uint8:
                    in_list = [int(x) for x in in_array]
                elif in_array.dtype == np.int8:
                    in_list = [int(x) for x in in_array]
                elif in_array.dtype in [np.int16, np.int32, np.uint16, np.uint32]:
                    in_list = [int(x) for x in in_array]
                else:
                    in_list = [float(x) for x in in_array]
                
                if out_array.dtype == np.uint8:
                    out_list = [int(x) for x in out_array]
                elif out_array.dtype == np.int8:
                    out_list = [int(x) for x in out_array]
                elif out_array.dtype in [np.int16, np.int32, np.uint16, np.uint32]:
                    out_list = [int(x) for x in out_array]
                else:
                    out_list = [float(x) for x in out_array]
                
                test_model_info = {
                    "name": model_name,
                    "type": "exact",
                    "input": in_list,
                    "output": out_list,
                    "expected_class": 0
                }
                user_tests.append(test_model_info)
                print(f"✓ Run {i+1}/{iterations} completed")
                print(f"  Input shape: {inputs[in_key].shape}, dtype: {inputs[in_key].dtype}")
                print(f"  Output shape: {outputs[out_key].shape}, dtype: {outputs[out_key].dtype}")
                print(f"  Sample input values: {in_list[:5]}...")
                print(f"  Sample output values: {out_list[:5]}...")
            else:
                print(f"⚠️  No valid inputs/outputs found in run {i+1}")
                
        except Exception as e:
            print(f"⚠️  Error during run {i+1}: {e}")
            traceback.print_exc()
    
    if user_tests:
        # Save all test cases
        test_file_name = f"datasets/models/{model_name}/user_tests.json"
        os.makedirs(os.path.dirname(test_file_name), exist_ok=True)
        
        with open(test_file_name, 'w') as f:
            json.dump(user_tests, f, indent=2)
        
        print(f"\n✅ {len(user_tests)} tests saved to: {test_file_name}")
        
        # Print summary of generated data
        if user_tests:
            first_test = user_tests[0]
            print(f"\nTest data summary:")
            print(f"  Input length: {len(first_test['input'])}")
            print(f"  Output length: {len(first_test['output'])}")
            print(f"  Input type: {type(first_test['input'][0]) if first_test['input'] else 'empty'}")
            print(f"  Output type: {type(first_test['output'][0]) if first_test['output'] else 'empty'}")
    else:
        print("\n⚠️  No tests were generated")

if __name__ == "__main__":
    main()