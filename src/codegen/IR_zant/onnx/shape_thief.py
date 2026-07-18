import onnx
import onnxruntime as ort
import numpy as np
import os
from collections import defaultdict
import json
from onnx import helper, TensorProto
import argparse
from onnx import shape_inference
from onnxsim import simplify

def load_extracted_shapes(shapes_file):
    """Load the extracted shapes from JSON file"""
    try:
        with open(shapes_file, 'r') as f:
            shapes_data = json.load(f)
        print(f"✅ Loaded {len(shapes_data)} extracted shapes from {shapes_file}")
        return shapes_data
    except Exception as e:
        print(f"❌ Failed to load shapes file: {e}")
        return None

def numpy_dtype_to_onnx_type(dtype_str):
    """Convert numpy dtype string to ONNX tensor type"""
    dtype_map = {
        'uint8': TensorProto.UINT8,
        'int8': TensorProto.INT8,
        'uint16': TensorProto.UINT16,
        'int16': TensorProto.INT16,
        'uint32': TensorProto.UINT32,
        'int32': TensorProto.INT32,
        'uint64': TensorProto.UINT64,
        'int64': TensorProto.INT64,
        'float16': TensorProto.FLOAT16,
        'float32': TensorProto.FLOAT,
        'float64': TensorProto.DOUBLE,
        'bool': TensorProto.BOOL
    }
    
    # Handle variations in dtype string format
    for key, value in dtype_map.items():
        if key in dtype_str.lower():
            return value
    
    print(f"Warning: Unknown dtype {dtype_str}, defaulting to FLOAT")
    return TensorProto.FLOAT

def update_model_with_shapes(original_model_path, shapes_data, output_model_path):
    """Update the ONNX model with extracted tensor shapes"""
    
    print(f"🔄 Updating model: {original_model_path}")
    
    # Load original model
    try:
        model = onnx.load(original_model_path)
        print(f"✅ Original model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load original model: {e}")
        return False
    
    # Create a copy of the model
    updated_model = onnx.ModelProto()
    updated_model.CopyFrom(model)
    
    print(f"📊 Original model info:")
    print(f"   - IR version: {model.ir_version}")
    print(f"   - Producer: {model.producer_name} {model.producer_version}")
    print(f"   - Graph nodes: {len(model.graph.node)}")
    print(f"   - Existing value_info entries: {len(model.graph.value_info)}")
    
    # Clear existing value_info and rebuild with extracted shapes
    updated_model.graph.ClearField("value_info")
    
    # Create new value_info entries for all intermediate tensors
    added_count = 0
    updated_count = 0
    
    for tensor_name, shape_info in shapes_data.items():
        try:
            shape = shape_info['shape']
            dtype_str = shape_info['dtype']
            
            # Convert dtype to ONNX type
            onnx_type = numpy_dtype_to_onnx_type(dtype_str)
            
            # Create tensor value info
            tensor_value_info = helper.make_tensor_value_info(
                name=tensor_name,
                elem_type=onnx_type,
                shape=shape
            )
            
            # Add to value_info
            updated_model.graph.value_info.append(tensor_value_info)
            added_count += 1
            
            if added_count <= 5:  # Show first few for verification
                print(f"✅ Added: {tensor_name} -> {shape} ({dtype_str})")
            
        except Exception as e:
            print(f"⚠️ Failed to add shape for {tensor_name}: {e}")
            continue
    
    # Also update input shapes if they were in the extracted data
    for i, input_info in enumerate(updated_model.graph.input):
        if input_info.name in shapes_data:
            shape_info = shapes_data[input_info.name]
            shape = shape_info['shape']
            dtype_str = shape_info['dtype']
            onnx_type = numpy_dtype_to_onnx_type(dtype_str)
            
            # Update input shape
            updated_input = helper.make_tensor_value_info(
                name=input_info.name,
                elem_type=onnx_type,
                shape=shape
            )
            updated_model.graph.input[i].CopyFrom(updated_input)
            updated_count += 1
            print(f"✅ Updated input: {input_info.name} -> {shape}")
    
    # Update output shapes if they were in the extracted data
    for i, output_info in enumerate(updated_model.graph.output):
        if output_info.name in shapes_data:
            shape_info = shapes_data[output_info.name]
            shape = shape_info['shape']
            dtype_str = shape_info['dtype']
            onnx_type = numpy_dtype_to_onnx_type(dtype_str)
            
            # Update output shape
            updated_output = helper.make_tensor_value_info(
                name=output_info.name,
                elem_type=onnx_type,
                shape=shape
            )
            updated_model.graph.output[i].CopyFrom(updated_output)
            updated_count += 1
            print(f"✅ Updated output: {output_info.name} -> {shape}")
    
    print(f"\n📊 Shape update summary:")
    print(f"   - Added {added_count} intermediate tensor shapes")
    print(f"   - Updated {updated_count} input/output shapes")
    print(f"   - Total value_info entries: {len(updated_model.graph.value_info)}")
    
    # Validate the updated model
    try:
        onnx.checker.check_model(updated_model)
        print(f"✅ Model validation passed")
    except Exception as e:
        print(f"⚠️ Model validation warning: {e}")
        print(f"   Model may still work despite validation warnings")
    
    # Save the updated model
    try:
        onnx.save(updated_model, output_model_path)
        print(f"✅ Updated model saved to: {output_model_path}")
        
        # Verify the saved model can be loaded
        test_model = onnx.load(output_model_path)
        print(f"✅ Saved model verification passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to save updated model: {e}")
        return False

def compare_models(original_path, updated_path):
    """Compare original and updated models"""
    
    print(f"\n🔍 COMPARING MODELS")
    print(f"="*50)
    
    try:
        original = onnx.load(original_path)
        updated = onnx.load(updated_path)
        
        print(f"Original model:")
        print(f"   - Value info entries: {len(original.graph.value_info)}")
        print(f"   - Graph nodes: {len(original.graph.node)}")
        print(f"   - Inputs: {len(original.graph.input)}")
        print(f"   - Outputs: {len(original.graph.output)}")
        
        print(f"\nUpdated model:")
        print(f"   - Value info entries: {len(updated.graph.value_info)}")
        print(f"   - Graph nodes: {len(updated.graph.node)}")
        print(f"   - Inputs: {len(updated.graph.input)}")
        print(f"   - Outputs: {len(updated.graph.output)}")
        
        # Show some example shapes
        print(f"\n📐 Sample intermediate tensor shapes in updated model:")
        for i, value_info in enumerate(updated.graph.value_info[:10]):
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            type_name = TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
            print(f"   {i+1:2d}. {value_info.name[:50]:50s} : {shape} ({type_name})")
        
        if len(updated.graph.value_info) > 10:
            print(f"   ... and {len(updated.graph.value_info) - 10} more")
        
        # Check file sizes
        orig_size = os.path.getsize(original_path)
        updated_size = os.path.getsize(updated_path)
        print(f"\n📁 File sizes:")
        print(f"   - Original: {orig_size:,} bytes")
        print(f"   - Updated:  {updated_size:,} bytes")
        print(f"   - Difference: {updated_size - orig_size:+,} bytes")
        
    except Exception as e:
        print(f"❌ Failed to compare models: {e}")

def test_updated_model(model_path):
    """Test if the updated model can be loaded by ONNX Runtime"""
    
    print(f"\n🧪 TESTING UPDATED MODEL")
    print(f"="*30)
    
    try:
        import onnxruntime as ort
        
        # Try to create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model_path, sess_options)
        
        print(f"✅ ONNX Runtime can load the updated model")
        print(f"   - Providers: {sess.get_providers()}")
        
        # Show input/output info
        print(f"\n📥 Model inputs:")
        for inp in sess.get_inputs():
            print(f"   - {inp.name}: {inp.shape} ({inp.type})")
        
        print(f"\n📤 Model outputs:")
        for out in sess.get_outputs():
            print(f"   - {out.name}: {out.shape} ({out.type})")
        
        return True
        
    except ImportError:
        print(f"⚠️ ONNX Runtime not available for testing")
        return True
    except Exception as e:
        print(f"❌ ONNX Runtime failed to load model: {e}")
        return False
    
def get_runtime_input_shapes_and_types(model_path):
    """Get runtime inputs with their data types (from your working script)"""
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
        elif 'float16' in dtype_str:
            np_dtype = np.float16
        elif 'float' in dtype_str:
            np_dtype = np.float32
        elif 'double' in dtype_str:
            np_dtype = np.float64
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

def generate_random_input(shape, dtype, input_name=""):
    """Generate random input data with correct dtype (from your working script)"""
    
    if dtype == np.uint8:
        data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    elif dtype == np.int8:
        data = np.random.randint(-128, 128, size=shape, dtype=np.int8)
    elif dtype in [np.uint16, np.uint32]:
        max_val = np.iinfo(dtype).max
        data = np.random.randint(0, min(max_val, 1000), size=shape, dtype=dtype)
    elif dtype in [np.int16, np.int32]:
        max_val = min(np.iinfo(dtype).max, 1000)
        min_val = max(np.iinfo(dtype).min, -1000)
        data = np.random.randint(min_val, max_val, size=shape, dtype=dtype)
    elif dtype in [np.float16, np.float32, np.float64]:
        if "mnist" in input_name.lower() or any(dim == 28 for dim in shape):
            data = np.random.rand(*shape).astype(dtype)
        elif "image" in input_name.lower() or len(shape) == 4:
            data = np.random.rand(*shape).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
            data = np.clip(data, -3.0, 3.0)
    else:
        print(f"Warning: Unsupported dtype {dtype}, using float32")
        data = np.random.rand(*shape).astype(np.float32)
    
    return data

def extract_all_intermediate_shapes(model_path):
    """Extract intermediate tensor shapes by modifying the model to output everything"""
    
    print(f"Extracting intermediate shapes from: {model_path}")
    
    # Load the original model
    try:
        model = onnx.load(model_path)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    # Get input info using your working method
    input_info = get_runtime_input_shapes_and_types(model_path)
    if not input_info:
        print("Could not get input info")
        return None
    
    #Find all intermediate tensors
    all_tensors = set()
    for node in model.graph.node:
        all_tensors.update(node.output)
    

    input_names = {inp.name for inp in model.graph.input}
    output_names = {out.name for out in model.graph.output}
    intermediate_tensors = all_tensors - input_names - output_names
    
    print(f"Found {len(intermediate_tensors)} intermediate tensors")
    
    # Create modified model with intermediate outputs
    # Try smaller batches and more aggressive retry logic
    batch_size = 10  # Reduced batch size
    intermediate_list = list(intermediate_tensors)
    all_shapes = {}
    failed_tensors = []
    
    for batch_start in range(0, len(intermediate_list), batch_size):
        batch_end = min(batch_start + batch_size, len(intermediate_list))
        batch_tensors = intermediate_list[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(intermediate_list) + batch_size - 1)//batch_size}")
        
        # Try multiple tensor type strategies for this batch
        type_strategies = [
            'infer_from_node',  # Try to infer from producing node
            'default_uint8',    # Assume uint8 for quantized models
            'default_float',    # Assume float32
        ]
        
        batch_success = False
        
        for strategy in type_strategies:
            if batch_success:
                break
                
            print(f"   Trying strategy: {strategy}")
            
            # Create modified model for this batch
            modified_model = onnx.ModelProto()
            modified_model.CopyFrom(model)
            
            # Add batch tensors as outputs with current strategy
            batch_added = 0
            for tensor_name in batch_tensors:
                try:
                    if strategy == 'infer_from_node':
                        tensor_type = onnx.TensorProto.FLOAT  # Default
                        
                        # Look for type info in existing value_info
                        for value_info in model.graph.value_info:
                            if value_info.name == tensor_name:
                                tensor_type = value_info.type.tensor_type.elem_type
                                break
                        
                        # Infer type from producing node
                        for node in model.graph.node:
                            if tensor_name in node.output:
                                if 'quant' in node.op_type.lower() or 'qlinear' in node.op_type.lower():
                                    tensor_type = onnx.TensorProto.UINT8
                                break
                    
                    elif strategy == 'default_uint8':
                        tensor_type = onnx.TensorProto.UINT8
                    
                    else:  # default_float
                        tensor_type = onnx.TensorProto.FLOAT
                    
                    intermediate_output = onnx.helper.make_tensor_value_info(
                        tensor_name,
                        tensor_type,
                        None  # Shape will be determined at runtime
                    )
                    modified_model.graph.output.append(intermediate_output)
                    batch_added += 1
                    
                except Exception as e:
                    print(f"Could not add tensor {tensor_name}: {e}")
                    failed_tensors.append(tensor_name)
                    continue
            
            if batch_added == 0:
                continue
            
            # Save temporary model
            temp_path = f"temp_batch_{batch_start}_{strategy}.onnx"
            try:
                onnx.save(modified_model, temp_path)
            except Exception as e:
                print(f"Could not save batch model: {e}")
                continue
            
            # Run inference on batch
            try:
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                sess = ort.InferenceSession(temp_path, sess_options)
                
                # Generate inputs
                inputs = {}
                for name, info in input_info.items():
                    inputs[name] = generate_random_input(info['shape'], info['dtype'], name)
                
                # Run inference
                outputs = sess.run(None, inputs)
                
                # Extract shapes from this batch
                output_metas = sess.get_outputs()
                original_output_count = len(model.graph.output)
                
                # Skip original outputs, get only intermediate ones
                batch_extracted = 0
                for i, output_meta in enumerate(output_metas[original_output_count:], original_output_count):
                    if i < len(outputs):
                        tensor_name = output_meta.name
                        shape = list(outputs[i].shape)
                        dtype = outputs[i].dtype
                        all_shapes[tensor_name] = {
                            'shape': shape,
                            'dtype': str(dtype)
                        }
                        batch_extracted += 1
                
                print(f"   Strategy {strategy} SUCCESS: {batch_extracted} tensors extracted")
                batch_success = True
                
            except Exception as e:
                print(f"   Strategy {strategy} failed: {e}")
                # Try with zero inputs
                try:
                    inputs = {}
                    for name, info in input_info.items():
                        inputs[name] = np.zeros(info['shape'], dtype=info['dtype'])
                    outputs = sess.run(None, inputs)
                    
                    # Extract shapes (same as above)
                    output_metas = sess.get_outputs()
                    original_output_count = len(model.graph.output)
                    
                    batch_extracted = 0
                    for i, output_meta in enumerate(output_metas[original_output_count:], original_output_count):
                        if i < len(outputs):
                            tensor_name = output_meta.name
                            shape = list(outputs[i].shape)
                            dtype = outputs[i].dtype
                            all_shapes[tensor_name] = {
                                'shape': shape,
                                'dtype': str(dtype)
                            }
                            batch_extracted += 1
                    
                    print(f"   Strategy {strategy} with zero inputs SUCCESS: {batch_extracted} tensors")
                    batch_success = True
                    
                except Exception as e2:
                    print(f"   Strategy {strategy} completely failed: {e2}")
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except:
                pass
        
        if not batch_success:
            print(f"   All strategies failed for batch {batch_start//batch_size + 1}")
            failed_tensors.extend(batch_tensors)
    
    print(f"\nExtraction summary:")
    print(f"   Successfully extracted: {len(all_shapes)} tensors")
    print(f"   Failed tensors: {len(failed_tensors)}")
    
    if failed_tensors:
        print(f"   Failed tensor names (first 10):")
        for tensor in failed_tensors[:10]:
            print(f"     {tensor}")
        if len(failed_tensors) > 10:
            print(f"     ... and {len(failed_tensors) - 10} more")
    
    return all_shapes

def main():

    parser = argparse.ArgumentParser(description="Extract shapes and update ONNX model with intermediate tensor shapes")
    parser.add_argument("--model", type=str, required=True, help="Name of your model. Automatic path is datasets/models/my_model/my_model.onnx")
    args = parser.parse_args()
    
    model_name = args.model
    model_path = f"datasets/models/{model_name}/{model_name}.onnx" #datasets/models/{model_name}/
    shapes_file = f"datasets/models/{model_name}/{model_name}_shapes.json" #datasets/models/{model_name}/
    
    shapes = extract_all_intermediate_shapes(model_path)
    
    if shapes:
        print(f"\n Shape extraction complete!")
        print(f"📊 Total shapes extracted: {len(shapes)}")
        
        # Save results
        import json
        with open(shapes_file, "w") as f:
            json.dump(shapes, f, indent=2)
        print(f"💾 Results saved to:{shapes_file}")
    else:
        print(f"\n❌ Could not extract shapes")


    print(f"🚀 ONNX MODEL SHAPE UPDATER")
    print(f"="*50)
    print(f"Original model: {model_path}")
    print(f"Shapes file: {shapes_file}")

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Original model not found: {model_path}")
        return
    
    if not os.path.exists(shapes_file):
        print(f"❌ Shapes file not found: {shapes_file}")
        print(f"   Run the shape extraction script first!")
        return
    
    # Load extracted shapes
    shapes_data = load_extracted_shapes(shapes_file)
    if not shapes_data:
        return
    
    # Update model with shapes
    success = update_model_with_shapes(model_path, shapes_data, model_path)
    
    if success:
        # Compare models
        compare_models(model_path, model_path)
        
        # Test updated model
        test_success = test_updated_model(model_path)
        
        if test_success:
            print(f"\n🎉 SUCCESS!")
            print(f" Updated model created successfully: {model_path}")
            print(f" Model passes ONNX Runtime compatibility test")
            print(f"\n You can now use the updated model with proper shape information!")
        else:
            print(f"\n⚠️ Model created but may have compatibility issues")
    else:
        print(f"\n❌ Failed to create updated model")

    #for redundancy try also with the infer shape-----------------------
    
    model = onnx.load(model_path)
    inferred_model = shape_inference.infer_shapes(model)

    model_simp, check = simplify(inferred_model)

    if check:
       print("Simplified model is valid!")
       onnx.save(model_simp, model_path)
    else:
       raise RuntimeError("Something went wrong in the onnx simplifier()")

if __name__ == "__main__":
    main()