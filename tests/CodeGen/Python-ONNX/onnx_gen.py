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

# Import all the operator modules
from operators.relu import generate_relu_model
from operators.sigmoid import generate_sigmoid_model
from operators.clip import generate_clip_model
from operators.add import generate_add_model
from operators.sub import generate_sub_model
from operators.div import generate_div_model
from operators.mul import generate_mul_model
from operators.conv import generate_conv_model
from operators.convRelu import generate_conv_relu_model
from operators.matmul import generate_matmul_model
from operators.maxpool import generate_maxpool_model
from operators.reshape import generate_reshape_model
from operators.quantizeLinear import generate_quantizelinear_model
from operators.batchNormalization import generate_batchnormalization_model
from operators.transpose import generate_transpose_model
from operators.softmax import generate_softmax_model
from operators.concat import generate_concat_model
from operators.squeeze import generate_squeeze_model
from operators.ceil import generate_ceil_model
from operators.tanh import generate_tanh_model
from operators.identity import generate_identity_model
from operators.neg import generate_neg_model
from operators.shape import generate_shape_model
from operators.floor import generate_floor_model
from operators.sqrt import generate_sqrt_model
from operators.gelu import generate_gelu_model
from operators.leakyrelu import generate_leakyrelu_model
from operators.reduceMean import generate_reducemean_model
from operators.constant import generate_constant_model
from operators.oneHot import generate_onehot_model
from operators.gather import generate_gather_model
from operators.elu import generate_elu_model
from operators.flatten import generate_flatten_model
from operators.pad import generate_pad_model
from operators.resize import generate_resize_model
from operators.slice import generate_slice_model
from operators.split import generate_split_model
from operators.unsqueeze import generate_unsqueeze_model
from operators.gemm import generate_gemm_model
from operators.averagePool import generate_averagepool_model
from operators.globalAveragePool import generate_globalaveragepool_model
from operators.mean import generate_mean_model
from operators.dequantizeLinear import generate_dequantizelinear_model
from operators.cast import generate_cast_model
from operators.dynamicQuantizeLinear import generate_dynamicquantizelinear_model
from operators.qLinearConv import generate_qlinearconv_model
from operators.qLinearGlobalAveragePool import generate_qlinearglobalaveragepool_model
from operators.qLinearAdd import generate_qlinearadd_model
from operators.qLinearMatMul import generate_qlinearmatmul_model
from operators.convInteger import generate_convinteger_model
from operators.padConv import generate_padconv_model
from operators.pow import generate_pow_model


def random_shape(rank, min_dim=1, max_dim=10):
    """Generates a random shape of length 'rank'."""
    return [random.randint(min_dim, max_dim) for _ in range(rank)]


def generate_fuzz_model(op_name):
    """
    Creates the initializers, nodes and outputs with random parameters for the operator op_name.
    In this case all inputs are inserted as initializers, so the input list of the
    graph is returned empty.
    """
    # Pre-generation of names for input and output
    input_names = [f"{op_name}_param_in_{i}" for i in range(10)]
    output_names = [f"{op_name}_param_out_{i}" for i in range(10)]
    
    # Dictionary mapping operator names to their generation functions
    operator_generators = {
        "Relu": generate_relu_model,
        "Sigmoid": generate_sigmoid_model,
        "Clip": generate_clip_model,
        "Add": generate_add_model,
        "Sub": generate_sub_model,
        "Div": generate_div_model,
        "Mul": generate_mul_model,
        "Conv": generate_conv_model,
        "ConvRelu": generate_conv_relu_model,
        "MatMul": generate_matmul_model,
        "MaxPool": generate_maxpool_model,
        "Reshape": generate_reshape_model,
        "QuantizeLinear": generate_quantizelinear_model,
        "BatchNormalization": generate_batchnormalization_model,
        "Transpose": generate_transpose_model,
        "Softmax": generate_softmax_model,
        "Concat": generate_concat_model,
        "Squeeze": generate_squeeze_model,
        "Ceil": generate_ceil_model,
        "Tanh": generate_tanh_model,
        "Identity": generate_identity_model,
        "Neg": generate_neg_model,
        "Shape": generate_shape_model,
        "Floor": generate_floor_model,
        "Sqrt": generate_sqrt_model,
        "Gelu": generate_gelu_model,
        "LeakyRelu": generate_leakyrelu_model,
        "ReduceMean": generate_reducemean_model,
        "Constant": generate_constant_model,
        "OneHot": generate_onehot_model,
        "Gather": generate_gather_model,
        "Elu": generate_elu_model,
        "Flatten": generate_flatten_model,
        "Pad": generate_pad_model,
        "Resize": generate_resize_model,
        "Slice": generate_slice_model,
        "Split": generate_split_model,
        "Unsqueeze": generate_unsqueeze_model,
        "Gemm": generate_gemm_model,
        "AveragePool": generate_averagepool_model,
        "GlobalAveragePool": generate_globalaveragepool_model,
        "Mean": generate_mean_model,
        "DequantizeLinear": generate_dequantizelinear_model,
        "Cast": generate_cast_model,
        "DynamicQuantizeLinear": generate_dynamicquantizelinear_model,
        "QLinearConv": generate_qlinearconv_model,
        "QLinearGlobalAveragePool": generate_qlinearglobalaveragepool_model,
        "QLinearAdd": generate_qlinearadd_model,
        "QLinearMatMul": generate_qlinearmatmul_model,
        "ConvInteger": generate_convinteger_model,
        "PadConv": generate_padconv_model,
        "Pow": generate_pow_model,
    }
    
    if op_name in operator_generators:
        return operator_generators[op_name](input_names, output_names)
    else:
        # Fallback for operators not yet implemented
        raise NotImplementedError(f"Operator {op_name} not yet implemented. Please create operators/{op_name}.py")


def generate_model(op_name, filename, model_id=0):
    """Creates and saves an ONNX model."""
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
        helper.make_opsetid("", 10),
        helper.make_opsetid("", 13),  # Standard ONNX opset
        helper.make_opsetid("", 20),
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

def random_input(shape, elem_type):
    if elem_type == TensorProto.FLOAT:
        return np.random.randn(*shape).astype(np.float32)
    elif elem_type == TensorProto.INT64:
        return np.random.randint(0, 10, size=shape, dtype=np.int64)
    elif elem_type == TensorProto.UINT8:
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)
    elif elem_type == TensorProto.INT8:
        return np.random.randint(-128, 128, size=shape, dtype=np.int8)
    elif elem_type == TensorProto.INT32:
        return np.random.randint(-1000, 1000, size=shape, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported input type: {elem_type}")
    
def run_model(filename):
    """
    Runs the ONNX model.
    Since all inputs are defined as initializers, the runtime feed will be empty.
    """
    model = onnx.load(filename)
    graph = model.graph
    # Get the names of the initializers (runtime inputs will be empty)
    initializer_names = [init.name for init in graph.initializer]
    runtime_inputs = [inp for inp in graph.input if inp.name not in initializer_names]
    
    input_data = {}
    for inp in runtime_inputs:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        elem_type = inp.type.tensor_type.elem_type
        data = random_input(shape, elem_type)
        input_data[inp.name] = data
    
    session = ort.InferenceSession(filename)

    output_names = [out.name for out in graph.output]
    outputs = session.run(output_names, input_data)
    
    # Convert numpy arrays to lists for JSON serialization
    outputs_dict = {name: output.tolist() for name, output in zip(output_names, outputs)}
    input_data_dict = {name: data.tolist() for name, data in input_data.items()}
    
    return {"inputs": input_data_dict, "outputs": outputs_dict}


def load_supported_ops(filename="tests/CodeGen/Python-ONNX/available_operations.txt"):
    """Loads the supported operations from a file or returns a default list."""
    try:
        with open(filename, "r") as file:
            ops = [line.strip() for line in file if line.strip() and not line.strip().startswith("#")]
            return ops
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default operations.")
        return [
            "Relu", "Sigmoid", "Add", "Sub", "Div", "Mul", "Clip", "Conv", "MatMul", "MaxPool",
            "Reshape", "QuantizeLinear", "BatchNormalization", "Transpose", "Softmax", "Concat", 
            "Squeeze", "Ceil", "Tanh", "Identity", "Neg", "Shape", "Floor", "Sqrt", "Gelu", 
            "LeakyRelu", "ReduceMean", "Constant", "OneHot", "Gather", "Elu", "Flatten", "Pad",
            "Resize", "Slice", "Split", "Unsqueeze", "Gemm", "AveragePool", "GlobalAveragePool",
            "Mean", "DequantizeLinear", "Cast", "DynamicQuantizeLinear", "QLinearConv", 
            "QLinearGlobalAveragePool", "QLinearAdd", "QLinearMatMul", "ConvInteger"
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
    
    if args.op == "all": 
        supported_ops = load_supported_ops() 
    else: 
        supported_ops = [args.op] 
    
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
                continue

            try:
                try:
                    data = run_model(filename)
                except Exception as e:
                    print(f"----------------------ERROR----------------------")
                    raise e

                model_info = {
                    "operation": op,
                    "model_id": i,
                    "inputs": data["inputs"],
                    "outputs": data["outputs"],
                    "metadata": metadata
                }
                
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
                print(f"#################################################")
                print(f"Error running model for {op} (ID: {i}): {e} ")
                # raise RuntimeError(f"unable to handle {op}")

    
    with open(args.metadata_file, 'w') as f:
        json.dump(all_models, f, indent=2)
    print(f"Execution data saved to {args.metadata_file}")


if __name__ == "__main__":
    main()