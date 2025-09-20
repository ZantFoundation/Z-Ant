#!/usr/bin/env python3
"""
ONNX Node Extractor

This script takes an ONNX neural network and:
1. Extracts each node as a separate ONNX model
2. For a given input, computes and saves the input/output values for each node
3. Saves everything in organized folders with JSON metadata
"""

import onnx
import onnxruntime as ort
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
import re
from onnx import numpy_helper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXNodeExtractor:
    def __init__(self, model_path: str, output_dir: str = None):
        self.model_path = Path(model_path)
        # If no output directory specified, use the same folder as the model
        if output_dir is None:
            self.output_dir = self.model_path.parent / "extracted_nodes"
        else:
            self.output_dir = Path(output_dir)
        self.model = None
        self.session = None
        self.intermediate_values = {}
        
    def sanitize_filename(self, name: str) -> str:
        """Sanitize a string to be safe for use as a filename"""
        if not name:
            return "unnamed"
        
        # Replace problematic characters with underscores _
        # This includes: / \ : * ? " < > | . -
        sanitized = re.sub(r'[/\\:*?"<>|.\-]', '_', name)
        
        # Replace multiple consecutive underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it's not empty after sanitization
        if not sanitized:
            return "unnamed"
            
        # Limit length to avoid filesystem issues
        return sanitized[:100]
    
    def load_model(self):
        """Load the ONNX model"""
        logger.info(f"Loading ONNX model from {self.model_path}")
        self.model = onnx.load(str(self.model_path))
        onnx.checker.check_model(self.model)
        
        # Create inference session
        self.session = ort.InferenceSession(str(self.model_path))
        logger.info(f"Model loaded successfully. Found {len(self.model.graph.node)} nodes")
        
    def create_output_directories(self):
        """Create organized output directory structure"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "individual_nodes").mkdir(exist_ok=True)
        (self.output_dir / "node_data").mkdir(exist_ok=True)
        
    def get_intermediate_outputs(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get intermediate outputs for all nodes in the network"""
        logger.info("Computing intermediate outputs for all nodes...")
        
        # Create a new model with all intermediate outputs
        model_with_outputs = onnx.ModelProto()
        model_with_outputs.CopyFrom(self.model)
        
        # Add all intermediate tensors as outputs
        existing_outputs = {output.name for output in model_with_outputs.graph.output}
        
        for node in model_with_outputs.graph.node:
            for output_name in node.output:
                if output_name and output_name not in existing_outputs:
                    # Find the value info for this tensor
                    value_info = None
                    for vi in model_with_outputs.graph.value_info:
                        if vi.name == output_name:
                            value_info = vi
                            break
                    
                    if value_info is None:
                        # Create a generic value info if not found
                        value_info = onnx.helper.make_tensor_value_info(
                            output_name, onnx.TensorProto.FLOAT, None
                        )
                    
                    model_with_outputs.graph.output.append(value_info)
        
        # Save temporary model and run inference
        temp_model_path = self.output_dir / "temp_model.onnx"
        onnx.save(model_with_outputs, temp_model_path)
        
        try:
            temp_session = ort.InferenceSession(str(temp_model_path))
            outputs = temp_session.run(None, input_data)
            output_names = [output.name for output in temp_session.get_outputs()]
            
            intermediate_outputs = dict(zip(output_names, outputs))
            logger.info(f"Successfully computed {len(intermediate_outputs)} intermediate outputs")
            
        except Exception as e:
            logger.warning(f"Failed to get all intermediate outputs: {e}")
            # Fallback: just get final outputs
            outputs = self.session.run(None, input_data)
            output_names = [output.name for output in self.session.get_outputs()]
            intermediate_outputs = dict(zip(output_names, outputs))
        
        finally:
            # Clean up temporary file
            if temp_model_path.exists():
                temp_model_path.unlink()
        
        return intermediate_outputs
    
    def extract_single_node(self, node_idx: int, node: onnx.NodeProto, input_data: Dict[str, np.ndarray], intermediate_outputs: Dict[str, np.ndarray]) -> Tuple[str, Dict[str, Any]]:
        """Extract a single node as an individual ONNX model"""
        logger.info(f"Extracting node {node_idx}: {node.op_type}")
        
        # Create new graph with just this node
        new_graph = onnx.helper.make_graph(
            nodes=[node],
            name=f"single_node_{node_idx}_{node.op_type}",
            inputs=[],  # Will be filled below
            outputs=[]  # Will be filled below
        )
        
        # Find input and output value infos
        input_value_infos = []
        output_value_infos = []
        
        # Get input value infos
        for input_name in node.input:
            if input_name:  # Skip empty strings
                value_info = self._find_value_info(input_name)
                if value_info:
                    input_value_infos.append(value_info)
        
        # Get output value infos
        for output_name in node.output:
            if output_name:  # Skip empty strings
                value_info = self._find_value_info(output_name)
                if value_info:
                    output_value_infos.append(value_info)
        
        # Determine which input stays as runtime input: first non-initializer
        initializer_names = {init.name for init in self.model.graph.initializer}
        kept_input_name = None
        for input_name in node.input:
            if input_name and input_name not in initializer_names:
                kept_input_name = input_name
                break

        # Update graph outputs
        new_graph.output.extend(output_value_infos)

        # Copy relevant initializers that are directly referenced
        for initializer in self.model.graph.initializer:
            if initializer.name in node.input:
                new_graph.initializer.append(initializer)

        # For each input: keep one as real input, convert others to constants
        for vi in input_value_infos:
            inp_name = vi.name
            if kept_input_name is not None and inp_name == kept_input_name:
                new_graph.input.append(vi)
                continue

            # If already provided as initializer above, skip
            if any(init.name == inp_name for init in new_graph.initializer):
                continue

            # Try to materialize as constant from intermediate outputs or provided input data
            arr = None
            if inp_name in intermediate_outputs:
                arr = intermediate_outputs[inp_name]
            elif inp_name in input_data:
                arr = input_data[inp_name]
            else:
                # Fallback: create zeros using value_info shape if available
                shape = []
                tt = vi.type.tensor_type
                if tt.shape.dim:
                    for dim in tt.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        else:
                            shape.append(1)
                else:
                    shape = [1]
                arr = np.zeros(shape, dtype=np.float32)

            try:
                tensor_proto = numpy_helper.from_array(arr.astype(np.float32, copy=False), name=inp_name)
                new_graph.initializer.append(tensor_proto)
            except Exception as e:
                logger.warning(f"Failed to bake constant for input '{inp_name}' of node {node_idx}: {e}. Leaving as graph input.")
                new_graph.input.append(vi)
         
        # Create new model
        new_model = onnx.helper.make_model(new_graph)
        new_model.opset_import.extend(self.model.opset_import)
        
        # Save the individual node model
        sanitized_node_name = self.sanitize_filename(node.name if node.name else "unnamed")
        node_filename = f"node_{node_idx:03d}_{node.op_type}_{sanitized_node_name}.onnx"
        node_path = self.output_dir / "individual_nodes" / node_filename
        
        try:
            onnx.save(new_model, node_path)
            
            # Create metadata
            metadata = {
                "node_index": node_idx,
                "op_type": node.op_type,
                "node_name": node.name if node.name else "unnamed",
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": self._extract_attributes(node),
                "model_path": str(node_path.relative_to(self.output_dir))
            }
            
            logger.info(f"Successfully extracted node {node_idx}")
            return node_filename, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract node {node_idx}: {e}")
            return None, None
    
    def _find_value_info(self, tensor_name: str) -> onnx.ValueInfoProto:
        """Find value info for a tensor by name"""
        # Check in graph inputs
        for value_info in self.model.graph.input:
            if value_info.name == tensor_name:
                return value_info
        
        # Check in graph outputs
        for value_info in self.model.graph.output:
            if value_info.name == tensor_name:
                return value_info
        
        # Check in value_info
        for value_info in self.model.graph.value_info:
            if value_info.name == tensor_name:
                return value_info
        
        # Check in initializers and create generic value info
        for initializer in self.model.graph.initializer:
            if initializer.name == tensor_name:
                return onnx.helper.make_tensor_value_info(
                    tensor_name, initializer.data_type, initializer.dims
                )
        
        # Create generic value info if not found
        return onnx.helper.make_tensor_value_info(
            tensor_name, onnx.TensorProto.FLOAT, None
        )
    
    def _extract_attributes(self, node: onnx.NodeProto) -> Dict[str, Any]:
        """Extract node attributes to a serializable format"""
        attributes = {}
        for attr in node.attribute:
            attr_value = onnx.helper.get_attribute_value(attr)
            # Convert non-serializable types to JSON-serializable formats
            if isinstance(attr_value, np.ndarray):
                attr_value = attr_value.tolist()
            elif isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8', errors='ignore')
            elif hasattr(attr_value, '__iter__') and not isinstance(attr_value, (str, bytes)):
                # Handle lists/tuples that might contain numpy arrays or bytes
                try:
                    attr_value = [
                        item.tolist() if isinstance(item, np.ndarray) 
                        else item.decode('utf-8', errors='ignore') if isinstance(item, bytes)
                        else item
                        for item in attr_value
                    ]
                except (TypeError, AttributeError):
                    # If conversion fails, convert to string
                    attr_value = str(attr_value)
            attributes[attr.name] = attr_value
        return attributes
    
    def save_node_data(self, input_data: Dict[str, np.ndarray], 
                      intermediate_outputs: Dict[str, np.ndarray]):
        """Save input/output data for each node"""
        logger.info("Saving node input/output data...")
        
        for node_idx, node in enumerate(self.model.graph.node):
            node_data = {
    
                "name": node.name if node.name else "unnamed",
                "type": "exact",
                "input": [],
                "output": [],
                "expected_class": 0,
            }
            
            # Determine the kept input (first non-initializer), and save its data
            kept_input_name = None
            initializer_names = {init.name for init in self.model.graph.initializer}
            for input_name in node.input:
                if input_name and input_name not in initializer_names:
                    kept_input_name = input_name
                    break

            if kept_input_name:
                if kept_input_name in input_data:
                    node_data["input"] = input_data[kept_input_name].flatten().tolist()
                elif kept_input_name in intermediate_outputs:
                    node_data["input"] = intermediate_outputs[kept_input_name].flatten().tolist()
                else:
                    for initializer in self.model.graph.initializer:
                        if initializer.name == kept_input_name:
                            tensor_data = onnx.numpy_helper.to_array(initializer)
                            node_data["input"] = tensor_data.flatten().tolist()
                            break
             
             
            # Collect output data as arrays
            for output_name in node.output:
                if output_name and output_name in intermediate_outputs:
                    node_data["output"] = intermediate_outputs[output_name].flatten().tolist()

            # Save node data
            sanitized_node_name = self.sanitize_filename(node.name if node.name else "unnamed")
            data_filename = f"node_{node_idx:03d}_{node.op_type}_{sanitized_node_name}_data.json"
            data_path = self.output_dir / "node_data" / data_filename
            
            with open(data_path, 'w') as f:
                json.dump([node_data], f, indent=2)
        
        logger.info(f"Saved data for {len(self.model.graph.node)} nodes")
    
    def generate_random_input(self) -> Dict[str, np.ndarray]:
        """Generate random input data based on model input specifications with correct data types"""
        input_data = {}
        
        for input_info in self.model.graph.input:
            # Skip inputs that are initializers (weights/biases)
            if any(init.name == input_info.name for init in self.model.graph.initializer):
                continue
            
            # Get shape from type info
            shape = []
            if input_info.type.tensor_type.shape.dim:
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    else:
                        # Use a default size for dynamic dimensions
                        shape.append(1)
            else:
                # Default shape if not specified
                shape = [1, 3, 224, 224]  # Common for image models
            
            # Get data type and generate appropriate data
            elem_type = input_info.type.tensor_type.elem_type
            
            if elem_type == onnx.TensorProto.FLOAT:
                data = np.random.randn(*shape).astype(np.float32)
                logger.info(f"Generated float32 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.DOUBLE:
                data = np.random.randn(*shape).astype(np.float64)
                logger.info(f"Generated float64 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.INT8:
                data = np.random.randint(-128, 128, shape).astype(np.int8)
                logger.info(f"Generated int8 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.UINT8:
                data = np.random.randint(0, 256, shape).astype(np.uint8)
                logger.info(f"Generated uint8 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.INT16:
                data = np.random.randint(-32768, 32768, shape).astype(np.int16)
                logger.info(f"Generated int16 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.UINT16:
                data = np.random.randint(0, 65536, shape).astype(np.uint16)
                logger.info(f"Generated uint16 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.INT32:
                data = np.random.randint(-1000, 1000, shape).astype(np.int32)
                logger.info(f"Generated int32 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.UINT32:
                data = np.random.randint(0, 1000, shape).astype(np.uint32)
                logger.info(f"Generated uint32 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.INT64:
                data = np.random.randint(0, 100, shape).astype(np.int64)
                logger.info(f"Generated int64 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.UINT64:
                data = np.random.randint(0, 100, shape).astype(np.uint64)
                logger.info(f"Generated uint64 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.FLOAT16:
                data = np.random.randn(*shape).astype(np.float16)
                logger.info(f"Generated float16 input '{input_info.name}' with shape {shape}")
            else:
                # Default to float32 for unknown types
                data = np.random.randn(*shape).astype(np.float32)
                logger.warning(f"Unknown data type {elem_type}, defaulting to float32 for input '{input_info.name}' with shape {shape}")
            
            input_data[input_info.name] = data
        
        return input_data
    
    def run_extraction(self, input_data: Dict[str, np.ndarray] = None):
        """Main method to run the complete extraction process"""
        self.load_model()
        self.create_output_directories()
        
        # Use provided input data or generate random data
        if input_data is None:
            input_data = self.generate_random_input()
        
        # Get intermediate outputs
        intermediate_outputs = self.get_intermediate_outputs(input_data)
        
        # Extract individual nodes
        extracted_nodes = []
        for node_idx, node in enumerate(self.model.graph.node):
            node.name = node.name if node.name else "unnamed"
            filename, metadata = self.extract_single_node(node_idx, node, input_data, intermediate_outputs)
            if metadata:
                extracted_nodes.append(metadata)
        
        # Save node input/output data
        self.save_node_data(input_data, intermediate_outputs)
        
        # Convert extracted_nodes to JSON-serializable format
        json_serializable_nodes = []
        for node_info in extracted_nodes:
            serializable_node = {}
            for key, value in node_info.items():
                if isinstance(value, bytes):
                    # Convert bytes to string or skip
                    serializable_node[key] = value.decode('utf-8', errors='ignore')
                elif isinstance(value, np.ndarray):
                    # Convert numpy arrays to lists
                    serializable_node[key] = value.tolist()
                elif hasattr(value, '__dict__'):
                    # For complex objects, convert to string representation
                    serializable_node[key] = str(value)
                else:
                    serializable_node[key] = value
            json_serializable_nodes.append(serializable_node)

        # Save summary
        summary = {
            "original_model": str(self.model_path),
            "total_nodes": len(self.model.graph.node),
            "extracted_nodes": len(extracted_nodes),
            "input_shape": {name: list(data.shape) for name, data in input_data.items()},
            "nodes": json_serializable_nodes  # Use the serializable version
        }

        summary_path = self.output_dir / "extraction_summary.json"
        # with open(summary_path, 'w') as f:
        #     json.dump(summary, f, indent=2)
        
        logger.info(f"Extraction complete! Results saved to {self.output_dir}")
        logger.info(f"- Individual node models: {len(extracted_nodes)}")
        logger.info(f"- Node data files: {len(self.model.graph.node)}")
        logger.info(f"- Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract individual nodes from ONNX neural network")
    parser.add_argument("--path", help="Path to the input ONNX model")
    parser.add_argument("-o", "--output", help="Output directory (default: same folder as model)")
    parser.add_argument("--input-data", help="Path to numpy file with input data (optional)")
    
    args = parser.parse_args()
    
    # Load custom input data if provided
    input_data = None
    if args.input_data:
        input_data = np.load(args.input_data, allow_pickle=True).item()
        logger.info(f"Loaded custom input data from {args.input_data}")
    
    # Run extraction
    extractor = ONNXNodeExtractor(args.path, args.output)
    extractor.run_extraction(input_data)


if __name__ == "__main__":
    main()