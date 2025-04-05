import argparse
import onnx
from onnx import shape_inference

def main():
    print(f"\n __main__")
    parser = argparse.ArgumentParser(description=" Upgrade your model with all intermediate tensor's shapes")
    parser.add_argument("--path", type=str, required=True, help="Path of your model.")
    args = parser.parse_args()
    
    model = onnx.load(args.path)

    inferred_model = shape_inference.infer_shapes(model)

    onnx.save(inferred_model, args.path)
    
if __name__ == "__main__":
    main()