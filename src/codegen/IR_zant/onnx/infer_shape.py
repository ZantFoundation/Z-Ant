import argparse
import onnx
from onnx import shape_inference
from onnxsim import simplify


def main():
    print(f"\n __main__")
    parser = argparse.ArgumentParser(description=" Upgrade your model with all intermediate tensor's shapes")
    parser.add_argument("--model", type=str, required=True, help="Name of your model. Automatic path is datasets/models/my_model/my_model.onnx ")
    args = parser.parse_args()
    
    model_name = args.model
    model_path = f"datasets/models/{model_name}/{model_name}.onnx"

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