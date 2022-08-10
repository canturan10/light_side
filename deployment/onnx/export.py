import os
import tempfile
import torch
import argparse
import onnx

import torch

import light_side as ls


def parse_arguments():
    """
    Parse command line arguments.

    Returns: Parsed arguments
    """
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--model_name",
        type=str,
        default=ls.available_models()[0],
        choices=ls.available_models(),
        help="Model architecture",
    )
    arg.add_argument(
        "--version",
        type=str,
        help="Model version",
    )
    arg.add_argument(
        "--target",
        "-t",
        type=str,
        help="Target path to save the model",
    )
    arg.add_argument(
        "--quantize",
        "-q",
        action="store_true",
        help="Quantize the model",
    )
    arg.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="Onnx opset version",
    )
    return arg.parse_args()


def main(args):
    """
    Main function.

    Args:
        args : Parsed arguments
    """
    # pylint: disable=no-member
    if args.version:
        if args.version not in ls.get_model_versions(args.model_name):
            raise ValueError(
                f"model version {args.version} not available for model {args.model_name}, available versions: {ls.get_model_versions(args.model_name)}"
            )
        version = args.version
    else:
        version = ls.get_model_latest_version(args.model_name)

    model = ls.Enhancer.from_pretrained(
        args.model_name,
        version=version,
    )
    model.eval()

    if args.target:
        target_path = args.target
    else:
        target_path = os.path.join(
            ls.core._get_model_dir(),
            args.model_name,
            f"v{version}",
        )

    print(f"Target Path: {target_path}")

    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: "batch_size", 2: "width", 3: "height"},
        "output": {0: "batch_size", 2: "width", 3: "height"},
    }

    input_sample = torch.rand(1, 3, model.input_size, model.input_size)

    if args.quantize:

        try:
            from onnxruntime.quantization import quantize_qat
        except ImportError:
            raise AssertionError("run `pip install onnxruntime`")

        target_model_path = os.path.join(
            target_path,
            "{}_quantize.onnx".format(args.model_name),
        )
        with tempfile.NamedTemporaryFile(suffix=".onnx") as temp:
            model.to_onnx(
                temp.name,
                input_sample=input_sample,
                opset_version=args.opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                export_params=True,
            )
            quantize_qat(temp.name, target_model_path)
    else:
        target_model_path = os.path.join(
            target_path,
            "{}.onnx".format(args.model_name),
        )
        model.to_onnx(
            target_model_path,
            input_sample=input_sample,
            opset_version=args.opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )

    print("Model saved")


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
