import argparse

import imageio as imageio
import torch
import os
import light_side as ls
import time


def parse_arguments():
    """
    Parse command line arguments.

    Returns: Parsed arguments
    """
    arg = argparse.ArgumentParser()

    arg.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="GPU device to use",
    )
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
        "--source",
        "-s",
        type=str,
        required=True,
        help="Path to the image file or directory",
    )
    return arg.parse_args()


def main(args):
    """
    Main function.

    Args:
        args : Parsed arguments
    """
    if args.version:
        if args.version not in ls.get_model_versions(args.model_name):
            raise ValueError(
                f"model version {args.version} not available for model {args.model_name}, available versions: {ls.get_model_versions(args.model_name)}"
            )
        version = args.version
    else:
        version = ls.get_model_latest_version(args.model_name)

    model = ls.Enhancer.from_pretrained(args.model_name, version=version)
    model.eval()
    model.to(args.device)
    print(model.summarize(max_depth=1))

    if os.path.isdir(args.source):
        for file in sorted(os.listdir(args.source)):
            print(file)
            file_path = os.path.join(args.source, file)
            if os.path.isfile(file_path):
                img = imageio.imread(file_path)[:, :, :3]
                results = model.predict(img)
                pil_img = ls.utils.visualize(results[0])
                pil_img.show()
                time.sleep(1)
    else:
        if os.path.isfile(args.source):
            img = imageio.imread(args.source)[:, :, :3]
            results = model.predict(img)
            pil_img = ls.utils.visualize(results[0])
            pil_img.show()


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
