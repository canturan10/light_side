import argparse

import imageio as imageio
import numpy as np
from PIL import Image

from deepsparse import compile_model


def parse_arguments():
    """
    Parse command line arguments.

    Returns: Parsed arguments
    """
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Model path",
    )
    arg.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Path to the image file",
    )
    return arg.parse_args()


def main(args):
    """
    Main function.

    Args:
        args : Parsed arguments
    """
    img = imageio.imread(args.source)[:, :, :3]

    batch = np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype(np.float32)

    engine = compile_model(
        args.model_path,
        len(batch),
    )
    preds = engine.run([np.ascontiguousarray(batch, dtype=np.float32)])[0]

    enh_img = Image.fromarray(
        (preds.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
    )
    enh_img.show()


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
