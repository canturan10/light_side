import argparse
import os

import imageio as imageio
import numpy as np
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


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
    # TODO: It give error when image shape is more than 224x224. Actually not error, just change the image color balance
    img = Image.fromarray(img).resize((224, 224))
    batch = np.expand_dims(np.transpose(img, (2, 0, 1)), 0).astype(np.float32)

    model = tf.saved_model.load(args.model_path)
    model.trainable = False

    input_tensor = tf.convert_to_tensor(batch)
    preds = model(**{"input": input_tensor})["output"]
    x = np.array(preds).squeeze().transpose(1, 2, 0)

    enh_img = Image.fromarray(
        (np.clip(x, 0, 1) * 255).astype(np.uint8),
        mode="RGB",
    )
    enh_img.show()


if __name__ == "__main__":
    pa = parse_arguments()
    main(pa)
