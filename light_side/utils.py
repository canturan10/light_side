from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def configure_batch(
    batch: List[torch.Tensor],
    target_size: int,
    adaptive_batch: bool = True,
) -> torch.Tensor:
    """
    Configure batch for the required size

    Args:
        batch (List[torch.Tensor]): List of torch.Tensor(C, H, W)
        target_size (int): Max dimension of the target image
        adaptive_batch (bool, optional): If true, the batch will be adaptive (target_size is the max dimension of the each image), otherwise it will use given `target_size`. Defaults to False.

    Returns:
        torch.Tensor: batched inputs as B x C x target_size x target_size
    """
    # pylint: disable=no-member

    for i, img in enumerate(batch):

        # Check adaptive batch is given
        if adaptive_batch:
            # Get max dimension of the image
            target_size: int = max(img.size(1), img.size(2))

        # Apply interpolation to the image
        img_h: int = img.size(-2)
        img_w: int = img.size(-1)

        scale_factor: float = min(target_size / img_h, target_size / img_w)

        img = F.interpolate(
            img.unsqueeze(0),
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )

        new_h: int = img.size(-2)
        new_w: int = img.size(-1)

        # Apply padding to the image
        pad_left = (target_size - new_w) // 2
        pad_right = pad_left + (target_size - new_w) % 2

        pad_top = (target_size - new_h) // 2
        pad_bottom = pad_top + (target_size - new_h) % 2

        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        batch[i] = img

    batch = torch.cat(batch, dim=0).contiguous()

    return batch


def convert_np(
    batch: List[torch.Tensor],
    preds: List[torch.Tensor],
) -> List[Dict]:
    """
    Convert the predictions to a numpy format

    Args:
        batch (List[torch.Tensor]): List of torch.Tensor
        preds (List[torch.Tensor]): List of torch.Tensor

    Returns:
        List[Dict]: List of np arrays
    """
    outputs = []
    for img, pred in zip(batch, preds):
        img = img.clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        pred = (
            pred.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )

        outputs.append({"image": img, "enhanced": pred})

    return outputs


def visualize(input: dict) -> Image:
    """
    Virtualize the image with the predictions

    Args:
        input (dict): Dictionary of the image and the predictions

    Returns:
        Image: 3 channel PIL Image that will be shown on screen
    """
    orj_img = Image.fromarray(input["image"])
    enh_img = Image.fromarray(input["enhanced"])

    new_img = Image.new(
        "RGB", (orj_img.width + enh_img.width, min(orj_img.height, enh_img.height))
    )
    new_img.paste(orj_img, (0, 0))
    new_img.paste(enh_img, (orj_img.width, 0))
    return new_img
