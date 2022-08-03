import os
from typing import Dict, List

import torch
import torch.nn as nn
from .blocks.loss import *


class ZeroDCE(nn.Module):
    """
    Implementation of ZeroDCE: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
    The network is the one described in arxiv.org/abs/2001.06826v2 .
    """

    # pylint: disable=no-member

    __CONFIGS__ = {
        "default": {
            "input": {
                "input_size": 256,
                "normalized_input": True,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            },
        },
    }

    def __init__(
        self,
        config: Dict,
    ):
        super().__init__()
        self.config = config

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(32 * 2, 32, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(32 * 2, 32, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(32 * 2, 24, 3, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.color_loss = ColorConstancyLoss()
        self.spatial_consistency_loss = SpatialConsistancyLoss()
        self.exposure_loss = ExposureLoss(patch_size=16, mean_val=0.6)
        self.illumination_smoothing_loss = IlluminationSmoothnessLoss()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r

    def logits_to_preds(self, logits: List[torch.Tensor]):
        """
        Convert logits to predictions.
        """
        enhance_image_1, _, _ = logits
        return enhance_image_1

    @classmethod
    def build(
        cls,
        config: str = "",
        **kwargs,
    ) -> nn.Module:
        """
        Build the model with random weights.

        Args:
            config (str, optional): Configuration name. Defaults to "".
            labels (List[str], optional): List of labels. Defaults to None.

        Returns:
            nn.Module: Model with random weights.
        """
        # return model with random weight initialization

        return cls(
            config=ZeroDCE.__CONFIGS__[config],
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: str,
        *args,
        **kwargs,
    ) -> nn.Module:
        """
        Load a model from a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model
            config (str): Configuration of the model

        Returns:
            nn.Module: Model with pretrained weights
        """

        *_, full_model_name, _ = model_path.split(os.path.sep)

        s_dict = torch.load(
            os.path.join(model_path, f"{full_model_name}.pth"), map_location="cpu"
        )

        model = cls(
            config=ZeroDCE.__CONFIGS__[config],
            *args,
            **kwargs,
        )

        # model.load_state_dict(s_dict["state_dict"], strict=True)
        model.load_state_dict(s_dict, strict=False)

        return model

    def compute_loss(
        self,
        logits: List[torch.Tensor],
        targets: List,
        hparams: Dict,
    ):
        """
        Compute the loss for the model.

        Args:
            logits (List[torch.Tensor]): _description_
            targets (List): _description_
            hparams (Dict, optional): _description_. Defaults to {}.

        Raises:
            ValueError: Unknown criterion

        Returns:  Loss
        """
        _, enhanced_image, A = logits
        loss_tv = 200 * self.illumination_smoothing_loss(A)
        loss_spa = torch.mean(self.spatial_consistency_loss(enhanced_image, targets))
        loss_col = 5 * torch.mean(self.color_loss(enhanced_image))
        loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image))
        loss = loss_tv + loss_spa + loss_col + loss_exp

        return {
            "loss": loss,
            "loss_tv": loss_tv,
            "loss_spa": loss_spa,
            "loss_col": loss_col,
            "loss_exp": loss_exp,
        }

    def configure_optimizers(self, hparams: Dict):
        """
        Configure optimizers for the model.

        Args:
            hparams (Dict): Hyperparameters

        Raises:
            ValueError: Unknown optimizer
            ValueError: Unknown scheduler

        Returns: optimizers and scheduler
        """
        hparams_optimizer = hparams.get("optimizer", "sgd")
        if hparams_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=hparams.get("learning_rate", 1e-1),
                momentum=hparams.get("momentum", 0.9),
                weight_decay=hparams.get("weight_decay", 1e-5),
            )
        elif hparams_optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=hparams.get("learning_rate", 1e-1),
                # betas=hparams.get("betas", (0.9, 0.999)),
                # eps=hparams.get("eps", 1e-08),
                weight_decay=hparams.get("weight_decay", 1e-5),
            )
        elif hparams_optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=hparams.get("learning_rate", 1e-1),
                betas=hparams.get("betas", (0.9, 0.999)),
                eps=hparams.get("eps", 1e-08),
                weight_decay=hparams.get("weight_decay", 1e-5),
            )
        else:
            raise ValueError("Unknown optimizer")

        hparams_scheduler = hparams.get("scheduler", "steplr")
        if hparams_scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hparams.get("step_size", 4),
                gamma=hparams.get("gamma", 0.5),
            )
        elif hparams_scheduler == "multisteplr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                gamma=hparams.get("gamma", 0.5),
                milestones=hparams.get("milestones", [500000, 1000000, 1500000]),
            )
        else:
            raise ValueError("Unknown scheduler")

        return [optimizer], [scheduler]
