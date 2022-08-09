import os
from re import A
from typing import Dict, List

import torch
import torch.nn as nn
from .blocks.loss import *
from .blocks.dce import DCE_Net


class ZeroDCE(nn.Module):
    """
    Implementation of ZeroDCE: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
    The network is the one described in arxiv.org/abs/2001.061626v2 .
    """

    # pylint: disable=no-member

    __CONFIGS__ = {
        "3-32-16": {
            "input": {
                "input_size": 224,
                "normalized_input": True,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            },
            "model": {
                "layers": 3,
                "maps": 32,
                "iterations": 8,
            },
        },
        "7-16-8": {
            "input": {
                "input_size": 224,
                "normalized_input": True,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            },
            "model": {
                "layers": 7,
                "maps": 16,
                "iterations": 8,
            },
        },
        "7-32-1": {
            "input": {
                "input_size": 224,
                "normalized_input": True,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            },
            "model": {
                "layers": 7,
                "maps": 32,
                "iterations": 1,
            },
        },
        "7-32-8": {
            "input": {
                "input_size": 224,
                "normalized_input": True,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            },
            "model": {
                "layers": 7,
                "maps": 32,
                "iterations": 8,
            },
        },
        "7-32-16": {
            "input": {
                "input_size": 224,
                "normalized_input": True,
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            },
            "model": {
                "layers": 7,
                "maps": 32,
                "iterations": 16,
            },
        },
    }

    def __init__(
        self,
        config: Dict,
    ):
        super().__init__()
        self.config = config

        self.layers = self.config["model"]["layers"]
        self.maps = self.config["model"]["maps"]
        self.iterations = self.config["model"]["iterations"]

        self.backbone = DCE_Net(
            layers=self.layers,
            maps=self.maps,
            iterations=self.iterations,
        )

        self.color_loss = ColorConstancyLoss()
        self.spatial_consistency_loss = SpatialConsistancyLoss()
        self.exposure_loss = ExposureLoss(patch_size=16, mean_val=0.6)
        self.illumination_smoothing_loss = IlluminationSmoothnessLoss()

    def forward(self, x):
        return self.backbone(x)

    def logits_to_preds(self, logits: List[torch.Tensor]):
        """
        Convert logits to predictions.
        """
        x, _ = logits
        return x

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

        model.load_state_dict(s_dict["state_dict"], strict=True)

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
        enhanced_image, le_curve = logits
        loss_tv = 200 * self.illumination_smoothing_loss(le_curve)
        loss_spa = 0.5 * torch.mean(
            self.spatial_consistency_loss(enhanced_image, targets)
        )
        loss_col = 5 * torch.mean(self.color_loss(enhanced_image))
        loss_exp = 10 * torch.mean(self.exposure_loss(enhanced_image))
        loss = loss_tv + loss_spa + loss_col + loss_exp

        return {
            "loss": loss,
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
                # eps=hparams.get("eps", 1e-016),
                weight_decay=hparams.get("weight_decay", 1e-5),
            )
        elif hparams_optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=hparams.get("learning_rate", 1e-1),
                betas=hparams.get("betas", (0.9, 0.999)),
                eps=hparams.get("eps", 1e-016),
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
        elif hparams_scheduler == "reduce":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, mode="min", factor=0.97
            )
        else:
            raise ValueError("Unknown scheduler")

        return [optimizer], [scheduler]
