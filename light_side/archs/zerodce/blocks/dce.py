import torch
from torch import nn


class DCE_Net(nn.Module):
    def __init__(
        self,
        layers: int,
        maps: int,
        iterations: int,
    ):
        """
        Zero Dce Net

        Args:
                layers (int): number of layers
                maps (int): number of maps
                iterations (int): number of iterations
        """
        super().__init__()
        self.layers = layers
        self.maps = maps
        self.iterations = iterations

        self.relu = nn.ReLU(inplace=True)

        for i in range(1, self.layers + 1):
            if i == 1:
                setattr(
                    self,
                    f"conv{i}",
                    nn.Conv2d(3, self.maps, 3, 1, 1, bias=True),
                )
            elif i == self.layers:
                setattr(
                    self, f"conv{i}", nn.Conv2d(self.maps * 2, 24, 3, 1, 1, bias=True)
                )
            elif i >= self.layers / 2 + 1:
                setattr(
                    self,
                    f"conv{i}",
                    nn.Conv2d(self.maps * 2, self.maps, 3, 1, 1, bias=True),
                )
            else:
                setattr(
                    self,
                    f"conv{i}",
                    nn.Conv2d(self.maps, self.maps, 3, 1, 1, bias=True),
                )

    def forward(self, x: torch.Tensor):
        self.x0 = x
        j = 0
        for i in range(1, self.layers + 1):
            if i == self.layers:
                setattr(
                    self,
                    "x_r",
                    torch.tanh(
                        getattr(self, f"conv{i}")(
                            torch.cat(
                                [
                                    getattr(self, f"x{int(self.layers/2)-j}"),
                                    getattr(self, f"x{int(self.layers/2)+1+j}"),
                                ],
                                1,
                            )
                        )
                    ),
                )
            elif i >= self.layers / 2 + 1:
                setattr(
                    self,
                    f"x{i}",
                    self.relu(
                        getattr(self, f"conv{i}")(
                            torch.cat(
                                [
                                    getattr(self, f"x{int(self.layers/2)-j}"),
                                    getattr(self, f"x{int(self.layers/2)+1+j}"),
                                ],
                                1,
                            )
                        )
                    ),
                )
                j = j + 1
            else:
                setattr(
                    self,
                    f"x{i}",
                    self.relu(getattr(self, f"conv{i}")(getattr(self, f"x{i-1}"))),
                )
        x_s = torch.split(getattr(self, "x_r"), int(24 / self.iterations), dim=1)
        for xx in x_s:
            x = x + xx * (x - torch.pow(x, 2))
        return x, getattr(self, "x_r")
