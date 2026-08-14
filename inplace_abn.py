"""
Pure-PyTorch compatibility replacement for inplace_abn.
Eliminates the CUDA extension build requirement.
For inference, this is functionally equivalent to the original.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ABN(nn.Module):
    """
    Activated Batch Normalization — pure PyTorch fallback.
    Matches the interface of inplace_abn.ABN exactly.
    """
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        activation="leaky_relu",
        activation_param=1e-2,
    ):
        super(ABN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.activation = activation
        self.activation_param = activation_param

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )
        if self.activation == "relu":
            return F.relu(x, inplace=True)
        elif self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == "elu":
            return F.elu(x, alpha=self.activation_param, inplace=True)
        elif self.activation == "identity":
            return x
        else:
            raise RuntimeError(f"Unknown activation: {self.activation}")

    def extra_repr(self):
        rep = f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, " \
              f"affine={self.affine}, activation={self.activation}"
        if self.activation_param != 1e-2:
            rep += f", activation_param={self.activation_param}"
        return rep


class InPlaceABN(ABN):
    """
    In-place ABN — for inference, behaves identically to ABN.
    The name is kept so isinstance() checks in existing code still work.
    """
    pass
