'''
code adapted from the efficient KAN repo, found here: https://github.com/Blealtan/efficient-kan.git
'''

import torch
import torch.nn.functional as F
import math

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.MLP import get_activation

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU(),
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
    
class KAN(torch.nn.Module):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=10, n_layers=3, learn_basis_functions=True, grid_size=5, spline_order=3, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]
        if hidden_size is None:
            hidden_size = 2*input_size+1
        # grid size 5 and spline order 3
        if n_layers == 1:
            n_params = (input_size * output_size * 2 + input_size * output_size * (grid_size+spline_order))
        elif n_layers == 2:
            n_params = (input_size * hidden_size * 2 + input_size * hidden_size * (grid_size+spline_order)) + (hidden_size * output_size * 2 + hidden_size * output_size * (grid_size+spline_order))
        else:
            n_params =  (input_size * hidden_size * 2 + input_size * hidden_size * (grid_size+spline_order))  + \
                        (hidden_size * hidden_size * 2 + hidden_size * hidden_size * (grid_size+spline_order))  * (n_layers - 2)  + \
                        (hidden_size * output_size * 2 + hidden_size * output_size * (grid_size+spline_order))
        return n_params
    
    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=10,
                 n_layers:int=3,
                 activation:str="silu",
                 learn_basis_functions=True,
                 grid_size=5,
                 spline_order=3,
                 scale_noise=0.1,
                 scale_base=1.0,
                 scale_spline=1.0,
                 grid_eps=0.02,
                 grid_range=[-1, 1]):
        super(KAN, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.learn_basis_functions = learn_basis_functions

        if not self.learn_basis_functions:
            n_basis = 1
            self.n_basis = 1

        # get inputs
        input_size = input_size[0]  # only 1D input supported for now
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]

        act = get_activation(activation)
        # build net
        layers = []
        if n_layers == 1:
            layers.append(KANLinear(input_size, output_size, base_activation=act, grid_size=grid_size, spline_order=spline_order,
                            scale_noise=scale_noise,
                            scale_base=scale_base,
                            scale_spline=scale_spline,
                            grid_eps=grid_eps,
                            grid_range=grid_range,))
        elif n_layers == 2:
            layers.append(KANLinear(input_size, hidden_size, base_activation=act, grid_size=grid_size, spline_order=spline_order,
                            scale_noise=scale_noise,
                            scale_base=scale_base,
                            scale_spline=scale_spline,
                            grid_eps=grid_eps,
                            grid_range=grid_range,))
            layers.append(KANLinear(hidden_size, output_size, base_activation=act, grid_size=grid_size, spline_order=spline_order,
                            scale_noise=scale_noise,
                            scale_base=scale_base,
                            scale_spline=scale_spline,
                            grid_eps=grid_eps,
                            grid_range=grid_range,))
        else:
            layers.append(KANLinear(input_size, hidden_size, base_activation=act, grid_size=grid_size, spline_order=spline_order,
                            scale_noise=scale_noise,
                            scale_base=scale_base,
                            scale_spline=scale_spline,
                            grid_eps=grid_eps,
                            grid_range=grid_range,))
            for i in range(1, n_layers-1):
                layers.append(KANLinear(hidden_size, hidden_size, base_activation=act, grid_size=grid_size, spline_order=spline_order,
                                scale_noise=scale_noise,
                                scale_base=scale_base,
                                scale_spline=scale_spline,
                                grid_eps=grid_eps,
                                grid_range=grid_range,))
            layers.append(KANLinear(hidden_size, output_size, base_activation=act, grid_size=grid_size, spline_order=spline_order,
                            scale_noise=scale_noise,
                            scale_base=scale_base,
                            scale_spline=scale_spline,
                            grid_eps=grid_eps,
                            grid_range=grid_range,))
        self.model = torch.nn.Sequential(*layers)
        # verify number of parameters
        n_params = 0
        for i in range(n_layers):
            n_params += sum(p.numel() for p in layers[i].parameters() if p.requires_grad)
        estimated_n_params = self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers, learn_basis_functions=learn_basis_functions, grid_size=grid_size, spline_order=spline_order)
        assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."

    def forward(self, x, update_grid=False):
        assert x.shape[-1] == self.input_size[0], f"Expected input size {self.input_size[0]}, got {x.shape[-1]}"
        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        if len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)
        
        # this is the main part of this function. The rest is just error handling
        outs = x
        if update_grid:
            for layer in self.model:
                layer.update_grid(outs)
                outs = layer(outs)
        else:
            outs = self.model(outs)

        if self.learn_basis_functions:
            Gs = outs.view(*x.shape[:2], *self.output_size, self.n_basis)
        else:
            Gs = outs.view(*x.shape[:2], *self.output_size)

        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
        return Gs
    

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )