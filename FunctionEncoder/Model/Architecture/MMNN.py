import math
import torch
import torch.nn as nn

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.MLP import get_activation


class MMNN(nn.Module):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size, rank, n_layers, fixWb, learn_basis_functions, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0]
        
        output_size = output_size * n_basis if learn_basis_functions else output_size
        ranks = [input_size] + [rank] * (n_layers - 1) + [output_size]
        widths = [hidden_size] * n_layers
        
        fc_sizes = [ranks[0]]
        for j in range(n_layers):
            fc_sizes += [widths[j], ranks[j+1]]
        total_params = 0
        for j in range(len(fc_sizes) - 1):
            weight_params = fc_sizes[j] * fc_sizes[j+1]
            bias_params = fc_sizes[j+1]
            total_params += weight_params + bias_params
        
        return total_params
    
    def __init__(self, 
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=512,
                 rank: int = 36,
                 n_layers:int=6,
                 ResNet=False, 
                 fixWb=True, 
                 activation:str="relu",
                 learn_basis_functions=True):
        super(MMNN, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.n_layers = n_layers
        self.learn_basis_functions = learn_basis_functions
        self.ResNet = ResNet
        
        if not self.learn_basis_functions:
            n_basis = 1
            self.n_basis = 1
        # get inputs
        input_size = input_size[0]  # only 1D input supported for now
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]
        self.act = get_activation(activation)
        
        ranks = [input_size] + [rank] * (n_layers - 1) + [output_size]
        widths = [hidden_size] * n_layers
        
        fc_sizes = [ranks[0]]
        for j in range(n_layers):
            fc_sizes += [ widths[j], ranks[j+1] ]

        fcs = []
        for j in range(len(fc_sizes) - 1):
            fc = nn.Linear(fc_sizes[j], fc_sizes[j+1])
            # setattr(self, f"fc{j}", fc)
            fcs.append(fc)            
        self.fcs = nn.ModuleList(fcs)
        
        if fixWb:
            for j in range(len(fcs)):
                if j % 2 == 0:
                    self.fcs[j].weight.requires_grad = False
                    self.fcs[j].bias.requires_grad = False

        
        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, rank, n_layers, fixWb, learn_basis_functions)
        assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."
    
    
    
    def forward(self, x):
        assert x.shape[-1] == self.input_size[0], f"Expected input size {self.input_size[0]}, got {x.shape[-1]}"
        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        if len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)
            
            
        outs = x
        for j in range(self.n_layers):
            if self.ResNet:
                if 0 < j < self.n_layers - 1:
                    outs_id = outs.clone()
            outs = self.fcs[2*j](outs)
            outs = self.act(outs)
            outs = self.fcs[2*j+1](outs)
            
            if self.ResNet:
                if 0 < j < self.n_layers - 1:
                    if outs.shape[1] <= outs_id.shape[1]:
                        outs = outs + outs_id[:, :outs.shape[1]]
                    else:
                        outs0 = torch.zeros_like(outs)
                        outs0[:, :outs_id.shape[1]] = outs_id
                        outs = outs + outs0
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
        