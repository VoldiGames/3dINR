from torch.nn import Linear, Module, Sequential
import torch
import numpy as np

class SineLayer(Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, np.sqrt(6 / self.linear.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class Siren(Sequential):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, omega_0=30):
        if outermost_linear:
            final_layer = Linear(hidden_features, out_features)
            with torch.no_grad():
                final_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0, np.sqrt(6 / hidden_features) / omega_0)
        else:
            final_layer = SineLayer(hidden_features, out_features, omega_0=omega_0)

        super().__init__(SineLayer(in_features, hidden_features, is_first=True),
            *[SineLayer(hidden_features, hidden_features, omega_0=omega_0)for _ in range(hidden_layers)],
            final_layer)
