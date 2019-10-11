######################################################


######################################################

import torch
import torch.nn as nn


class HighwayNetworkModule(nn.Module):
    """ """
    def __init__(self, input_size, activation_function=nn.functional.relu, gate_activation=nn.functional.sigmoid, gate_bias=-1):
        super(HighwayNetworkModule, self).__init__()

        # Activation for H(x)
        self.activation_function = activation_function
        # Activation for T(x)
        self.gate_activation = gate_activation

        # H & T layers
        self.normal_layer = nn.Linear(input_size, input_size)
        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):
        # H(x)*T(x) + x*(1 - T(x))
        # H(x)
        normal_layer_result = self.activation_function(self.normal_layer(x))
        # T(x)
        gate_layer_result = self.gate_activation(self.gate_layer(x))
        # H(x)*T(x)
        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        # x*(1 - T(x))
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)
        # H(x)*T(x) + x*(1 - T(x))
        return torch.add(multiplyed_gate_and_normal, multiplyed_gate_and_input)


class HighwayNetwork(nn.Module):
    """ """
    def __init__(self, input_size, output_size, highway_size):
        super(HighwayNetwork, self).__init__()
        self.highway_layers = nn.ModuleList( [ HighwayNetworkModule(input_size) for _ in range(highway_size) ] )
        # The highway ends with a linear layer activated by softmax
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        for current_layer in self.highway_layers:
            x = current_layer(x)
        x = F.softmax(self.linear(x))
        return x