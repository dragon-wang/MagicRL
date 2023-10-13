from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hidden_activation=nn.ReLU, output_activation=nn.Identity):
        super(MLP, self).__init__()
        sizes = [input_dim] + hidden_size + [output_dim]
        layers = []
        for j in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[j], sizes[j + 1]), hidden_activation()]
        layers += [nn.Linear(sizes[-2], sizes[-1]), output_activation()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)