import torch.nn as nn
import torch
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self, model_type, *params):
        super().__init__()
        assert model_type.lower() in ['cnn', 'fc', 'mlp'], "Model '{}' not supported".format(model_type.lower())
        if model_type == 'cnn':
            self.model = CNN(*params)
        else:
            self.model = FullyConnected(*params)

    def forward(self, x):
        return self.model(x)

    def get_sv_stats(self, mode=None):
        stats = dict()
        for name, module in list(self.model.model.named_modules())[1:]:
            if "act" not in name and 'flat' not in name and 'drop' not in name:
                params = list(module.parameters())[0].detach().cpu()

                U, S, V = torch.svd(params, compute_uv=True)
                stats[name] = dict()
                stats[name]['max'] = S.max(axis=-1)[0]
                stats[name]['min'] = S.min(axis=-1)[0]
                stats[name]['mean'] = S.mean().item()

                if "conv" in name:
                    stats[name]['std_dev_max'] = stats[name]['max'].std().item()
                    stats[name]['std_dev_min'] = stats[name]['min'].std().item()
                    stats[name]['max'] = stats[name]['max'].mean().item()
                    stats[name]['min'] = stats[name]['min'].mean().item()
                else:
                    stats[name]['max'] = stats[name]['max'].item()
                    stats[name]['min'] = stats[name]['min'].item()

                if mode == 'spectral':
                    sd = self.model.model[name].state_dict()
                    if "conv" in name:
                        sd['weight'] = params/stats[name]['max'].unsqueeze(2).unsqueeze(3)
                    else:
                        sd['weight'] = params / stats[name]['max']
                    self.model.model[name].load_state_dict(sd)

                elif mode == 'thresh':
                    S = S * (S < 0.1)   # Set lower than threshold singular values to 0
                    S = torch.diag_embed(S)
                    new_layer = torch.matmul(torch.matmul(U, S), V.transpose(-1,-2)) # Reconstruct Layer
                    sd = self.model.model[name].state_dict()
                    sd['weight'] = new_layer
                    self.model.model[name].load_state_dict(sd)
        return stats


class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_dim, n_classes, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        assert activation in [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU], "Activation Function not supported!"
        n_inputs = 1
        for e in input_size:
            n_inputs *= e
        self.model = nn.ModuleDict(OrderedDict([
            ('fc1', nn.Linear(n_inputs, hidden_dim)),
            ('drop1', nn.Dropout(dropout)),
            ('act1', activation()),
            ('fc2', nn.Linear(hidden_dim, hidden_dim)),
            ('drop2', nn.Dropout(dropout)),
            ('act2', activation()),
            ('cls', nn.Linear(hidden_dim, n_classes))
        ]))

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1)
        for key, module in self.model.items():
            x = module(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_classes, activation=nn.ReLU, dropout=0.0):
        super().__init__()
        assert activation in [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU], "Activation Function not supported!"
        in_channels, h, w = input_size
        o1 = int((h - 3 + 2)/2) + 1     # (H - FilterSize + 2*Padding)/Stride + 1
        o2 = int((o1 - 3 + 2)/2) + 1    # (H - FilterSize + 2*Padding)/Stride + 1
        n_inputs = 1
        for e in input_size:
            n_inputs *= e
        self.model = nn.ModuleDict(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, hidden_dim, 3, 2, padding=1)),
            ('drop1', nn.Dropout(dropout)),
            ('act1', activation()),
            ('conv2', nn.Conv2d(hidden_dim, hidden_dim, 3, 2, padding=1)),
            ('drop2', nn.Dropout(dropout)),
            ('act2', activation()),
            ('flat1', nn.Flatten()),
            ('drop3', nn.Dropout(dropout)),
            ('cls', nn.Linear(o2*o2*hidden_dim, n_classes))
        ]))

    def forward(self, x):
        for key, module in self.model.items():
            x = module(x)
        return x


if __name__ == '__main__':
    model = Model('cnn', (1, 28, 28), 100, 20)
    model.get_sv_stats()