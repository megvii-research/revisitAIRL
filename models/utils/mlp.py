import torch.nn as nn


def get_mlp(mlp_name, input_dim, hidden_dim, low_dim, norm_layer, predictor=False):
    if mlp_name == 'moco':
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, low_dim)
        )

    elif mlp_name == 'byol':
        if predictor:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim), #, bias=False),
                nn.Linear(low_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim), #, bias=False),
            )

        else:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim), #, bias=False),
            )

    elif mlp_name == 'mcv2p':
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, low_dim, bias=False),
            nn.Linear(low_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, low_dim, bias=False),
        )

    elif mlp_name == 'mcv2p_2l':
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, low_dim, bias=False),
        )

    elif mlp_name == 'cls':
        return nn.Sequential(nn.Linear(input_dim, low_dim))

    else:
        raise NotImplementedError("MLP version is wrong!")
