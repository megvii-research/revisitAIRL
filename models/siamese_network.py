from collections import OrderedDict

import torch
import torch.nn as nn

from .resnet import resnet50


class SiameseNetwork(nn.Module):
    def __init__(self, low_dim, hidden_dim, width, MLP, predictor, bn, CLS):
        super(SiameseNetwork, self).__init__()

        self.online_encoder = resnet50(low_dim=low_dim, hidden_dim=hidden_dim, width=width, MLP=MLP,
                                       predictor=predictor['online'], bn=bn['online'], CLS=CLS)
        self.target_encoder = resnet50(low_dim=low_dim, hidden_dim=hidden_dim, width=width, MLP=MLP,
                                       predictor=predictor['target'], bn=bn['target'], CLS=False)
        self.CLS = CLS
        self.momentum_update(0)

    @torch.no_grad()
    def momentum_update(self, m):
        for p1, p2 in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            p2.data = m * p2.data + (1 - m) * p1.detach().data

    @torch.no_grad()
    def partial_eval(self):
        def set_bn_train_helper(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.train()
        self.target_encoder.eval()
        self.target_encoder.apply(set_bn_train_helper)

    def forward(self, inps):
        output_dict = OrderedDict()

        for i in range(len(inps)):
            if self.CLS:
                online, gp_feat = self.online_encoder(inps[i])
            else:
                online = self.online_encoder(inps[i])
            output_dict['online{}'.format(i+1)] = online

        with torch.no_grad():
            for i in range(len(inps)):
                target = self.target_encoder(inps[i])
                output_dict['target{}'.format(i+1)] = target

        if self.CLS:
            output_dict['gp_feat'] = gp_feat
        return output_dict