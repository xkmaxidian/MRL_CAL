import torch.nn as nn
from torch.nn.functional import normalize
import torch
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        return self.discriminator(x)
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2000, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.discriminators = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
            self.discriminators.append(Discriminator(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.discriminators = nn.ModuleList(self.discriminators)

        self.high_feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.cluster_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view
    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        xfakes = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.high_feature_contrastive_module(z), dim=1)
            q = self.cluster_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)

        for v in range(self.view):
            common_z = torch.zeros((zs[0].shape[0], zs[0].shape[1]))
            for i in range(self.view):
                if i == v:
                    continue
                common_z = common_z + zs[i]
            xfake = self.decoders[v](common_z)
            xfakes.append(xfake)
        d = []
        d_ = []
        for i in range(self.view):
            d.append(self.discriminators[i](xs[i]))
            d_.append(self.discriminators[i](xfakes[i]))
        return hs, qs, xrs, zs, xfakes, d, d_

    def forward_plot(self, xs):
        zs = []
        hs = []
        xfakes = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.high_feature_contrastive_module(z)
            hs.append(h)
        for v in range(self.view):
            common_z = np.zeros((zs[0].shape[0], zs[0].shape[1]))
            for i in range(self.view):
                if i == v:
                    continue
                common_z = common_z + zs[i]
            xfake = self.decoders[v](common_z)
            xfakes.append(xfake)
        return zs, hs, xfakes

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.cluster_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds