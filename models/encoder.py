import torch
import torch.nn as nn
from layers import GCN


class Encoder(nn.Module):
    def __init__(self, ft_size, hid_units, n_adj=1):
        super(Encoder, self).__init__()
        self.n_adj = n_adj
        if self.n_adj == 1:
            self.gcn = GCN(ft_size, hid_units)
        else:
            self.gcn_list = nn.ModuleList([GCN(ft_size, hid_units) for i in range(n_adj)])

    def forward(self, embs, adj, adj_idx=0):
        if self.n_adj == 1:
            h = self.gcn(embs, adj)
        else:
            h = self.gcn_list[adj_idx](embs, adj)
        return h


class Attention(nn.Module):
    def __init__(self, hid_units, n_adj):
        super(Attention, self).__init__()
        # self.att_score = nn.Linear(hid_units*2, 1)
        self.att_score = nn.Linear(hid_units, 1)
        self.proj = nn.Linear(hid_units, hid_units)
        self.proj2 = nn.Linear(hid_units, hid_units)
        # self.layer_embs = nn.Parameter(torch.FloatTensor(n_adj, hid_units))

        # torch.nn.init.normal(self.layer_embs.data)

        self.layer_embs = nn.Embedding(n_adj, hid_units)
        self.layer_idx = torch.unsqueeze(torch.arange(0, n_adj), dim=-1)

        self.proj_cat = nn.Linear(hid_units*2, hid_units)

    # def forward(self, embs):
    #     h = self.proj(embs)
    #     layer_idx = self.layer_idx.repeat(1, h.shape[1]).to(h.device)
    #     layer_embs = self.layer_embs(layer_idx)
    #
    #     h = torch.cat([h, layer_embs], dim=-1)
    #     score = self.att_score(h)
    #     score = torch.softmax(score, dim=0)
    #     h = score * embs
    #     h = torch.sum(h, dim=0)
    #     return h

    def forward(self, embs):
        # h = self.proj(embs)
        h = embs
        score = self.att_score(h)
        # score = torch.tanh(score)
        score = torch.softmax(score, dim=0)
        # embs = self.proj2(embs)
        # embs = torch.tanh(embs)
        h = score * embs
        h = torch.sum(h, dim=0)
        return h

    # concat
    # def forward(self, embs):
    #     h = torch.transpose(embs, 0, 1)  # [n_nodes, n_layer, dim]
    #     h = torch.reshape(h, [h.shape[0], -1])
    #     h = self.proj_cat(h)
    #     h = torch.tanh(h)
    #     return h


# class SelfAttention(nn.Module):
#     def __init__(self, hid_units):
#