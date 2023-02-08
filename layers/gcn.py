import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, isBias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.proj = nn.Linear(in_ft, out_ft)
        self.act = nn.Tanh()

        if isBias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.isBias = isBias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        h = self.fc(seq)
        h = torch.spmm(adj, h)

        if self.isBias:
            h += self.bias
        h += self.proj(seq)
        h = self.act(h)
        return h
