import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

from Code.GNN_Playground.Models import embedded_size

"""
    source: https://github.com/galsang/BiDAF-pytorch
"""


class AttentionFlow(nn.Module):

    def __init__(self):
        super().__init__()

        self.att_weight_c = Linear(embedded_size, 1)
        self.att_weight_q = Linear(embedded_size, 1)
        self.att_weight_cq = Linear(embedded_size, 1)

    def forward(self, c, q):
        """
        :param c: (batch, c_len, embedding_dim)
        :param q: (batch, q_len, embedding_dim)
        :return: (batch, c_len, q_len)
        """
        c_len = c.size(1)
        q_len = q.size(1)

        # (batch, c_len, q_len, embedding_dim)
        # c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
        # (batch, c_len, q_len, embedding_dim)
        # q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
        # (batch, c_len, q_len, embedding_dim)
        # cq_tiled = c_tiled * q_tiled
        # cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

        cq = []
        for i in range(q_len):
            # (batch, 1, embedding_dim)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)

        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, embedding_dim) -> (batch, c_len, embedding_dim)
        c2q_att = torch.bmm(a, q)
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, c_len) * (batch, c_len, embedding_dim) -> (batch, embedding_dim)
        q2c_att = torch.bmm(b, c).squeeze()
        # (batch, c_len, embedding_dim) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # (batch, c_len, embedding_dim * 4)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        return x