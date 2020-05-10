import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

from Code.Models import embedded_size

"""
    source: https://github.com/galsang/BiDAF-pytorch
    author: Taeuk Kim, Ph.D. student, Seoul National University
"""


class AttentionFlow(nn.Module):

    def __init__(self, c_size, q_size):
        super().__init__()

        self.att_weight_c = Linear(c_size, 1)
        self.att_weight_q = Linear(q_size, 1)
        self.att_weight_cq = Linear(c_size, 1)

    def forward(self, c, q):
        """
        :param c: (batch, c_len, s_size)
        :param q: (batch, q_len, q_size)
        :return: (batch, c_len, q_len)
        """
        c_len = c.size(1)
        q_len = q.size(1)
        batch_size = c.size(0)

        cq = []
        for i in range(q_len):
            # (batch, 1, q_size)
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
        q2c_att = torch.bmm(b, c).squeeze().view(batch_size,-1)
        # (batch, c_len, embedding_dim) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(batch_size, c_len, -1)

        # (batch, c_len, embedding_dim * 4)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
        return x