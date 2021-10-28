

import torch as t


class PtrAttention(t.nn.Module):

    def __init__(self, h_dim, att_units):
        super(PtrAttention, self).__init__()

        self.W2_dec = t.nn.Linear(h_dim, att_units, bias=False)
        self.W1_enc = t.nn.Linear(h_dim, att_units, bias=False)
        self.V = t.nn.Linear(att_units, 1, bias=False)

        self.tanh = t.nn.Tanh()
        self.softmax = t.nn.Softmax(dim=0)

        self.inf = t.nn.Parameter(t.FloatTensor([float('-inf')]), requires_grad=False)

    def forward(self, enc_out, dec_h, mask):

        # mask is list of bools
        enc_out = enc_out.squeeze(1)
        d = self.W2_dec(dec_h)
        e = self.W1_enc(enc_out)
        u = self.tanh(d + e)
        u = self.V(u)

        inf = self.inf.repeat(len(mask)).unsqueeze(1)
        if len(u[mask]) > 0:
            u[mask] = inf[mask]

        a = self.softmax(u)

        # weighted enc_out
        enc_w = a * enc_out
        out = enc_w.sum(dim=0)

        return out, u
