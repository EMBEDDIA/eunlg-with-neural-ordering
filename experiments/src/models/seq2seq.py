

import random

import torch as t

from experiments.src.models.attention import PtrAttention
from experiments.src.vars import DEVICE


class LSTMEncoder(t.nn.Module):

    def __init__(self, model_pars, emb_dim):
        super(LSTMEncoder, self).__init__()

        len_stack = int(model_pars['nl'][0])
        h_size = int(model_pars['h'][0])
        lstm_dropout = float(model_pars['drop'][0])
        dirs = int(model_pars['dirs'][0])
        bidir = True if dirs == 2 else False

        self.lstm = t.nn.LSTM(input_size=emb_dim, hidden_size=h_size, num_layers=len_stack,
                              batch_first=False, dropout=lstm_dropout, bidirectional=bidir)

    def forward(self, embs, hc=None):

        # packed sequence or tensor (seq_len, batch, dim)
        outs, hc = self.lstm(embs, hc) if hc else self.lstm(embs)
        return outs, hc


class Decoder(t.nn.Module):

    def __init__(self, emb_dim, h_dim, att_units, enc_dirs, enc_nl):
        super(Decoder, self).__init__()

        self.lstm = t.nn.LSTM(emb_dim, h_dim * enc_dirs, batch_first=False)
        self.attention = PtrAttention(h_dim * enc_dirs, att_units)

        self.enc_dirs = enc_dirs
        self.enc_nl = enc_nl

    def forward(self, emb, hc, enc_out, mask):
        # mask is a list of ones and zeros

        h = hc[0][-1]

        di, u = self.attention(enc_out, h, t.eq(mask, 0))

        x = t.cat([di, emb]).view(1, 1, -1)

        _, hc = self.lstm(x, hc)
        return hc, u


class PointerNet(t.nn.Module):

    def __init__(self, params, n_sents=None, embeddings=None):
        super(PointerNet, self).__init__()

        # default params
        model_pars = {'dirs': ['2'], 'h': ['100'], 'drop': ['0'], 'nl': ['1'], 'a_drop': ['0'], 'nh': ['1'],
                      'au': ['100']}
        sent_enc_pars = {'dirs': ['2'], 'h': ['100'], 'drop': ['0'], 'nl': ['1'], 'a_drop': ['0'], 'nh': ['1'],
                         'au': ['100']}
        sent_enc_pars.update(params.sent_enc_pars) if hasattr(params, 'sent_enc_pars') else sent_enc_pars
        model_pars.update(params.model_pars)
        params.model_pars = model_pars

        emb_dim, emb_type = params.emb_pars['dim'], params.emb_pars['type']
        n_lstm_layers = int(model_pars['nl'][0])
        h_size = int(model_pars['h'][0])
        dirs = int(model_pars['dirs'][0])
        att_units = int(model_pars['au'][0])

        self.loss_fn = params.loss_fn
        if self.loss_fn != t.nn.CrossEntropyLoss():
            self.loss_fn = t.nn.CrossEntropyLoss()

        self.bidir = True if dirs == 2 else False

        # Emb layer if using loaded embs / random encodings
        self.embed = t.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if params.emb_pars['enc'] == 'rand' and n_sents:
            self.embed = t.nn.Embedding(n_sents + 1, emb_dim, padding_idx=0)

        self.sent_encoder = None
        if emb_type == 'tokens':
            self.sent_encoder = LSTMEncoder(sent_enc_pars, emb_dim)
            self.encoder = LSTMEncoder(model_pars, h_size * dirs)
            dec0_size = h_size * dirs
            dec_size = dec0_size * 2
            sent_dirs, sent_nl, sent_h = map(int, (sent_enc_pars['dirs'][0], sent_enc_pars['nl'][0],
                                                   sent_enc_pars['h'][0]))
            self.sent_h0 = t.nn.Parameter(t.zeros(sent_dirs * sent_nl, 1, sent_h), requires_grad=False)
            self.sent_c0 = t.nn.Parameter(t.zeros(sent_dirs * sent_nl, 1, sent_h), requires_grad=False)
        else:
            self.encoder = LSTMEncoder(model_pars, emb_dim)
            dec0_size = emb_dim
            dec_size = dec0_size + h_size * dirs
        self.decoder = Decoder(emb_dim=dec_size, h_dim=h_size, att_units=att_units, enc_dirs=dirs,
                               enc_nl=n_lstm_layers)
        self.d0 = t.nn.Parameter(t.FloatTensor(dec0_size), requires_grad=False)
        t.nn.init.uniform_(self.d0, -1, 1)

        self.h0 = t.nn.Parameter(t.zeros(dirs * n_lstm_layers, 1, h_size), requires_grad=False)
        self.c0 = t.nn.Parameter(t.zeros(dirs * n_lstm_layers, 1, h_size), requires_grad=False)

        self.mask = t.nn.Parameter(t.ones(1), requires_grad=False)

    def forward(self, seq, order, teacher_force_ratio=.5, sent_hc=None, hc=None):

        # only one sequence given as input - packed sequence if token embs
        if isinstance(seq, t.nn.utils.rnn.PackedSequence):
            if sent_hc is None:
                sent_hc = (self.sent_h0, self.sent_c0)
            out, sent_hc = self.sent_encoder(seq, sent_hc)
            seq = t.nn.utils.rnn.pad_packed_sequence(out)[0][-1]
        # two dims (seqlen, emb_dim) when using sent embs -> add batch dim
        seq = seq.unsqueeze(1)
        seqlen = seq.size(0)
        # out: (seq_len, batch, h_dim), hs: (num_layers, batch, h_dim)
        if hc is None:
            hc = (self.h0, self.c0)
        out, hc = self.encoder(seq, hc)
        hc = (hc[0].view(1, 1, -1), hc[1].view(1, 1, -1))

        pred_order = t.zeros(seqlen, dtype=t.long)
        loss = 0

        mask = self.mask.repeat(seqlen)     # ones of size seqlen

        # first decoder input
        dec_in = self.d0
        for i in range(seqlen):
            hc, u = self.decoder(dec_in, hc, out, mask)
            att_w = t.softmax(u, dim=0)

            # ensure same index is not chosen twice, by setting teacher force to 0
            pred = t.tensor([att_w.argmax(dim=0)], device=DEVICE)

            teacher_force = random.random() < teacher_force_ratio
            true_i = t.tensor([order[i]], dtype=t.int64, device=DEVICE)
            ptr_i = true_i if teacher_force else pred
            if teacher_force_ratio == 0:
                mask[ptr_i] = 0     # to ensure pred_order contains no duplicates
            else:
                mask[true_i] = 0

            dec_in = seq[ptr_i].squeeze()

            u = u.T
            loss_incr = self.loss_fn(u, ptr_i) if teacher_force_ratio == 0 else self.loss_fn(u, true_i)
            loss += loss_incr
            pred_order[i] = pred

        return pred_order, loss


class PositionNet(t.nn.Module):
    """
    Position classifier
    """
    def __init__(self, params, n_sents=None, embeddings=None):
        super(PositionNet, self).__init__()

        enc, emb_dim, emb_type = params.emb_pars['enc'], params.emb_pars['dim'], params.emb_pars['type']

        model_pars = {'dirs': ['2'], 'h': ['100'], 'drop': ['0'], 'nl': ['1'], 'a_drop': ['0'], 'nh': ['1'],
                      'au': ['100'], 'nq': ['10'], 'seq_emb': ['1'], 'enc': ['lstm']}
        model_pars.update(params.model_pars)

        # Emb layer if using loaded embs / random encodings
        self.embed = t.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if enc == 'rand' and n_sents:
            self.embed = t.nn.Embedding(n_sents + 1, emb_dim, padding_idx=0)

        params.model_pars = model_pars
        use_seq_emb = bool(int(model_pars['seq_emb'][0]))
        # emb_dim * 2 because token embs concatenated with doc emb
        emb_dim = emb_dim * 2 if use_seq_emb else emb_dim

        enc_mod = model_pars['enc'][0]
        self.encoder = None
        if emb_type == 'tokens':
            if enc_mod == 'lstm':
                self.encoder = LSTMEncoder(model_pars, emb_dim)
            else:
                print('Incorrect encoder or encoder not implemented')

        self.n_quantiles = int(model_pars['nq'][0])
        h_size = int(model_pars['h'][0])
        dirs = int(model_pars['dirs'][0])
        in_feats = h_size * dirs if self.encoder else emb_dim
        self.fc = t.nn.Linear(in_features=in_feats, out_features=self.n_quantiles)
        self.softmax = t.nn.Softmax(dim=1)

        self.task = params.task
        len_stack = int(model_pars['nl'][0])
        self.h0 = t.nn.Parameter(t.zeros(dirs * len_stack, 1, h_size), requires_grad=False)
        self.c0 = t.nn.Parameter(t.zeros(dirs * len_stack, 1, h_size), requires_grad=False)

    def forward(self, emb, hc=None):

        if self.encoder:
            if isinstance(self.encoder, LSTMEncoder):
                if hc is None:
                    hc = (self.h0, self.c0)
                outs, hc = self.encoder(emb, hc)
                outs = t.nn.utils.rnn.pad_packed_sequence(outs)[0]
                emb = outs[-1]
            else:
                print('Incorrect encoder given or not implemented')
        out = self.fc(emb)
        # return self.softmax(out)      # with CrossEntropyLoss, don't use softmax here
        return out
