
import random

import torch

# functions for merging embeddings
MERGES = {'bln': torch.nn.Bilinear, 'abs': lambda x, y: torch.abs(x - y), 'sqd': lambda x, y: (x - y)**2,
          'cct': lambda x, y: torch.cat((x, y), dim=0), 'avg': lambda x: torch.mean(x, dim=0)}


class PairwiseLSTM(torch.nn.Module):

    def __init__(self, params, n_embs=None, embeddings=None):
        super(PairwiseLSTM, self).__init__()

        enc, emb_dim = params.emb_pars['enc'], params.emb_pars['dim']
        model_pars = {'nl': ['1'], 'h': ['100'], 'drop': ['0'], 'dirs': ['2'], 'nh': ['1'], 'a_drop': ['0'],
                      'q': ['rand']}
        model_pars.update(params.model_pars)  # update default pars with given pars

        # embedding layer if embs given
        self.embed = torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if enc == 'rand' and n_embs:
            self.embed = torch.nn.Embedding(n_embs + 1, emb_dim, padding_idx=0)

        len_stack = int(model_pars['nl'][0])
        h_size = int(model_pars['h'][0])
        lstm_dropout = float(model_pars['drop'][0])
        dirs = int(model_pars['dirs'][0])
        bidir = True if dirs == 2 else False
        self.lstm = torch.nn.LSTM(input_size=emb_dim, hidden_size=h_size, num_layers=len_stack, batch_first=False,
                                  dropout=lstm_dropout, bidirectional=bidir)

        self.fc = PairFC(params, in_dim=h_size * dirs, embeddings=None)

    def forward(self, left, right):

        if type(left) != torch.nn.utils.rnn.PackedSequence:
            if len(left.size()) < 3:
                left, right = left.unsqueeze(1), right.unsqueeze(1)

        if self.embed and type(left) == torch.nn.utils.rnn.PackedSequence:
            left, right = torch.nn.utils.rnn.pad_packed_sequence(left)[0], torch.nn.utils.rnn.pad_packed_sequence(right)[0]
            left, right = torch.transpose(left, 0, 1), torch.transpose(right, 0, 1)  # transpose batch and seq_len
            left, right = self.embed(left), self.embed(right)
            left, right = torch.transpose(left, 0, 1), torch.transpose(right, 0, 1)

        # (seq_, batch size, dim)
        out1, _ = self.lstm(left)
        out2, _ = self.lstm(right)
        if type(out1) == torch.nn.utils.rnn.PackedSequence:
            out1, out2 = torch.nn.utils.rnn.pad_packed_sequence(out1)[0], \
                         torch.nn.utils.rnn.pad_packed_sequence(out2)[0]

        left, right = out1[-1], out2[-1]        # take last output vector
        h = self.fc(left, right)
        return h


class PairwiseCNN(torch.nn.Module):

    def __init__(self, params, n_embs=None, embeddings=None):
        # super().__init__()
        super(PairwiseCNN, self).__init__()

        enc, emb_dim = params.emb_pars['enc'], params.emb_pars['dim']
        h, w = params.emb_pars['dim'], int(params.emb_pars['len'])
        model_pars = {'nk': ['10'], 'krn': ['Hx2'], 'pl': ['1x2'], 'str': ['1x1'], 'dil': ['1x1'],
                      'pad': ['0x0'], 'chl': ['1']}

        model_pars.update(params.model_pars)  # update default pars with given pars
        model_pars = {k: [tuple(v.split('x')) if 'x' in v else int(v) for v in vals]
                      for k, vals in model_pars.items()}

        # pars required: in_height, in_width, emb_pars (enc), emb_dim, model_pars, conv_act_fn

        n_convs = len(model_pars['nk'])
        n_channels = int(model_pars['chl'][0])
        n_kernels = list(map(int, model_pars['nk']))
        kernel_shapes = [(emb_dim, int(tup[1])) if tup[0] == 'H' else tuple(map(int, tup))
                         for tup in model_pars['krn']]
        strides = [tuple(map(int, tup)) for tup in model_pars['str']]
        paddings = [tuple(map(int, tup)) for tup in model_pars['pad']]
        dilations = [tuple(map(int, tup)) for tup in model_pars['dil']]
        pool_sizes = [tuple(map(int, tup)) for tup in model_pars['pl']]
        del model_pars['chl']      # remove to enable assert
        assert all(len(model_pars[k]) == n_convs for k in model_pars.keys())  # check equal amount of pars given

        conv_act_fn = params.act_fns['conv'] if 'conv' in params.act_fns else torch.nn.ReLU()

        # embedding layer if embs given
        self.embed = torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if enc == 'rand' and n_embs:
            self.embed = torch.nn.Embedding(n_embs + 1, emb_dim, padding_idx=0)
        # get convnet
        conv_layers = []
        hs = [h]
        ws = [w]  # dimensions after each convolution/pooling operation
        for i in range(n_convs):
            in_channels = n_channels if i == 0 else n_kernels[i - 1]
            conv_layers += [torch.nn.Conv2d(in_channels, n_kernels[i], kernel_size=kernel_shapes[i],
                                            stride=strides[i], dilation=dilations[i], padding=paddings[i])]
            conv_layers += [conv_act_fn]
            conv_layers += [torch.nn.MaxPool2d(kernel_size=pool_sizes[i])]

            # compute the dimensions
            h = abs((hs[i] + 2 * paddings[i][0] - dilations[i][0] * (kernel_shapes[i][0] - 1) - 1) // strides[i][0] + 1)
            hs += [abs((h - pool_sizes[i][0]) // pool_sizes[i][0]) + 1]  # after pooling
            w = abs((ws[i] + 2 * paddings[i][1] - dilations[i][1] * (kernel_shapes[i][1] - 1) - 1) // strides[i][1] + 1)
            ws += [abs((w - pool_sizes[i][1]) // pool_sizes[i][1]) + 1]  # after pooling

        n_conv_weights = int(n_kernels[-1] * ws[-1] * hs[-1])  # num. of weights after the conv layers

        self.conv_net = torch.nn.Sequential(*conv_layers)
        self.fc_net = PairFC(params, in_dim=n_conv_weights, embeddings=None)

    def forward(self, left, right):

        left, right = torch.transpose(left, 0, 1), torch.transpose(right, 0, 1)
        left, right = left.unsqueeze(0).unsqueeze(0), right.unsqueeze(0).unsqueeze(0)

        # inputs are sentence indices, use Emb layer
        if self.embed:
            left, right = self.embed(left), self.embed(right)
            left, right = torch.transpose(left, dim0=1, dim1=2).unsqueeze(1), torch.transpose(right, 1, 2).unsqueeze(1)

        h_left = self.conv_net(left)
        h_right = self.conv_net(right)
        h_left = torch.flatten(h_left, start_dim=1)
        h_right = torch.flatten(h_right, start_dim=1)
        h = self.fc_net(h_left, h_right)
        return h


class PairFC(torch.nn.Module):

    def __init__(self, params, in_dim=None, n_embs=None, embeddings=None):
        super(PairFC, self).__init__()

        # assuming sent embs used (emb_dim x 1)
        in_dim = params.emb_pars['dim'] * params.emb_pars['len'] if not in_dim else in_dim
        h_units = params.h_units if params.h_units else [100]
        fc_act_fn = params.act_fns['fc']
        out_act_fn = params.act_fns['out']
        dropout = params.dropout

        # pars required: in_height, in_width, fc_act_fn, out_act_fn, emb_pars (enc), emb_dim, merge_pars,
        # merge_pos, merge_dim, h_units, dropout, loss_fn

        # embedding layer if embs given
        self.embed = torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if params.emb_pars['enc'] == 'rand' and n_embs:
            self.embed = torch.nn.Embedding(n_embs + 1, params.emb_dim, padding_idx=0)

        before_merge = []
        after_merge = []
        i = 0
        for i in range(params.merge['pos']):
            in_feats = in_dim if i == 0 else h_units[i - 1]
            before_merge += [torch.nn.Linear(in_features=in_feats, out_features=h_units[i])]
            before_merge += [fc_act_fn]

        n_weights = in_dim if not before_merge else h_units[i]
        self.merge_fn = params.merge['fn']
        if self.merge_fn == torch.nn.Bilinear:
            self.merge_fn = self.merge_fn(n_weights, n_weights, params.merge['dim'])
        else:   # merge other than Bilinear - get number of incoming features
            params.merge['dim'] = 2 * n_weights if type(self.merge_fn) == MERGES['cct'] else n_weights

        for i in range(params.merge['pos'], len(h_units)):
            in_feats = params.merge['dim'] if i == 0 else h_units[i - 1]
            after_merge += [torch.nn.Linear(in_features=in_feats, out_features=h_units[i])]
            after_merge += [fc_act_fn]

        after_merge += [torch.nn.Dropout(p=dropout)]
        after_merge += [torch.nn.Linear(in_features=h_units[-1], out_features=1)]
        after_merge += [out_act_fn]

        self.fc1 = torch.nn.Sequential(*before_merge)
        self.fc2 = torch.nn.Sequential(*after_merge)

    def forward(self, left, right):

        if self.embed:
            left, right = self.embed(left), self.embed(right)
        left = torch.flatten(left, start_dim=1)
        right = torch.flatten(right, start_dim=1)
        if self.fc1:
            left = self.fc1(left)
            right = self.fc1(right)

        h = self.merge_fn(left, right)
        h = self.fc2(h)
        return h


class Bilinear(torch.nn.Module):

    def __init__(self, params, n_embs=None, embeddings=None):
        """
        :param input_dim: dimension of sentence embedding
        """
        super(Bilinear, self).__init__()

        enc, input_dim = params.emb_pars['enc'], params.emb_pars['dim']
        self.loss_fn = params.loss_fn
        self.h_units = params.h_units
        fc_act_fn = params.act_fns['fc']
        self.out_act_fn = params.act_fns['out']
        dropout = params.dropout
        out_dim1 = 1 if not self.h_units else self.h_units[0]
        self.bilinear = torch.nn.Bilinear(input_dim, input_dim, out_dim1)      # first layer Bilinear

        # embedding layer if embs given
        self.embed = torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if enc == 'rand' and n_embs:
            self.embed = torch.nn.Embedding(n_embs + 1, input_dim, padding_idx=0)

        if self.h_units:
            fc_layers = [fc_act_fn]
            if len(self.h_units) > 1:
                for i in range(len(self.h_units) - 1):
                    fc_layers += [torch.nn.Linear(in_features=self.h_units[i], out_features=self.h_units[i + 1])]
                    fc_layers += [fc_act_fn]
            fc_layers += [torch.nn.Dropout(p=dropout)]
            fc_layers += [torch.nn.Linear(in_features=self.h_units[-1], out_features=1)]
            self.fc_net = torch.nn.Sequential(*fc_layers)

    def forward(self, left, right):

        if self.embed:
            left, right = self.embed(left), self.embed(right)
        h = self.bilinear(left, right)
        if self.h_units:
            h = self.fc_net(h)
        h = self.out_act_fn(h)
        return h


class LSTMEncoder(torch.nn.Module):

    def __init__(self, model_pars, emb_dim):
        super(LSTMEncoder, self).__init__()

        # params required: model_pars, emb_dim

        len_stack = int(model_pars['nl'][0])
        h_size = int(model_pars['h'][0])
        lstm_dropout = float(model_pars['drop'][0])
        dirs = int(model_pars['dirs'][0])
        bidir = True if dirs == 2 else False

        self.lstm = torch.nn.LSTM(input_size=emb_dim, hidden_size=h_size, num_layers=len_stack,
                                  batch_first=False, dropout=lstm_dropout, bidirectional=bidir)

    def forward(self, embs, hc=None):

        # packed sequence or tensor (seq_len, batch, dim)
        outs, hc = self.lstm(embs, hc) if hc else self.lstm(embs)
        return outs, hc


class Decoder(torch.nn.Module):
    """
    Based on github.com/shirgur/PointerNet/blob/master/PointerNetorch.py
    """
    def __init__(self, emb_dim, h_dim, att_units, enc_dirs, enc_nl):
        super(Decoder, self).__init__()

        self.lstm = torch.nn.LSTM(emb_dim, h_dim * enc_dirs, batch_first=False)
        self.attention = PtrAttention(h_dim * enc_dirs, att_units)

        self.enc_dirs = enc_dirs
        self.enc_nl = enc_nl
        # self.W_dh = torch.nn.Linear(n_embs, 4 * h_dim)
        # self.W_hh = torch.nn.Linear(h_dim, 4 * h_dim)
        # self.W_ho = torch.nn.Linear(h_dim * 2, h_dim)
        # self.att = PtrAttention(h_dim, h_dim)
        # Used for propagating .cuda() command

        # self.runner = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, emb, hc, enc_out, mask):
        # mask is a list of ones and zeros

        h = hc[0][-1]

        di, u = self.attention(enc_out, h, torch.eq(mask, 0))
        # di.size is (h_size,), att_w is (seq_len,)

        x = torch.cat([di, emb]).view(1, 1, -1)

        _, hc = self.lstm(x, hc)
        return hc, u


class PointerNet(torch.nn.Module):
    """
    Based on github.com/shirgur/PointerNet/blob/master/PointerNetorch.py
    """

    def __init__(self, params, loss_fn=None, n_embs=None, embeddings=None):
        super(PointerNet, self).__init__()

        enc, emb_dim, emb_type = params.emb_pars['enc'], params.emb_pars['dim'], params.emb_pars['type']

        n_lstm_layers = int(params.model_pars['nl'][0])
        h_size = int(params.model_pars['h'][0])
        dirs = int(params.model_pars['dirs'][0])
        att_units = int(params.model_pars['au'][0])

        self.loss_fn = torch.nn.CrossEntropyLoss() if not loss_fn else loss_fn
        self.bidir = True if dirs == 2 else False

        # Emb layer if using loaded embs / random encodings
        self.embed = torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if enc == 'rand' and n_embs:
            self.embed = torch.nn.Embedding(n_embs + 1, emb_dim, padding_idx=0)

        sent_encoder = LSTMEncoder

        self.W = torch.nn.Linear(in_features=2, out_features=emb_dim)
        self.encoder = sent_encoder(params.model_pars, emb_dim)
        self.decoder = Decoder(emb_dim=emb_dim + h_size * dirs, h_dim=h_size, att_units=att_units, enc_dirs=dirs,
                               enc_nl=n_lstm_layers)
        self.d0 = torch.nn.Parameter(torch.FloatTensor(emb_dim), requires_grad=False)
        torch.nn.init.uniform_(self.d0, -1, 1)
        self.h0 = torch.nn.Parameter(torch.zeros(dirs * n_lstm_layers, 1, h_size), requires_grad=False)
        self.c0 = torch.nn.Parameter(torch.zeros(dirs * n_lstm_layers, 1, h_size), requires_grad=False)

        self.mask = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, seq, order, teacher_force_ratio=.5, device=None):

        # only one sequence given as input - packed sequence if token embs
        if len(seq.size()) < 3:
            # two dims (seqlen, emb_dim) when using sent embs -> add batch dim
            seq = seq.unsqueeze(1)
            seqlen = seq.size(0)
        else:
            # if using token embs and PackedSequence -> get seq embs for pointer
            out, _ = self.encoder(seq, (self.h0, self.c0))
            seq = torch.nn.utils.rnn.pad_packed_sequence(out)[0][-1]
            seqlen = out.size(1)

        # out: (seq_len, batch, h_dim), hs: (num_layers, batch, h_dim)
        out, hc = self.encoder(seq, (self.h0, self.c0))
        hc = (hc[0].view(1, 1, -1), hc[1].view(1, 1, -1))

        pred_order = torch.zeros(seqlen, dtype=torch.long)
        loss = 0

        mask = self.mask.repeat(seqlen)     # ones of size seqlen

        # first decoder input
        dec_in = self.d0
        for i in range(seqlen):
            hc, u = self.decoder(dec_in, hc, out, mask)

            att_w = torch.softmax(u, dim=0)

            # ensure same index is not chosen twice, by setting teacher force to 0
            pred = torch.tensor([att_w.argmax(dim=0)])

            teacher_force = random.random() < teacher_force_ratio
            true_i = torch.tensor([order[i]], dtype=torch.int64, device=device)
            ptr_i = true_i if teacher_force else pred
            if teacher_force_ratio == 0:
                mask[ptr_i] = 0
            else:
                mask[true_i] = 0

            dec_in = seq[ptr_i].squeeze()

            u = u.T
            loss += self.loss_fn(u, true_i)
            pred_order[i] = pred

        return pred_order, loss


class PtrAttention(torch.nn.Module):
    """
    Based on github.com/shirgur/PointerNet/blob/master/PointerNetorch.py
    """

    def __init__(self, h_dim, att_units):
        super(PtrAttention, self).__init__()

        self.W2_dec = torch.nn.Linear(h_dim, att_units, bias=False)
        self.W1_enc = torch.nn.Linear(h_dim, att_units, bias=False)
        self.V = torch.nn.Linear(att_units, 1, bias=False)

        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=0)

        self.inf = torch.nn.Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)

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


class PositionNet(torch.nn.Module):
    """
    PPD - classification of sentences into news article quantiles.
    """
    def __init__(self, params, n_embs=None):
        super(PositionNet, self).__init__()

        # params required: model_pars

        enc, emb_dim, emb_type = params.emb_pars['enc'], params.emb_pars['dim'], params.emb_pars['type']

        # Emb layer if using loaded embs / random encodings
        if enc == 'rand' and n_embs:
            self.embed = torch.nn.Embedding(n_embs + 1, emb_dim, padding_idx=0)

        use_seq_emb = bool(int(params.model_pars['seq_emb'][0])) if 'seq_emb' in params.model_pars else True
        # emb_dim * 2 because token embs concatenated with doc emb
        emb_dim = emb_dim * 2 if use_seq_emb else emb_dim
        enc_mod = params.model_pars['enc'][0] if 'enc' in params.model_pars else 'lstm'
        self.encoder = None
        if emb_type == 'tokens':
            if enc_mod == 'lstm':
                self.encoder = LSTMEncoder(params.model_pars, emb_dim) if emb_type == 'tokens' else None
            else:
                print('Incorrect encoder or encoder not implemented')

        n_quantiles = int(params.model_pars['nq'][0])
        h_size = int(params.model_pars['h'][0])
        dirs = int(params.model_pars['dirs'][0])

        in_feats = h_size * dirs if self.encoder else emb_dim
        self.fc = torch.nn.Linear(in_features=in_feats, out_features=n_quantiles)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, emb):

        if self.encoder:
            if isinstance(self.encoder, LSTMEncoder):
                if len(emb.size()) < 3:
                    emb = emb.unsqueeze(1)        # add batch dim
                outs, _ = self.encoder(emb)
                emb = outs[-1]
            else:
                pass
                # TODO: adapt for other encoders

        out = self.fc(emb)
        return self.softmax(out)
