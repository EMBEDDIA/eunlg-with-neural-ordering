

import torch as t

from experiments.src.vars import MERGES, EMB_DIMS


class Params:
    def __init__(self, params):
        emb_pars = {par.split('=')[0]: par.split('=')[1] for par in params.emb_pars}
        self.lang = params.test_ind_file.split('-')[0] if 'lang' not in emb_pars else emb_pars['lang']
        self.emb_type = emb_pars['type']
        self.enc = emb_pars['enc']
        self.emb_dim = EMB_DIMS[self.enc][self.lang] if self.enc in ['elmo', 'sbert'] else EMB_DIMS[self.enc]

        h, w = params.input_shape.split('x')
        self.in_height = self.emb_dim if h == 'H' else int(h)
        in_width = w if w == 'W' else int(w)
        self.in_width = 1 if self.emb_type == 'sents' else in_width

        # model params
        model_pars = {par.split('=')[0]: par.split('=')[1] for par in params.model_pars} if params.model_pars else {}
        self.model_pars = {k: v.split('+') for k, v in model_pars.items()}

        # activations
        act_fns = {p.split('=')[0]: p.split('=')[1] for p in params.act_fns} if params.act_fns else {}
        self.conv_act_fn = act_fns['conv'] if 'conv' in act_fns else None
        if not self.conv_act_fn and 'CNN' in params.model:
            self.conv_act_fn = 'relu'
        self.fc_act_fn = act_fns['fc'] if 'fc' in act_fns else 'relu'
        self.out_act_fn = act_fns['out'] if 'out' in act_fns else 'sig'

        self.merge_pars = {p.split('=')[0]: p.split('=')[1] for p in params.merge} if params.merge else {}
        self.merge_fn = MERGES[self.merge_pars['fn']] if params.merge else None
        self.merge_dim = int(self.merge_pars['dim']) if self.merge_pars and 'dim' in self.merge_pars else None
        merge_pos = int(self.merge_pars['pos']) if self.merge_pars and 'pos' in self.merge_pars else 0
        self.merge_pos = len(params.h_units) if params.h_units and merge_pos > len(params.h_units) else merge_pos
        self.att = params.att
        # 0 = before first fc, 1 = bef. 2nd etc.
        self.loss_fn = params.loss_fn


class PairCNN(t.nn.Module):

    def __init__(self, params, n_sents=None, embeddings=None):
        """
        Pairwise CNN classifier
        """
        super(PairCNN, self).__init__()

        enc, emb_dim, emb_type = params.emb_pars['enc'], params.emb_pars['dim'], params.emb_pars['type']
        w = int(params.emb_pars['len'])

        # default params
        model_pars = {'nk': ['10'], 'krn': ['Hx2'], 'pl': ['1x2'], 'str': ['1x1'], 'dil': ['1x1'],
                      'pad': ['0x0'], 'chl': ['1']}

        model_pars.update(params.model_pars)  # update default pars with given pars
        params.model_pars = model_pars

        model_pars = {k: [tuple(v.split('x')) if 'x' in v else int(v) for v in vals]
                      for k, vals in model_pars.items()}

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

        conv_act_fn = params.act_fns['conv']

        # embedding layer if embs given
        self.embed = t.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if enc == 'rand' and n_sents:
            self.embed = t.nn.Embedding(n_sents + 1, emb_dim, padding_idx=0)
        # get convnet
        conv_layers = []
        hs = [emb_dim]
        ws = [w]  # dimensions after each convolution/pooling operation
        for i in range(n_convs):
            in_channels = n_channels if i == 0 else n_kernels[i - 1]
            conv_layers += [t.nn.Conv2d(in_channels, n_kernels[i], kernel_size=kernel_shapes[i],
                                        stride=strides[i], dilation=dilations[i], padding=paddings[i])]
            conv_layers += [conv_act_fn]
            conv_layers += [t.nn.MaxPool2d(kernel_size=pool_sizes[i])]

            # compute the dimensions
            h = abs((hs[i] + 2 * paddings[i][0] - dilations[i][0] * (kernel_shapes[i][0] - 1) - 1) // strides[i][0] + 1)
            hs += [abs((h - pool_sizes[i][0]) // pool_sizes[i][0]) + 1]  # after pooling
            w = abs((ws[i] + 2 * paddings[i][1] - dilations[i][1] * (kernel_shapes[i][1] - 1) - 1) // strides[i][1] + 1)
            ws += [abs((w - pool_sizes[i][1]) // pool_sizes[i][1]) + 1]  # after pooling

        n_conv_weights = int(n_kernels[-1] * ws[-1] * hs[-1])  # num. of weights after the conv layers

        print('conv_layers: ', [conv_layers])
        self.conv_net = t.nn.Sequential(*conv_layers)
        self.fc_net = PairFC(params, in_dim=n_conv_weights, embeddings=None)

    def forward(self, left, right):

        left, right = t.transpose(left, 1, 2), t.transpose(right, 1, 2)
        left, right = t.unsqueeze(left, 1), t.unsqueeze(right, 1)

        # inputs are token indices, use Emb layer
        if self.embed:
            left, right = self.embed(left), self.embed(right)
            left, right = t.transpose(left, dim0=1, dim1=2).unsqueeze(1), t.transpose(right, 1, 2).unsqueeze(1)

        h_left = self.conv_net(left)
        h_right = self.conv_net(right)
        h_left = t.flatten(h_left, start_dim=1)
        h_right = t.flatten(h_right, start_dim=1)
        h = self.fc_net(h_left, h_right)
        return h


class PairLSTM(t.nn.Module):

    def __init__(self, params, n_sents=None, embeddings=None):
        """
        Pairwise LSTM classifier
        :param params:
        :param n_sents:
        :param embeddings:
        """

        super(PairLSTM, self).__init__()

        model_pars = {'nl': ['1'], 'h': ['100'], 'drop': ['0'], 'dirs': ['2'], 'nh': ['1'], 'a_drop': ['0'],
                      'q': ['rand']}
        model_pars.update(params.model_pars)  # update default pars with given pars
        params.model_pars = model_pars

        self.in_width = params.emb_pars['len']
        # embedding layer if embs given
        self.embed = t.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if params.emb_pars['enc'] == 'rand' and n_sents:
            self.embed = t.nn.Embedding(n_sents + 1, params.emb_pars['dim'], padding_idx=0)

        len_stack = int(model_pars['nl'][0])
        h_size = int(model_pars['h'][0])
        lstm_dropout = float(model_pars['drop'][0])
        dirs = int(model_pars['dirs'][0])
        self.h_units = params.h_units
        self.dropout = params.dropout
        bidir = True if dirs == 2 else False
        self.lstm = t.nn.LSTM(input_size=params.emb_pars['dim'], hidden_size=h_size, num_layers=len_stack,
                              batch_first=False, dropout=lstm_dropout, bidirectional=bidir)

        self.fc = PairFC(params, in_dim=h_size * dirs, embeddings=None)

    def forward(self, left, right, l_hidden, r_hidden):

        if self.embed:
            if self.in_width == 'W':
                left, right = t.nn.utils.rnn.pad_packed_sequence(left)[0], t.nn.utils.rnn.pad_packed_sequence(right)[0]
                left, right = t.transpose(left, 0, 1), t.transpose(right, 0, 1)  # transpose batch and seq_len
            left, right = self.embed(left), self.embed(right)
            left, right = t.transpose(left, 0, 1), t.transpose(right, 0, 1)
        elif self.in_width != 'W':
            left, right = t.transpose(left, 0, 1), t.transpose(right, 0, 1)

        # (seq_, batch size, dim)
        out1, l_hidden = self.lstm(left, l_hidden)
        out2, r_hidden = self.lstm(right, r_hidden)
        if all(isinstance(o, t.nn.utils.rnn.PackedSequence) for o in (out1, out2)):
            out1, out2 = t.nn.utils.rnn.pad_packed_sequence(out1)[0], t.nn.utils.rnn.pad_packed_sequence(out2)[0]

        left, right = out1[-1], out2[-1]        # take last output vector
        h = self.fc(left, right)
        return h, l_hidden, r_hidden


class PairRNN(t.nn.Module):

    def __init__(self, params, n_sents=None, embeddings=None):
        """
        Pairwise RNN
        :param params:
        :param n_sents:
        :param embeddings:
        """

        super(PairRNN, self).__init__()

        model_pars = {'nl': ['1'], 'h': ['80'], 'drop': ['0'], 'dirs': ['1'], 'nh': ['1'], 'a_drop': ['0'],
                      'q': ['rand']}
        model_pars.update(params.model_pars)  # update default pars with given pars
        params.model_pars = model_pars

        self.in_width = params.emb_pars['len']
        # embedding layer if embs given
        self.embed = t.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if params.emb_pars['enc'] == 'rand' and n_sents:
            self.embed = t.nn.Embedding(n_sents + 1, params.emb_pars['dim'], padding_idx=0)

        len_stack = int(model_pars['nl'][0])
        h_size = int(model_pars['h'][0])
        nonlin = params.act_fns['fc']
        dirs = int(model_pars['dirs'][0])
        rnn_dropout = float(model_pars['drop'][0])
        bidir = True if dirs == 2 else False
        self.rnn = t.nn.RNN(input_size=params.emb_pasr['dim'], hidden_size=h_size, num_layers=len_stack,
                            nonlinearity=nonlin, batch_first=False, dropout=rnn_dropout, bidirectional=bidir)

        self.fc = PairFC(params, in_dim=h_size * dirs, embeddings=None)

    def forward(self, left, right):

        if self.embed:
            if type(left) == t.nn.utils.rnn.PackedSequence:
                left, right = t.nn.utils.rnn.pad_packed_sequence(left)[0], t.nn.utils.rnn.pad_packed_sequence(right)[0]
                left, right = t.transpose(left, 0, 1), t.transpose(right, 0, 1)
            left, right = self.embed(left), self.embed(right)
            left, right = t.transpose(left, 0, 1), t.transpose(right, 0, 1)

        out1, _ = self.rnn(left)
        out2, _ = self.rnn(right)
        if not self.embed and self.in_width == 'W':
            out1, out2 = t.nn.utils.rnn.pad_packed_sequence(out1)[0], t.nn.utils.rnn.pad_packed_sequence(out2)[0]

        if self.attention:
            left = self.attention(out1)
            right = self.attention(out2)
        else:
            left, right = out1[-1], out2[-2]    # take last output
        h = self.fc(left, right)
        return h


class Bilinear(t.nn.Module):

    def __init__(self, params, n_sents=None, embeddings=None):
        """
        Bilinear classifier: merge two sentence embeddings with torch.nn.Bilinear
        """
        super(Bilinear, self).__init__()

        enc, input_dim = params.emb_pars['enc'], params.emb_pars['dim']
        self.loss_fn = params.loss_fn
        self.h_units = params.h_units
        fc_act_fn = params.act_fns['fc']
        self.out_act_fn = params.act_fns['out']
        dropout = params.dropout
        out_dim1 = 1 if not self.h_units else self.h_units[0]
        self.bilinear = t.nn.Bilinear(input_dim, input_dim, out_dim1)      # first layer Bilinear

        # embedding layer if embs given
        self.embed = t.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if enc == 'rand' and n_sents:
            self.embed = t.nn.Embedding(n_sents + 1, input_dim, padding_idx=0)

        if self.h_units:
            fc_layers = [fc_act_fn]
            if len(self.h_units) > 1:
                for i in range(len(self.h_units) - 1):
                    fc_layers += [t.nn.Linear(in_features=self.h_units[i], out_features=self.h_units[i + 1])]
                    fc_layers += [fc_act_fn]
            fc_layers += [t.nn.Dropout(p=dropout)]
            fc_layers += [t.nn.Linear(in_features=self.h_units[-1], out_features=1)]
            self.fc_net = t.nn.Sequential(*fc_layers)

    def forward(self, left, right):

        if self.embed:
            left, right = self.embed(left), self.embed(right)
        h = self.bilinear(left, right)
        if self.h_units:
            h = self.fc_net(h)
        if self.loss_fn != 'bce':
            h = self.out_act_fn(h)
        return h


class PairFC(t.nn.Module):

    def __init__(self, params, in_dim=None, n_sents=None, embeddings=None):
        super(PairFC, self).__init__()

        # assuming sent embs used (emb_dim x 1)
        in_dim = params.emb_pars['dim'] if not in_dim else in_dim
        h_units = params.h_units if params.h_units else [100]
        fc_act_fn = params.act_fns['fc']
        out_act_fn = params.act_fns['out']
        dropout = params.dropout

        # embedding layer if embs given
        self.embed = t.nn.Embedding.from_pretrained(embeddings, padding_idx=0) if embeddings is not None else None
        if params.emb_pars['enc'] == 'rand' and n_sents:
            self.embed = t.nn.Embedding(n_sents + 1, params.emb_pars['dim'], padding_idx=0)

        before_merge = []
        after_merge = []
        i = 0
        for i in range(params.merge['pos']):
            in_feats = in_dim if i == 0 else h_units[i - 1]
            before_merge += [t.nn.Linear(in_features=in_feats, out_features=h_units[i])]
            before_merge += [fc_act_fn]

        n_weights = in_dim if not before_merge else h_units[i]
        self.merge_fn = params.merge['fn']
        if self.merge_fn == t.nn.Bilinear:
            self.merge_fn = self.merge_fn(n_weights, n_weights, params.merge['dim'])
        else:   # merge other than Bilinear - get number of incoming features
            params.merge['dim'] = 2 * n_weights if type(self.merge_fn) == MERGES['cct'] else n_weights

        for i in range(params.merge['pos'], len(h_units)):
            in_feats = params.merge['dim'] if i == 0 else h_units[i - 1]
            after_merge += [t.nn.Linear(in_features=in_feats, out_features=h_units[i])]
            after_merge += [fc_act_fn]

        after_merge += [t.nn.Dropout(p=dropout)]
        after_merge += [t.nn.Linear(in_features=h_units[-1], out_features=1)]

        if params.loss_fn != 'bce':
            after_merge += [out_act_fn]

        self.fc1 = t.nn.Sequential(*before_merge)
        self.fc2 = t.nn.Sequential(*after_merge)

    def forward(self, left, right):

        if self.embed:
            left, right = self.embed(left), self.embed(right)
        left = t.flatten(left, start_dim=1)
        right = t.flatten(right, start_dim=1)
        if self.fc1:
            left = self.fc1(left)
            right = self.fc1(right)

        h = self.merge_fn(left, right)
        h = self.fc2(h)
        return h
