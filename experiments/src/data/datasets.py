

import torch


class SentPairDataset(torch.utils.data.Dataset):

    def __init__(self, sentlist, ind_pairs, params, encoder, vocab=None):

        self.inds1, self.inds2 = [p[0] for p in ind_pairs], [p[1] for p in ind_pairs]
        self.sents1 = [sentlist[i] for i in self.inds1]
        self.sents2 = [sentlist[i] for i in self.inds2]
        self.labels = torch.tensor([p[2] for p in ind_pairs], dtype=torch.float32)

        self.encoder = encoder
        self.model = params.model
        self.in_width = params.emb_pars['len']

        self.vocab = vocab
        self.load_embs = True if self.encoder.name in ['word2vec', 'fasttext', 'glove'] else False

    def __len__(self):
        return len(self.sents1)

    def __getitem__(self, i):
        if not self.load_embs:
            # if encoding at tthe time of fetching splits
            if not self.encoder:        # e.g. fine tuning SBERT
                pair = self.sents1[i], self.sents2[i]
            else:
                pair = self.encoder.encode_sent(self.sents1[i]), self.encoder.encode_sent(self.sents2[i])
        else:
            # if using look-up table and Embedding layer
            if not self.vocab:
                # give index of sent in sentlist (and emb file)
                pair = torch.LongTensor([self.inds1[i]]), torch.LongTensor([self.inds2[i]])
            else:
                pair = _to_indices(self.sents1[i], self.encoder, self.vocab), \
                       _to_indices(self.sents2[i], self.encoder, self.vocab)
        label = self.labels[i]

        if self.in_width != 'W' and self.in_width != '1':
            pair = _resize(pair[0], int(self.in_width)), _resize(pair[1], int(self.in_width))

        return pair, label


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, sentlist, iseqs, encoder, params,
                 seed=None, vocab=None):
        """
        A dataset class for predicting order of (unordered) sequences (paragraph / article).

        """
        self.sents = sentlist
        self.iseqs = iseqs  # seq of sent ids (par / art), tuples of (start, end)
        self.encoder = encoder

        self.in_width = 'W' if params.emb_pars['len'] == 'W' else int(params.emb_pars['len'])

        self.vocab = vocab
        self.load_embs = True if self.encoder.name in ['word2vec', 'fasttext', 'glove'] else False
        if seed:
            torch.manual_seed(seed)
        self.params = params

    def __len__(self):
        # number of sentence sequences
        return len(self.iseqs)

    def __getitem__(self, i):

        sents = [self.sents[si] for si in range(*self.iseqs[i])]
        order = torch.randperm(n=len(sents)).tolist()  # get shuffled order
        sents = [sents[ri] for ri in order]
        if not self.load_embs:
            # if encoding at every iteration
            if self.encoder:
                seq = [self.encoder.encode_sent(s) for s in sents]

                if self.params.model_type == 'pos':
                    seq_emb = self.encoder.encode_seq([self.sents[si] for si in range(*self.iseqs[i])])
                    # concatenate seq_emb with each token emb
                    if self.encoder.emb_type == 'tokens':
                        seq = [torch.cat([s, seq_emb.unsqueeze(0).repeat(s.size(0), 1)], dim=1) for s in seq]
                    else:
                        seq = [torch.cat([s, seq_emb], dim=0) for s in seq]
            else:
                # fine-tuning SBERT, just feed strings
                seq = sents
        else:
            # if using look-up table and Embedding layer
            if not self.vocab:
                # give index of sent in sentlist (ie sent embs)
                seq = torch.LongTensor([ind for ind in range(*self.iseqs[i])])
            else:
                seq = [_to_indices(s, self.encoder, self.vocab) for s in sents]

        if self.in_width != 'W':
            seq = [_resize(s, self.in_width) for s in seq]

        return seq, torch.tensor(order, dtype=torch.long)


class PositionDataset(torch.utils.data.Dataset):

    def __init__(self, sentlist, iseqs, encoder, params, seed=None, vocab=None):

        self.sents = sentlist
        self.iseqs = iseqs
        self.sent_inds = [snt for seq in [[i for i in range(*t)] for t in iseqs] for snt in seq]

        self.encoder = encoder

        self.in_width = params.emb_pars['len']

        self.vocab = vocab
        self.load_embs = True if self.encoder.name in ['word2vec', 'fasttext', 'glove'] else False
        if seed:
            torch.manual_seed(seed)

        self.n_quantiles = int(params.model_pars['nq'][0]) if 'nq' in params.model_pars else 10

    def __len__(self):
        # len is all sents contained in iseqs
        return len(self.sent_inds)

    def __getitem__(self, i):
        # i = sent ind
        # get the par / art to encode also
        sent_i = self.sent_inds[i]
        for seq_i, (st, end) in enumerate(self.iseqs):
            if st <= sent_i < end:
                break

        len_seq = end - st
        div = len_seq / self.n_quantiles
        # no +1 here -- add only when computing sentence score
        pos = torch.tensor((sent_i - st) // div).long()
        if pos >= self.n_quantiles:
            pos = self.n_quantiles - 1
        # encode one sent, concatenating with avg emb for sequence (par/art)
        if not self.load_embs:
            # if encoding at the time of fetching splits
            sent_emb = self.encoder.encode_sent(self.sents[sent_i])
            seq_emb = self.encoder.encode_seq([self.sents[si] for si in range(*self.iseqs[seq_i])])
            # concatenate seq_emb with each token emb
            if self.encoder.emb_type == 'tokens':
                seq_emb = seq_emb.unsqueeze(0).repeat(sent_emb.size(0), 1)
                emb = torch.cat([sent_emb, seq_emb], dim=1)
            else:
                emb = torch.cat([sent_emb, seq_emb], dim=0)
        else:
            # if using look-up table and Embedding layer
            if not self.vocab:      # using sent embs
                # give index of sent in sentlist (and emb file)
                emb = torch.LongTensor([sent_i])
            else:
                emb = _to_indices(self.sents[sent_i], self.encoder, self.vocab)

        if 1 != self.in_width and self.in_width != 'W':
            emb = _resize(emb, int(self.in_width))

        return emb, pos


def _to_indices(sent, encoder, vocab):
    # transform to embedding indices, if using a look-up file / table for embs
    toks = encoder.tokeniser.tokenize(sent) if 'bert' in encoder.name \
        else encoder.tokeniser(sent, encoder.lang)
    token_inds = [vocab.index(tok) + 1 for tok in toks]
    return torch.LongTensor(token_inds)


def _resize(emb, maxlen):
    elen, h = len(emb), emb.shape[-1]
    if elen >= maxlen:
        return emb[:maxlen]
    else:
        e = torch.zeros(maxlen, h, dtype=emb.dtype, device=emb.device)
        lp = (maxlen - elen) // 2
        rp = int(maxlen - elen - lp)
        e[lp:maxlen - rp] = emb
        return e
