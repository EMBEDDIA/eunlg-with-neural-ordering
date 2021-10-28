
import os

import torch as t
import numpy as np
import h5py
from allennlp.commands.elmo import ElmoEmbedder
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from experiments.src.vars import PROJ_DIR, EMB_DIR, EMB_MODELS, MERGES, DEVICE, MODEL_DIR
from experiments.src.data.statfi import tokenise_sent_into_words


class Encoder(object):

    def __init__(self, emb_pars):

        self.lang = emb_pars['lang']
        model_name = emb_pars['model'] if 'model' in emb_pars else None
        self.name = emb_pars['enc']     # bert / elmo / w2v etc.
        self.emb_type = emb_pars['type']
        self.elmo_dim = int(emb_pars['edim']) if 'edim' in emb_pars else 2
        self.emb_merge = MERGES[emb_pars['mfn']] if 'mfn' in emb_pars else MERGES['avg']

        self.emb_dim = emb_pars['dim']
        in_width = emb_pars['len'] if emb_pars['len'] == 'W' else int(emb_pars['len'])
        self.in_width = 1 if self.emb_type == 'sents' else in_width
        self.tokeniser = tokenise_sent_into_words

        if self.name == 'elmo':
            enc_model = EMB_MODELS[self.name][self.lang]
            model_path = os.path.join(PROJ_DIR, 'models', self.name, self.lang)
            opt_fname = 'options.json' if self.lang in ['finnish', 'swedish'] else enc_model + '_options.json'
            w_fname = '-'.join([enc_model, 'weights.hdf5']) if self.lang in ['finnish', 'swedish'] \
                else '_'.join([enc_model, 'weights.hdf5'])
            options_file = os.path.join(model_path, opt_fname)
            weight_file = os.path.join(model_path, w_fname)
            cuda_device = 0 if t.cuda.is_available() else -1
            self.model = ElmoEmbedder(options_file=options_file, weight_file=weight_file, cuda_device=cuda_device)

        elif self.name == 'sbert':
            if model_name != 'statfi':
                model_name = 'xlm-r-bert-base-nli-stsb-mean-tokens' if self.lang != 'english' else \
                    'bert-large-nli-stsb-mean-tokens'
            else:
                # see that fine-tuned model exists in the following path
                model_name = os.path.join(MODEL_DIR, 'sbert', 'statfi_' + self.lang[:2])
            self.model = SentenceTransformer(model_name)
            self.tokeniser = self.model.tokenizer
            self.model.to(DEVICE)

        elif self.name == 'bert':
            # downloading from transformers if model not in cache (not finest)
            enc_model = EMB_MODELS[self.name][self.lang]
            model_path = os.path.join(PROJ_DIR, 'models', self.name, self.lang, enc_model) \
                if self.lang == 'finest' else enc_model
            cache_dir = os.path.join(PROJ_DIR, 'models', 'cache')
            if self.lang == 'finest':
                self.tokeniser = BertTokenizer.from_pretrained(model_path)
                self.model = BertModel.from_pretrained(model_path)
            elif self.lang == 'swedish':
                self.tokeniser = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
                self.model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)
            else:
                self.tokeniser = BertTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
                self.model = BertModel.from_pretrained(model_path, cache_dir=cache_dir)
            self.model.eval()
            self.model.to(DEVICE)

        elif self.name in ['word2vec', 'glove']:
            self.model = get_w2v_embs(self.name)

        elif self.name == 'fasttext':

            self.model = get_fasttext_embs(self.lang)

    def encode_sent(self, sent):
        """

        :param sent: a list of strings (tokenised sentence)
        :return:
        """
        lang = self.lang if self.lang != 'finest' else 'finnish'
        tokens = tokenise_sent_into_words(sent, lang)
        if self.name == 'elmo':
            sent = tokens if self.emb_type == 'tokens' else [sent]
            e = t.tensor(self.model.embed_sentence(sent))
            e = e[self.elmo_dim]

        elif self.name == 'sbert':
            if self.emb_type == 'sents':
                e = self.model.encode(sent, convert_to_tensor=True, output_value='sentence_embedding',
                                      device=DEVICE)
            else:
                e = self.model.encode(sent, convert_to_tensor=True, output_value='token_embeddings',
                                      device=DEVICE)

        elif self.name == 'bert':
            # bert uses its own tokens
            inds = t.tensor(self.tokeniser.encode(sent)).unsqueeze(0)
            with t.no_grad():
                output = self.model(inds)
            e = output[0].squeeze()
            e = self.emb_merge(e).unsqueeze(0) if self.emb_type == 'sents' else e

        elif self.name in ['glove', 'word2vec', 'fasttext']:
            e = t.empty(len(tokens), self.emb_dim)
            if self.name == 'fasttext':
                for i, w in enumerate(tokens):
                    e[i, :] = t.tensor(self.model.get_word_vector(w))
            else:
                for i, w in enumerate(tokens):
                    if w in self.model:
                        vec = self.model[w]
                        vec.flags.writeable = True  # to avoid user warning
                        vec = t.tensor(vec)
                    else:
                        vec = t.empty(self.emb_dim).uniform_(-.25, .25)
                    e[i, :] = vec
            e = self.emb_merge(e).unsqueeze(0) if self.emb_type == 'sents' else e

        else:   # random vec
            elen = 1 if self.emb_type == 'sents' else len(tokens)
            e = t.empty(elen, self.emb_dim).uniform_(-.25, .25)

        return e

    def encode_seq(self, seq, merge='avg'):
        # seq is a list of strings (sents)
        lang = self.lang if self.lang != 'finest' else 'finnish'
        if self.name == 'elmo':
            if merge == 'avg':
                embs = [t.tensor(self.model.embed_sentence([s])) for s in seq]
                embs = t.stack([e[self.elmo_dim] for e in embs])
                e = t.sum(embs, dim=0) / embs.size(0)
            else:
                seq = [' '.join(seq)]
                e = t.tensor(self.model.embed_sentence(seq))[self.elmo_dim]

        elif self.name == 'sbert':
            if merge == 'avg':
                embs = t.stack([self.model.encode(sent, convert_to_tensor=True, output_value='sentence_embedding',
                                                  device=DEVICE) for sent in seq])
                e = t.sum(embs, dim=0) / embs.size(0)
            else:
                seq = [' '.join(seq)]
                e = self.model.encode(seq, convert_to_tensor=True, output_value='sentence_embedding',
                                      device=DEVICE)

        elif self.name in ['glove', 'word2vec', 'fasttext']:
            tokens = [tokenise_sent_into_words(s, lang) for s in seq]
            tokens = [tok for s in tokens for tok in s]
            e = t.empty(len(tokens), self.emb_dim)
            if self.name == 'fasttext':
                for i, w in enumerate(tokens):
                    e[i, :] = t.tensor(self.model.get_word_vector(w))
            else:
                for i, w in enumerate(tokens):
                    if w in self.model:
                        vec = self.model[w]
                        vec.flags.writeable = True  # to avoid user warning
                        vec = t.tensor(vec)
                    else:
                        vec = t.empty(self.emb_dim).uniform_(-.25, .25)
                    e[i, :] = vec
            e = t.sum(e, dim=0) / e.size(0)

        else:  # random vec
            e = t.empty(self.emb_dim).uniform_(-.25, .25)

        return e

    def encode_token(self, token):

        if self.name == 'elmo':
            e = t.tensor(self.model.embed_sentence([token])).squeeze()
            return e
        elif self.name == 'bert':       # input is a token id
            with t.no_grad():
                output = self.model(token)[0].squeeze()
            return output
        elif self.name == 'sbert':
            return self.model.encode(token, convert_to_tensor=True)

        elif self.name in ['glove', 'word2vec']:
            if token in self.model:
                vec = self.model[token]
                vec.flags.writeable = True
                vec = t.tensor(vec)
            else:
                vec = t.empty(self.emb_dim).uniform_(-.25, .25)
            return vec

        elif self.name == 'fasttext':
            # TODO: implement
            pass

        else:   # random
            return t.empty(self.emb_dim).uniform_(-.25, .25)

    def encode_and_save(self, str_set, fp):
        """
        Read all sents / tokens, encode with given model, save to hdf5 (train / test)
        :param str_set:
        :param fp:
        :param vocab:
        :return:
        """
        print('Encoding sents / vocab...')
        f = h5py.File(fp, 'w', libver='latest', swmr=True)
        shape = (len(str_set), self.emb_dim) if self.name != 'elmo' else (len(str_set), 3, self.emb_dim)
        dset = f.create_dataset('sents', shape=shape, dtype=np.float32, fillvalue=0)

        encode = self.encode_sent if self.emb_type == 'sents' else self.encode_token
        for i in range(len(str_set)):
            dset[i] = encode(str_set[i]).numpy()
        f.close()
        print('Sents / vocab encoded and written into file!')

    def load_embs(self, sentlist):
        """

        :param sentlist: simple list with all sents, given in main.py
        :param params: get fname from params
        :param test: train / test set file
        :return:
        """
        # use separate for train and test??
        print('Loading embs...')

        # split = 'test' if test else 'train'
        # fn = '-'.join([self.lang, self.name, self.emb_type, split]) + '.hdf5'
        fn = '-'.join([self.lang, self.name, self.emb_type]) + '.hdf5'
        fp = os.path.join(EMB_DIR, fn)

        # get fpath from emb params - if no file, encode sents and save to file
        if self.emb_type == 'tokens':
            vocab_fp = os.path.join(EMB_DIR, 'vocabs', fn.replace('.hdf5', '.txt'))
            if os.path.exists(vocab_fp):
                with open(vocab_fp, 'r', encoding='utf-8') as f:
                    vocab = [line.strip() for line in f]
            else:
                # return ids for BERT
                vocab = self.get_vocab_from_sents(sentlist, lang=self.lang)
                with open(vocab_fp, 'w', encoding='utf-8') as f:
                    for w in vocab:
                        f.write(w + '\n')
            if not os.path.exists(fp):
                self.encode_and_save(vocab, fp)
        else:
            vocab = None
            if not os.path.exists(fp):
                self.encode_and_save(sentlist, fp)

        f = h5py.File(fp, 'r')
        embs = f['sents'][:]
        embs = embs[:, self.elmo_dim, :] if self.name == 'elmo' else embs
        embs = t.tensor(embs)
        pad_emb = t.zeros(1, *embs.shape[1:])
        print('Embs loaded!')
        return t.cat(tensors=(pad_emb, embs), dim=0), vocab

    def get_vocab_from_sents(self, sentlist, lang):
        """
        From a list of sentences, get the unique toekns.
        :param sentlist:
        :param lang:
        :return:
        """
        vocab = []
        for sent in sentlist:
            toks = self.tokeniser.tokenize(sent) if 'bert' in self.name else self.tokeniser(sent, lang=lang)
            toks = [tok.lower() for tok in toks]
            vocab += [tok for tok in toks if tok not in vocab]
            vocab = sorted(vocab)
        return vocab


def collate_pos(data):

    # collate fn for PosNet
    sents, pos_labels = [tup[0] for tup in data], t.tensor([tup[1] for tup in data], dtype=t.long)
    emb_dim = sents[0].shape[-1] if len(sents[0].size()) > 1 else None
    dtype, dev = sents[0].dtype, sents[0].device
    bs = len(sents)
    lens = [e.shape[0] for e in sents]
    maxlen = max(lens)
    # embs1, embs2 = t.zeros(bs, max(lens1), emb_dim), t.zeros(bs, max(lens2), emb_dim)
    for i in range(bs):
        shape = (maxlen - lens[i], emb_dim) if emb_dim else (maxlen - lens[i],)
        pad = t.zeros(*shape, dtype=dtype, device=dev)
        sents[i] = t.cat((sents[i], pad), dim=0)

    embs = t.stack(sents)
    embs = t.transpose(embs, 0, 1)

    # print('embs1. embs2 shapes: ', (embs1.shape, embs2.shape))
    p = t.nn.utils.rnn.pack_padded_sequence(embs, lens, batch_first=False, enforce_sorted=False)
    return p, pos_labels


def collate_pw(data):
    """
    For DataLoader when using variable length input sequences (RNNs)
    :param tens: list of tensors, where first dim varies
    :return:
    """
    pairs, labels = [tup[0] for tup in data], t.tensor([tup[1] for tup in data], dtype=t.float32)
    s1, s2 = [tup[0] for tup in pairs], [tup[1] for tup in pairs]
    emb_dim = s1[0].shape[-1] if len(s1[0].shape) > 1 else None
    dtype = s1[0].dtype
    dev = s1[0].device
    bs = len(s1)
    lens1, lens2 = [e.shape[0] for e in s1], [e.shape[0] for e in s2]
    maxlen1, maxlen2 = max(lens1), max(lens2)
    for i in range(bs):
        shape1 = (maxlen1 - lens1[i], emb_dim) if emb_dim else (maxlen1 - lens1[i],)
        shape2 = (maxlen2 - lens2[i], emb_dim) if emb_dim else (maxlen2 - lens2[i],)
        pad1 = t.zeros(*shape1, dtype=dtype, device=dev)
        s1[i] = t.cat((s1[i], pad1), dim=0)
        pad2 = t.zeros(*shape2, dtype=dtype, device=dev)
        s2[i] = t.cat((s2[i], pad2), dim=0)

    embs1, embs2 = t.stack(s1), t.stack(s2)
    embs1, embs2 = t.transpose(embs1, 0, 1), t.transpose(embs2, 0, 1)

    p1 = t.nn.utils.rnn.pack_padded_sequence(embs1, lens1, batch_first=False, enforce_sorted=False)
    p2 = t.nn.utils.rnn.pack_padded_sequence(embs2, lens2, batch_first=False, enforce_sorted=False)

    return (p1, p2), labels


def collate_order(data):
    seqs, orders = [tup[0] for tup in data], [tup[1] for tup in data]
    dtype, dev = seqs[0][0].dtype, seqs[0][0].device
    emb_type = 'sent' if len(seqs[0][0].shape) == 1 else 'token'
    emb_dim = None if dtype == t.int64 else seqs[0][0].shape[-1]

    # seq is a list of sentences, sent has dims (n_toks, emb_dim) or (emb_dim,)
    # if token embs, pack
    if emb_type == 'token':
        for j, seq in enumerate(seqs):
            sentlens = [len(s) for s in seq]
            maxsentlen = max(sentlens)
            for i in range(len(seq)):
                snt_psh = (maxsentlen - sentlens[i], emb_dim) if emb_dim else (maxsentlen - sentlens[i],)
                pad = t.zeros(*snt_psh, dtype=dtype, device=dev)
                seq[i] = t.cat((seq[i], pad), dim=0)
            seq = t.stack(seq)
            seq = t.transpose(seq, 0, 1)
            seqs[j] = t.nn.utils.rnn.pack_padded_sequence(seq, sentlens, batch_first=False, enforce_sorted=False)
    else:
        seqs = [t.stack(seq) for seq in seqs]
    return seqs, orders


def collate_var_len_seq_pad(data):
    seqs, orders = [tup[0] for tup in data], [tup[1] for tup in data]
    # seqs is a list of lists of sent embs - has to be because of different dims
    # last dim if not using embedding indices
    dtype, dev = seqs[0][0].dtype, seqs[0][0].device
    emb_type = 'sent' if len(seqs[0][0].shape) == 1 else 'token'
    emb_dim = None if dtype == t.int64 else seqs[0][0].shape[-1]

    bs = len(seqs)
    seqlens = [len(seq) for seq in seqs]
    maxseqlen = max(seqlens)
    sentlens = [[len(sent) for sent in seq] for seq in seqs]
    maxsentlen = max([max(l) for l in sentlens])
    for i in range(bs):
        if emb_type == 'token':
            for j, sent in enumerate(seqs[i]):
                snt_psh = (maxsentlen - sentlens[i][j], emb_dim) if emb_dim else (maxsentlen - sentlens[i][j],)
                pad = t.zeros(*snt_psh, dtype=dtype, device=dev)
                seqs[i][j] = t.cat((seqs[i][j], pad), dim=0)

        seq = t.stack(seqs[i])
        seq_pdsh = (maxseqlen - seqlens[i], *seq.shape[1:])
        seq_pad = t.zeros(*seq_pdsh, dtype=dtype, device=dev)
        seqs[i] = t.cat((seq, seq_pad), dim=0)

        orders[i] = t.tensor([o + 1 for o in orders[i]] + [0] * (maxseqlen - seqlens[i]), dtype=dtype, device=dev)

    # shape is (batch, seq_len, sent_len, word_emb_dim) or (batch, seq_len, emb_dim)
    embs = t.stack(seqs)
    embs = t.transpose(embs, 0, 1)
    packed = t.nn.utils.rnn.pack_padded_sequence(embs, seqlens, batch_first=False, enforce_sorted=False)

    orders = t.stack(orders)
    # return packed, orders, seqlens
    return packed, orders


def get_w2v_embs(enc):

    from gensim.scripts.glove2word2vec import glove2word2vec
    from gensim.models.keyedvectors import KeyedVectors

    print('Loading w2v/glove embeddings...')

    if enc == 'glove':
        vec_path = os.path.join(EMB_DIR, 'glove', 'vecs.txt')
        binary = False
        if not os.path.exists(vec_path):
            fname = 'glove.840B.300d.txt' if not TESTING else 'glove.6B.300d.txt'
            glove2word2vec(glove_input_file=os.path.join(EMB_DIR, 'glove', fname), word2vec_output_file=vec_path)
    else:
        vec_path = os.path.join(EMB_DIR, 'word2vec', 'GoogleNews-vectors-negative300.bin')
        binary = True
    model = KeyedVectors.load_word2vec_format(vec_path, binary=binary)
    return model


def get_fasttext_embs(lang):

    import fasttext.util

    lang = 'sv' if lang == 'swedish' else lang[:2]
    fn = 'cc.' + lang + '.300.bin'
    fp = os.path.join(EMB_DIR, 'fasttext', fn)

    if not os.path.exists(fp):
        fasttext.util.download_model(lang, if_exists='ignore')
        os.makedirs(fp.replace(fn, ''), exist_ok=True)
        cp = os.path.join(os.getcwd(), fn)
        os.replace(cp, fp)

    model = fasttext.load_model(fn)
    return model
