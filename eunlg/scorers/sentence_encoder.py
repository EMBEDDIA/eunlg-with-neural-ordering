

import os

import torch

from sentence_transformers import SentenceTransformer
from allennlp.commands.elmo import ElmoEmbedder

# choose path as desired
PROJ_DIR = '~/embeddia'
MODEL_DIR = os.path.join(PROJ_DIR, 'models/trained')


class SentenceEncoder:

    def __init__(self, emb_pars, device):
        super(SentenceEncoder, self).__init__()

        self.emb_type = emb_pars['type']
        self.name = emb_pars['enc']
        self.elmo_dim = int(emb_pars['edim']) if 'edim' in emb_pars else 2
        model_name = emb_pars['model'] if 'model' in emb_pars else None

        self.emb_dim = emb_pars['dim']
        in_width = 'W' if emb_pars['len'] == 'W' else int(emb_pars['len'])
        self.in_width = 1 if self.emb_type == 'sents' else in_width
        self.lang = 'finnish' if emb_pars['lang'][:2] == 'fi' else 'english'     # lang is fi / en / en-head etc
        self.device = device

        if self.name == 'sbert':
            if model_name == 'statfi':
                model_name = PROJ_DIR + 'models/trained/sbert/statfi_' + self.lang[:2]
            else:
                # change to use another S-BERT model
                model_name = 'xlm-r-bert-base-nli-stsb-mean-tokens' if self.lang != 'english' else \
                    'bert-large-nli-stsb-mean-tokens'

            self.model = SentenceTransformer(model_name)
            self.tokeniser = self.model.tokenizer
            self.model.to(device)

        elif self.name == 'elmo':

            model_path = os.path.join(PROJ_DIR, 'models', self.name, self.lang)
            opt_fname = 'options.json' if self.lang == 'finnish' else model_name + '_options.json'
            w_fname = '-'.join([model_name, 'weights.hdf5']) if self.lang == 'finnish' \
                else '_'.join([model_name, 'weights.hdf5'])
            options_file = os.path.join(model_path, opt_fname)
            weight_file = os.path.join(model_path, w_fname)
            cuda_device = 0 if torch.cuda.is_available() else -1
            self.model = ElmoEmbedder(options_file=options_file, weight_file=weight_file, cuda_device=cuda_device)

        elif self.name in ['word2vec', 'glove']:
            self.model = get_w2v_embs(self.name)
        elif self.name == 'fasttext':
            self.model = get_fasttext_embs(self.lang)
        else:
            print('Incorrect encoder name given')

    def encode_msg(self, msg_str, seq=None, merge='avg'):
        """
        :param msg_str: a list of tokens
        :return:
        """
        if self.name == 'sbert':
            snt = ' '.join(msg_str)
            out_val = 'sentence_embedding' if self.emb_type == 'sents' else 'token_embeddings'
            e = self.model.encode(snt, convert_to_tensor=True, output_value=out_val, device=self.device)
        elif self.name == 'elmo':
            snt = [' '.join(msg_str)] if self.emb_type == 'sents' else msg_str
            e = torch.tensor(self.model.embed_sentence(snt))[self.elmo_dim]
        elif self.name in ['glove', 'word2vec', 'fasttext']:
            e = torch.empty(len(msg_str), self.emb_dim)
            if self.name == 'fasttext':
                for i, w in enumerate(msg_str):
                    e[i, :] = torch.tensor(self.model.get_word_vector(w))
            else:
                for i, w in enumerate(msg_str):
                    if w in self.model:
                        vec = self.model[w]
                        vec.flags.writeable = True  # to avoid user warning
                        vec = torch.tensor(vec)
                    else:
                        vec = torch.empty(self.emb_dim).uniform_(-.25, .25)
                    e[i, :] = vec
            # if sent emb, merge into average
            e = torch.sum(e, dim=0) / len(msg_str) if self.emb_type == 'snt' else e
        else:
            e = torch.empty(self.emb_dim).uniform_(-.25, .25)

        # truncate / pad embs if needed
        if type(self.in_width) == int and self.emb_type == 'tokens':
            if len(e) >= self.in_width:
                e = e[:self.in_width]
            else:
                e = torch.zeros(self.in_width, self.emb_dim, dtype=e.dtype, device=e.device)
                lp = (self.in_width - len(e)) // 2
                rp = int(self.in_width - len(e) - lp)
                # lpad = torch.zeros(lp, h, dtype=emb.dtype, device=emb)
                e[lp:self.in_width - rp] = e

        if seq:
            seq_e = self.encode_seq(seq, merge)
            if self.emb_type == 'tokens':
                seq_e = seq_e.unsqueeze(0).repeat(e.size(0), 1)
                e = torch.cat([e, seq_e], dim=1)
            else:
                e = torch.cat([e, seq_e], dim=0)
        return e

    def encode_seq(self, seq, merge='avg'):
        # seq is a list of strings (sents)
        lang = self.lang if self.lang != 'finest' else 'finnish'
        if self.name == 'elmo':
            seq = [' '.join(s) for s in seq] if self.emb_type == 'sents' else seq
            if merge == 'avg':
                embs = [torch.tensor(self.model.embed_sentence([s])) for s in seq]
                embs = torch.stack([e[self.elmo_dim] for e in embs])
                e = torch.sum(embs, dim=0) / embs.size(0)       # take average
            else:
                seq = [' '.join(seq)]
                e = torch.tensor(self.model.embed_sentence(seq))[self.elmo_dim]

        elif self.name == 'sbert':
            seq = [' '.join(msg_str) for msg_str in seq]
            if merge == 'avg':
                embs = torch.stack([self.model.encode(sent, convert_to_tensor=True, output_value='sentence_embedding',
                                                      device=self.device) for sent in seq])
                e = torch.sum(embs, dim=0) / embs.size(0)
            else:
                seq = [' '.join(seq)]
                e = self.model.encode(seq, convert_to_tensor=True, output_value='sentence_embedding',
                                      device=self.device)

        elif self.name in ['glove', 'word2vec', 'fasttext']:

            tokens = [tok for s in seq for tok in s]
            e = torch.empty(len(tokens), self.emb_dim)
            if self.name == 'fasttext':
                for i, w in enumerate(tokens):
                    e[i, :] = torch.tensor(self.model.get_word_vector(w))
            else:
                for i, w in enumerate(tokens):
                    if w in self.model:
                        vec = self.model[w]
                        vec.flags.writeable = True  # to avoid user warning
                        vec = torch.tensor(vec)
                    else:
                        vec = torch.empty(self.emb_dim).uniform_(-.25, .25)
                    e[i, :] = vec
            e = torch.sum(e, dim=0) / e.size(0)

        else:  # random vec
            e = torch.empty(self.emb_dim).uniform_(-.25, .25)
        return e


def get_w2v_embs(enc):

    from gensim.scripts.glove2word2vec import glove2word2vec
    from gensim.models.keyedvectors import KeyedVectors

    if enc == 'glove':
        vec_path = os.path.join(PROJ_DIR, 'embs', 'glove', 'vecs.txt')
        binary = False
        if not os.path.exists(vec_path):
            fname = 'glove.840B.300d.txt'
            glove2word2vec(glove_input_file=os.path.join(PROJ_DIR, 'embs', 'glove', fname),
                           word2vec_output_file=vec_path)
    else:
        vec_path = os.path.join(PROJ_DIR, 'embs', 'word2vec', 'GoogleNews-vectors-negative300.bin')
        binary = True
    model = KeyedVectors.load_word2vec_format(vec_path, binary=binary)
    return model


def get_fasttext_embs(lang):

    import fasttext.util

    lang = lang[:2]
    fn = 'cc.' + lang + '.300.bin'
    fp = os.path.join(PROJ_DIR, 'embs', 'fasttext', fn)

    if not os.path.exists(fp):
        fasttext.util.download_model(lang, if_exists='ignore')
        os.makedirs(fp.replace(fn, ''), exist_ok=True)
        cp = os.path.join(os.getcwd(), fn)
        os.replace(cp, fp)

    model = fasttext.load_model(fn)
    return model
