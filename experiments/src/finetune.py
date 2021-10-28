

import argparse
import os
import shutil

import torch as t
from sentence_transformers import SentenceTransformer, evaluation, losses, InputExample, SentencesDataset

from experiments.src.data.sampling import get_training_and_test_sents, get_ranges, get_ind_pairs
from experiments.src.vars import MODEL_DIR


def fine_tune_sbert(sentlist, tr_ipairs, dev_ipairs, lang, args):

    dim = 1024 if lang == 'english' else 768
    model_name = 'xlm-r-bert-base-nli-stsb-mean-tokens' if lang != 'english' else \
        'bert-large-nli-stsb-mean-tokens'
    model = SentenceTransformer(model_name)

    examples = []
    dev_sents1, dev_sents2, dev_labels = [], [], []
    for i1, i2, lb in tr_ipairs:
        s1, s2 = sentlist[i1], sentlist[i2]
        examples += [InputExample(texts=[s1, s2], label=int(lb))]
    for i1, i2, lb in dev_ipairs:
        s1, s2 = sentlist[i1], sentlist[i2]
        dev_sents1 += [s1]
        dev_sents2 += [s2]
        dev_labels += [lb]

    dset = SentencesDataset(examples=examples, model=model)
    loader = t.utils.data.DataLoader(dataset=dset, shuffle=True, batch_size=args.batch_size)

    # get Evaluator
    evaluator = evaluation.BinaryClassificationEvaluator(sentences1=dev_sents1, sentences2=dev_sents2,
                                                         labels=dev_labels)
    ls = losses.SoftmaxLoss(model, sentence_embedding_dimension=dim, num_labels=2)

    outp = os.path.join(MODEL_DIR, 'sbert', 'statfi_' + lang[:2])
    if os.path.exists(outp) and len(os.listdir(outp)) > 0:
        shutil.rmtree(outp)
    model.fit(train_objectives=[(loader, ls)], evaluator=evaluator, epochs=args.n_epochs, output_path=outp)


if __name__ == '__main__':

    # run this to get the fine-tuned S-BERT model used for encoding sentences into embedddings,
    #   the model will be saved to neural_dp/experiments/sbert

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test_ind_file', default='english-0.2-1-arts.txt')
    parser.add_argument('--dev_ratio', nargs='?', type=float, default=0.1)
    parser.add_argument('--sample_pars', nargs='*', default=['split=pars', 'tr-sample=half', 'dev-sample=half'])
    parser.add_argument('--model', nargs='?', default='sbert')

    parser.add_argument('--n_epochs', nargs='?', type=int, default=20)
    parser.add_argument('--batch_size', nargs='?', type=int, default=16)

    pars = parser.parse_args()

    spars = {p.split('=')[0]: p.split('=')[1] for p in pars.sample_pars}

    tr_artlist, te_artlist = get_training_and_test_sents(pars.test_ind_file)
    # artlist = tr_artlist + te_artlist         # use whole corpus
    artlist = tr_artlist                        # use training set only
    parlist = [p for a in artlist for p in a]
    sentlist = [s for p in parlist for s in p]
    seqlist = artlist if spars['split'] == 'arts' else parlist

    # get tr, dev pairs
    if pars.seed:
        t.manual_seed(pars.seed)
    rngs = get_ranges(artlist)

    n = len(seqlist)
    n_dev = int(pars.dev_ratio * n)
    n_tr = n - n_dev
    inds = [i for i in range(len(seqlist))]
    tr_inds = inds[:n_tr]
    dev_inds = inds[n_tr:]
    tr_ipairs = get_ind_pairs(tr_inds, rngs, spars['tr-sample'], spars['split'])
    dev_ipairs = get_ind_pairs(dev_inds, rngs, spars['dev-sample'], spars['split'])

    # fine tuning:
    lang = pars.test_ind_file.split('-')[0]

    fine_tune_sbert(sentlist, tr_ipairs, dev_ipairs, lang, pars)
