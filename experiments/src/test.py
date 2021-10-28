

import argparse
import os
import random
import sys

import torch as t
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from experiments.src.data.sampling import get_training_and_test_sents, get_ind_pairs, get_ranges
from experiments.src.data.encoder import Encoder, collate_order, collate_pw, collate_pos
from experiments.src.data.datasets import SentPairDataset, SequenceDataset, PositionDataset
from experiments.src.vars import DEVICE, MODEL_DIR
from experiments.src.models import pairnets, seq2seq
from experiments.src.eval import log_test_scores, perfect_match_ratio, positional_acc, kendall_tau, predict_seq_order
from experiments.src.trainer import get_eyeball_sample

print('sys.path: ', sys.path)

parser = argparse.ArgumentParser()

parser.add_argument('--task', nargs='?', default='pos')
parser.add_argument('--test_ind_file', nargs='?', default='english-0.2-1-arts.txt')
parser.add_argument('--seed', nargs='?', type=int, default=1)
parser.add_argument('--sample_pars', nargs='*', default=['split=pars', 'sample=half'])
parser.add_argument('--model_fname', nargs='?')
parser.add_argument('--random', action='store_true')        # whether to make random predictions

test_args = parser.parse_args()

sample_pars = {p.split('=')[0]: p.split('=')[1] for p in test_args.sample_pars}

_, artlist = get_training_and_test_sents(test_args.test_ind_file)
parlist = [p for a in artlist for p in a]
sentlist = [s for p in parlist for s in p]


if os.sep in test_args.model_fname:
    dirname, model_fname = test_args.model_fname.split(os.sep)

# model_pars needed: loss_fn, emb_pars, ind_file, input shape, model name,

if test_args.seed:
    t.manual_seed(seed=test_args.seed)

modelpath = os.path.join(MODEL_DIR, test_args.model_fname)
model_args = t.load(modelpath, map_location=DEVICE)
model_args = argparse.Namespace(**model_args)

# init. encoder
encoder = Encoder(emb_pars=model_args.emb_pars)
# get embs
load_embs = False
if encoder.name in ['word2vec', 'glove', 'fasttext']:
    load_embs = True
embeddings, vocab = encoder.load_embs(sentlist) if load_embs else (None, None)

task = model_args.task if not test_args.task else test_args.task

if not hasattr(model_args, 'model_type'):
    model_args.model_type = 'pos' if 'Position' in model_args.model else 'pw'
    model_args.model_type = 'order' if 'Pointer' in model_args.model else model_args.model_type
model_class = model_args.model
emb_pars = model_args.emb_pars

model = getattr(pairnets, model_args.model) if model_args.model_type == 'pw' else getattr(seq2seq, model_args.model)
model = model(params=model_args, n_sents=len(sentlist), embeddings=embeddings)

model.load_state_dict(model_args.model_state_dict)

model = model.to(DEVICE)
model.eval()

collate_fn = None
if task == 'pw':
    collate_fn = collate_pw if emb_pars['len'] == 'W' else None
elif task == 'order':
    collate_fn = collate_order
else:
    collate_fn = collate_pos

ranges = get_ranges(artlist)
if task == 'order' and sample_pars['split'] == 'pars':
    ranges = {a_rn: [prn for prn in p_rns if prn[1] - prn[0] > 1] for a_rn, p_rns in ranges.items()}
    parlist = [par for par in parlist if len(par) > 1]

seqlist = artlist if sample_pars['split'] == 'arts' else parlist
n = len(seqlist)
print('n_seqs: ', n)
inds = t.randperm(n=n).tolist()

print('model.state_dict() keys: ', model.state_dict().keys())

loss_fn = model_args.loss_fn

if task == 'pw':
    samples = get_ind_pairs(inds, ranges, sample_pars['sample'], sample_pars['split'])
    print('npairs: ', len(samples))

    dset = SentPairDataset(sentlist, ind_pairs=samples, params=model_args, encoder=encoder, vocab=vocab)
    scores = {'loss': 0, 'P': 0, 'R': 0, 'F1': 0, 'A': 0, 'AP': 0}
else:
    a_rns, p_rns = list(ranges.keys()), [p for a in ranges.values() for p in a]
    samples = [p_rns[i] for i in inds] if sample_pars['split'] == 'pars' else [a_rns[i] for i in inds]

    if task == 'order':
        dset = SequenceDataset(sentlist, iseqs=samples, encoder=encoder, params=model_args, seed=model_args.seed,
                               seq_type=sample_pars['split'], vocab=vocab)
        scores = {'loss': 0, 'tau': 0, 'PMR': 0, 'PAcc': 0}
    else:   # pos
        dset = PositionDataset(sentlist, iseqs=samples, encoder=encoder, params=model_args, seed=model_args.seed)
        scores = {'loss': 0, 'P': 0, 'R': 0, 'F1': 0, 'A': 0}
        samples = dset.sent_inds

bs = 16
n = len(dset)
loader = t.utils.data.DataLoader(dataset=dset, batch_size=bs, shuffle=True, drop_last=False,
                                 collate_fn=collate_fn)

# get predictions on test set, both pairs and complete orderings of paragraphs / articles
actual, predicted, pred_labels = [], [], []
last_bs, last_loss = bs, 0
n_eb_samples = 0
loss = 0
for bi, (data, target) in enumerate(loader):

    if task == 'pw':
        data = (data[0].to(DEVICE), data[1].to(DEVICE))
        target = target.to(DEVICE)

        with t.no_grad():
            output = model(*data).squeeze()
        loss = loss_fn(output, target)

        target = target.cpu().numpy()
        if type(model_args.loss_fn) == t.nn.modules.loss.BCEWithLogitsLoss:
            output = t.sigmoid(output)
        preds = output.cpu().numpy()
        pred_lbs = (output >= 0.5).int().cpu().numpy()

        actual = np.concatenate((np.array(actual), target))
        predicted = np.concatenate((np.array(predicted), preds))
        pred_labels = np.concatenate((np.array(pred_labels), pred_lbs))

    elif task == 'order':   # list of tensors (ordering)
        data = [seq.to(DEVICE) for seq in data]
        target = [o.to(DEVICE) for o in target]
        preds = []
        with t.no_grad():

            if test_args.random:
                preds = [t.randperm(n=len(o)) for o in target]
                predicted += preds
                actual += target

            elif model_args.model_type == 'order':
                loss = 0
                for seq, order in zip(data, target):
                    pred, seq_loss = model(seq, order, teacher_force_ratio=0.)
                    loss += seq_loss
                    predicted += [pred]
                    actual += [order]

            elif model_args.model_type == 'pos':
                preds = [model(seq) for seq in data]  # one seq is a batch
                try:
                    losses = t.tensor([loss_fn(pred, order) for pred, order in zip(preds, target)])
                except IndexError:
                    print('Error, Target 10 out of bounds, preds, target: ')
                    print('preds: ', preds)
                    print('target: ', target)
                loss = t.sum(losses)
                preds = [t.softmax(p, dim=1) for p in preds]
                qs = t.arange(start=1, end=model.n_quantiles + 1, device=DEVICE)
                sums = [t.sum(pred * qs, dim=1) for pred in preds]
                pred_orders = [t.argsort(s) for s in sums]
                for pred, o in zip(pred_orders, target):
                    predicted += [pred]
                    actual += [o]
            else:
                # pairwise models
                preds = [predict_seq_order(model, seq) for seq in data]
                for pred, o in zip(preds, target):
                    predicted += [pred]
                    actual += [o]
                # how to compute loss? or simply skip

    else:       # pos
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        if test_args.random:
            nq = int(model_args.model_pars['nq'][0])
            pred_lbs = t.randint(low=0, high=nq, size=(len(target),)).cpu().numpy()
        else:
            with t.no_grad():
                output = model(data).squeeze()
            loss = loss_fn(output, target)
            target = target.cpu().numpy()
            if type(model_args.loss_fn) == t.nn.modules.loss.CrossEntropyLoss:
                output = t.softmax(output, dim=1)

            preds = output.cpu().numpy()
            pred_lbs = np.argmax(preds, axis=1)

        actual = np.concatenate((np.array(actual), target))
        pred_labels = np.concatenate((np.array(pred_labels), pred_lbs))

    loss = loss.item() if loss else 0
    if bi == len(loader) - 1:
        last_loss = loss
        last_bs = len(target)
    else:
        scores['loss'] += loss

    # get eyeball sample
    if bi % 100 == 0 and n_eb_samples < 10 and not last_loss and not test_args.random:
        n_eb_samples += 1
        st_i, end_i = bs * bi, bs * (bi + 1)
        eb_inds = random.sample(range(st_i, end_i), k=10)
        eb_smps = [samples[i] for i in eb_inds]
        if task != 'order':
            eb_preds = [preds[i % st_i] if st_i > 0 else preds[i] for i in eb_inds]
        else:
            eb_preds = [predicted[i] for i in eb_inds]
        eb_labels = [target[i % st_i] if st_i > 0 else target[i] for i in eb_inds] if task == 'pos' else None
        get_eyeball_sample(eb_smps, eb_preds, sentlist, model_fname, task, dset='te', labels=eb_labels)

n_before_last = bs * (len(loader) - 1)
len_ldr = max(1, len(loader) - 1)
scores['loss'] = (n_before_last * (scores['loss'] / len_ldr) + last_bs * last_loss) / n
if task != 'order':
    avg = 'micro' if task == 'pw' else 'weighted'
    scores['P'], scores['R'], scores['F1'], _ = precision_recall_fscore_support(actual, pred_labels, average='micro')
    scores['A'] = np.sum(pred_labels == actual) / n
    if task == 'pw':
        scores['AP'] = average_precision_score(actual, predicted)

else:
    scores['PMR'] = perfect_match_ratio(predicted, actual)
    scores['PAcc'] = positional_acc(predicted, actual)
    scores['tau'] = kendall_tau(predicted, actual)

if test_args.random:
    model_fname = 'random'
print('\nTest set scores for model {}:'.format(model_fname))
print('Test args: {}'.format(vars(test_args)))
args_to_print = {k: v for k, v in vars(model_args).items()}
args_to_print['model_state_dict'] = list(args_to_print['model_state_dict'].keys())
args_to_print['optimizer_state_dict'] = list(args_to_print['optimizer_state_dict'].keys())
print('wIth args: {}'.format(args_to_print))

if task == 'pw':
    score_line = 'L: {:.4f}, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%, Acc.: {:.2f}%, AP: {:.2f}%\n'
    print(score_line.format(scores['loss'], scores['P'] * 100., scores['R'] * 100., scores['F1'] * 100.,
                            scores['A'] * 100., scores['AP'] * 100.))
elif task == 'pos':
    print('with {} quantiles, weighted average of P, R, F1:'.format(model.n_quantiles))
    score_line = 'L: {:.4f}, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%, Acc.: {:.2f}%\n'
    print(score_line.format(scores['loss'], scores['P'] * 100., scores['R'] * 100., scores['F1'] * 100.,
                            scores['A'] * 100.))
else:
    score_line = 'L: {:.4f}, PMR: {:.2f}%, PAcc: {:.2f}%, tau: {:.3f}%\n'
    print(score_line.format(scores['loss'], scores['PMR'] * 100., scores['PAcc'] * 100., scores['tau']))

log_test_scores(scores, model_fname, model_args, test_args)
