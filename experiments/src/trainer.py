
import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from experiments.src.vars import DEVICE, WRKDIR
from experiments.src.utils import get_model_name
from experiments.src.eval import positional_acc, perfect_match_ratio, kendall_tau, predict_seq_order


class Trainer(object):

    def __init__(self, tr_loader, dev_loader, dev_samples, scores, opt, loss_fn, sentlist, params):

        self.params = params
        self.n_epochs = params.n_epochs
        self.batch_size = params.batch_size
        self.plot = params.plot
        self.final = params.final

        self.tr_loader = tr_loader
        self.dev_loader = dev_loader
        self.dev_samples = dev_samples
        self.scores = scores

        self.opt = opt
        self.loss_fn = loss_fn
        self.model_fname = get_model_name(params, add_time=False).replace('.pt', '')

        self.early_stop_score = params.early_stop
        self.early_stop = False

        self.plot_iters = []

        self.sentlist = sentlist

        self.task = params.task

    def train(self, model, epoch):

        model.train()
        # train
        l_hidden, r_hidden = None, None
        for bi, (data, target) in enumerate(self.tr_loader):

            if self.params.model_type == 'pw':
                sents1, sents2 = data
                sents1 = sents1.to(DEVICE)
                sents2 = sents2.to(DEVICE)
                target = target.to(DEVICE)

                self.opt.zero_grad()
                if 'LSTM' in self.params.model:
                    preds = model(sents1, sents2, l_hidden, r_hidden).squeeze()
                else:
                    preds = model(sents1, sents2).squeeze()

                loss = self.loss_fn(preds, target)

            elif self.params.model_type == 'order':
                seqs = [seq.to(DEVICE) for seq in data]
                orders = [o.to(DEVICE) for o in target]

                self.opt.zero_grad()
                # loop over single samples of training splits? then compute loss for a batch
                #    in order to avoid the variable-length input issue
                loss = 0
                for seq, order in zip(seqs, orders):
                    pred, seq_loss = model(seq, order)
                    loss += seq_loss
            else:       # pos
                sents = data.to(DEVICE)
                target = target.to(DEVICE)

                self.opt.zero_grad()

                preds = model(sents)
                loss = self.loss_fn(preds, target)

            loss.backward()
            self.opt.step()

            if bi % 500 == 0:
                print('Train epoch: {}/{} [{}--{}/{} ({:.0f}%)]\tTraining loss for batch {}/{}: {:.6f}'.format(
                    epoch + 1, self.n_epochs, bi * self.batch_size + 1, (bi + 1) * self.batch_size,
                    len(self.tr_loader.dataset), 100. * bi / len(self.tr_loader), bi + 1, len(self.tr_loader),
                    float(loss)))

            # validate for plotting - for every 1000th iteration
            iter_i = epoch * len(self.tr_loader) + bi   # iteration
            if self.plot and iter_i % 1000 == 0:
                print('self.plot: ', self.plot)
                print('self.final: ', self.final)
                self.validate(epoch, model) if self.task != 'order' else self.validate_order(epoch, model)
                self.plot_iters += [iter_i]
                model.train()
        if not self.plot and not self.final:
            self.validate(epoch, model) if self.task != 'order' else self.validate_order(epoch, model)
            model.train()

        return model

    def validate(self, epoch, model):

        model.eval()
        val_scores = {k: 0 for k in self.scores}
        n_eb_samples = 0
        actual, predicted, pred_labels = [], [], []
        last_bs, last_loss = self.batch_size, 0
        for bi, (data, target) in enumerate(self.dev_loader):

            if self.task == 'pw':
                sents1, sents2 = data
                sents1 = sents1.to(DEVICE)
                sents2 = sents2.to(DEVICE)
                target = target.to(DEVICE)
                with torch.no_grad():
                    output = model(sents1, sents2).squeeze()
                loss = self.loss_fn(output, target).data.item()
                target = target.cpu().numpy()
                output = torch.sigmoid(output)      # assuming loss is BCEWithLogitsLoss
                preds = output.cpu().numpy()
                pred_lbs = (output >= 0.5).int().cpu().numpy()
            else:   # task = pos
                sents = data.to(DEVICE)
                target = target.to(DEVICE)

                with torch.no_grad():
                    output = model(sents)

                loss = self.loss_fn(output, target).data.item()
                target = target.cpu().numpy()
                output = torch.softmax(output, dim=1)       # assuming CrossEntropyLoss (

                preds = output.cpu().numpy()
                pred_lbs = np.argmax(preds, axis=1)

            actual += [target]
            predicted += [preds]
            pred_labels += [pred_lbs]
            if bi == len(self.dev_loader) - 1:  # in case last batch smaller than others
                last_loss = loss
                last_bs = len(target)
            else:
                val_scores['loss'] += loss

            if (epoch == self.n_epochs - 1 or self.early_stop) and bi % 100 == 0 and n_eb_samples < 20:
                n_eb_samples += 1
                st_i, end_i = self.batch_size * bi, self.batch_size * (bi + 1)
                eb_inds = random.sample(range(st_i, end_i), k=10)
                eb_samples = [self.dev_samples[i] for i in eb_inds]
                eb_preds = [preds[i % st_i] if st_i > 0 else preds[i] for i in eb_inds]   # preds is of size batch size
                eb_labels = [target[i % st_i] if st_i > 0 else target[i] for i in eb_inds] if self.task == 'pos' \
                    else None
                get_eyeball_sample(eb_samples, eb_preds, self.sentlist, self.model_fname, self.params.task,
                                   labels=eb_labels)
                print('After epoch {}/{}, for dev batch {}/{} - 10 predictions stored in eyeball file.'
                      .format(epoch + 1, self.n_epochs, bi + 1, len(self.dev_loader)))

        actual = np.concatenate(actual)
        predicted = np.concatenate(predicted)
        pred_labels = np.concatenate(pred_labels)

        n_to_last = self.batch_size * (len(self.dev_loader) - 1)
        n = n_to_last + last_bs
        val_scores['loss'] = (n_to_last * (val_scores['loss'] / (len(self.dev_loader) - 1)) + last_bs * last_loss) / n
        avg = 'micro' if self.task == 'pw' else 'weighted'
        val_scores['P'], val_scores['R'], val_scores['F1'], _ = precision_recall_fscore_support(actual, pred_labels,
                                                                                                average=avg)
        val_scores['A'] = np.sum(pred_labels == actual) / n
        if 'AP' in self.scores:  # in case self.task == 'pw'
            val_scores['AP'] = average_precision_score(actual, predicted)
            self.scores['AP'] += [val_scores['AP']]
            score_line = 'L: {:.4f}, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%, Acc.: {:.2f}%, AP: {:.2f}%\n'
        else:
            score_line = 'L: {:.4f}, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%, Acc.: {:.2f}%\n'

        self.scores['loss'] += [val_scores['loss']]
        self.scores['P'] += [val_scores['P']]
        self.scores['R'] += [val_scores['R']]
        self.scores['F1'] += [val_scores['F1']]
        self.scores['A'] += [val_scores['A']]

        print('\nDev set scores for model {}:'.format(self.model_fname))
        if self.task == 'pw':   # AP in scores
            print(score_line.format(val_scores['loss'], val_scores['P'] * 100., val_scores['R'] * 100.,
                                    val_scores['F1'] * 100., val_scores['A'] * 100., val_scores['AP'] * 100.))
        else:
            print('with {} quantiles, weighted averages for P, R, F1:'.format(model.n_quantiles))
            print(score_line.format(val_scores['loss'], val_scores['P'] * 100., val_scores['R'] * 100.,
                                    val_scores['F1'] * 100., val_scores['A'] * 100.))

    def validate_order(self, epoch, model):

        model.eval()
        scores = {k: 0 for k in self.scores}
        n_eb_samples = 0
        predicted, actual = [], []
        last_bs, last_loss = self.batch_size, 0
        loss = 0    # for pw ordering, no loss
        for bi, (seqs, orders) in enumerate(self.dev_loader):

            seqs = [seq.to(DEVICE) for seq in seqs]
            orders = [o.to(DEVICE) for o in orders]

            with torch.no_grad():
                if self.params.model_type == 'order':     # using Pointer Net
                    loss = 0
                    for seq, order in zip(seqs, orders):
                        pred, seq_loss = model(seq, order, teacher_force_ratio=0.)
                        loss += seq_loss
                        predicted += [pred]
                        actual += [order]
                elif self.params.model_type == 'pos':
                    # CE loss is not perfect for the pos case, since 10 classes always predicted but seq length varies
                    preds = [model(seq) for seq in seqs]  # one seq is a batch
                    losses = torch.tensor([self.loss_fn(pred, order) for pred, order in zip(preds, orders)])
                    loss = torch.sum(losses)
                    preds = [torch.softmax(p, dim=1) for p in preds]
                    quantiles = torch.arange(start=1, end=model.n_quantiles + 1, device=DEVICE)
                    sums = [torch.sum(pred * quantiles, dim=1) for pred in preds]
                    pred_orders = [torch.argsort(s) for s in sums]
                    for pred, o in zip(pred_orders, orders):
                        predicted += [pred]
                        actual += [o]
                else:
                    # pw models
                    preds = [predict_seq_order(model, seq) for seq in seqs]
                    for pred, o in zip(preds, orders):
                        predicted += [pred]
                        actual += [o]

            loss = loss.item() if loss else 0
            if bi == len(self.dev_loader) - 1:
                last_loss = loss
                last_bs = len(orders)
            else:
                scores['loss'] += loss

            if (epoch == self.n_epochs - 1 or self.early_stop) and bi % 100 == 0 and n_eb_samples < 20:
                n_eb_samples += 1
                st_i, end_i = self.batch_size * bi, self.batch_size * (bi + 1)
                eb_inds = random.sample(range(st_i, end_i), k=10)
                eb_rngs = [self.dev_samples[i] for i in eb_inds]
                eb_preds = [predicted[i] for i in eb_inds]
                get_eyeball_sample(eb_rngs, eb_preds, self.sentlist, self.model_fname, self.params.task)
                print('After epoch {}/{}, for dev batch {}/{} - 10 predictions stored in eyeball file.'
                      .format(epoch + 1, self.n_epochs, bi + 1, len(self.dev_loader)))

        n_to_last = self.batch_size * (len(self.dev_loader) - 1)
        n = last_bs + n_to_last
        scores['loss'] = (n_to_last * (scores['loss'] / (len(self.dev_loader) - 1)) + last_bs * last_loss) / n

        scores['PMR'] = perfect_match_ratio(predicted, actual)
        scores['PAcc'] = positional_acc(predicted, actual)
        scores['tau'] = kendall_tau(predicted, actual)

        self.scores['loss'] += [scores['loss']]
        self.scores['PMR'] += [scores['PMR']]
        self.scores['PAcc'] += [scores['PAcc']]
        self.scores['tau'] += [scores['tau']]

        print('\nDev set scores for model {}:'.format(self.model_fname))
        score_line = 'L: {:.4f}, PMR: {:.2f}%, PAcc: {:.2f}%, tau: {:.3f}\n'
        print(score_line.format(scores['loss'], scores['PMR'] * 100., scores['PAcc'] * 100.,
                                scores['tau']))


def get_eyeball_sample(samples, preds, sentlist, model_fname, task='pw', dset='val', labels=None):
    """
    :param samples: (s1, s2, label) tuples or (i, j) ranges
    :param preds: ordering: List of indices, predicted orders, pw: labels
    :param sentlist:
    :param model_fname:
    :param task:
    :return:
    """
    # get an eyeball sample from the validaton set
    eb_fp = os.path.join(WRKDIR, 'eyeball', dset + '_eb_{}.txt'.format(model_fname))
    with open(eb_fp, 'a', encoding='utf-8') as f:
        for i, sample in enumerate(samples):
            if task == 'pw':
                sent1, sent2 = sentlist[sample[0]], sentlist[sample[1]]
                lb = sample[2]
                line = 'Sent pair {}:\nSent 1:\t{}\nSent 2:\t{}\nPred = {:.4f}, actual = {}.\n'
                f.write(line.format(sample, sent1, sent2, preds[i], lb))
            elif task == 'pos':
                sent = sentlist[sample]
                lb = labels[i]
                line = 'Sentence, i {}:\nSentence:\t{}\nPred = {} -> {}, actual = {}.\n'
                ps = ' '.join(['{:.2f}'] * len(preds[i])).format(*preds[i])
                f.write(line.format(sample, sent, ps, preds[i].argmax(), lb))
            else:
                sents = [sentlist[j] for j in range(*sample)]
                p = preds[i] if isinstance(preds[i], list) else preds[i].tolist()
                line = 'Sequence of sents {}:\n'.format(sample) +\
                    '\n'.join(['Sent {}:\t{}'.format(j, s) for j, s in enumerate(sents)]) +\
                    '\nPred. order = {}.\n'.format(preds[i])
                f.write(line)


def log_scores(fold_scores, params):
    n_epochs = params.n_epochs
    model_fname = get_model_name(params).replace('.pt', '')
    folds = params.cv_folds

    print('Write results to file...')
    with open(os.path.join(WRKDIR, 'score_log.txt'), 'a') as f:
        for fld in range(folds):
            total_epochs = len(fold_scores[fld]['loss'])
            # amount of actual epochs, in case of early stopping
            f.write('\n\n#####\n')
            f.write('\nScores for model: {}\n'.format(model_fname))
            f.write('--- Fold {} ---\n'.format(fld + 1))
            f.write('\nAfter training for {}/{} epochs, scores for model {}:\n'.format(total_epochs, n_epochs,
                                                                                       model_fname))
            f.write('Losses: {}\n'.format(' '.join(['{:.4f}'.format(s) for s in fold_scores[fld]['loss']])))
            if params.task != 'order':
                f.write('Precisions: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in fold_scores[fld]['P']])))
                f.write('Recalls: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in fold_scores[fld]['R']])))
                f.write('F1-s: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in fold_scores[fld]['F1']])))
                f.write('Accuracies: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in fold_scores[fld]['A']])))
                if params.task == 'pw':
                    f.write('Avg. precs: {}\n'.format(' '.join(['{:.2f}'.format(s * 100)
                                                                for s in fold_scores[fld]['AP']])))
            else:
                f.write('PMR: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in fold_scores[fld]['PMR']])))
                f.write('PAcc: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in fold_scores[fld]['PAcc']])))
                f.write('tau: {}\n'.format(' '.join(['{:.2f}'.format(s * 100) for s in fold_scores[fld]['tau']])))

        if folds > 1:
            np.set_printoptions(formatter={'float': '{:.4f}'.format})
            f.write('--- Mean + std / epoch over folds: ---\n')
            f.write('Losses: m-{}, std-{}\n'.format(np.mean([fold_scores[fld]['loss'] for fld in range(folds)], axis=0),
                                                    np.std([fold_scores[fld]['loss'] for fld in range(folds)], axis=0)))
            np.set_printoptions(formatter={'float': '{:.2f}'.format})
            if params.task != 'order':
                f.write('Precs: m-{}, std-{}\n'.format(np.mean([fold_scores[fld]['P'] for fld in range(folds)], axis=0) * 100.,
                                                        np.std([fold_scores[fld]['P'] for fld in range(folds)], axis=0) * 100.))
                f.write('Recs: m-{}, std-{}\n'.format(np.mean([fold_scores[fld]['R'] for fld in range(folds)], axis=0) * 100.,
                                                       np.std([fold_scores[fld]['R'] for fld in range(folds)], axis=0) * 100.))
                f.write('F1-s: m-{}, std-{}\n'.format(np.mean([fold_scores[fld]['F1'] for fld in range(folds)], axis=0) * 100.,
                                                      np.std([fold_scores[fld]['F1'] for fld in range(folds)], axis=0) * 100.))
                f.write('Accs: m-{}, std-{}\n'.format(np.mean([fold_scores[fld]['A'] for fld in range(folds)], axis=0) * 100.,
                                                      np.std([fold_scores[fld]['A'] for fld in range(folds)], axis=0) * 100.))
                if params.task == 'pw':
                    f.write('APs: m-{}, std-{}\n'.format(np.mean([fold_scores[fld]['AP'] for fld in range(folds)],
                                                                 axis=0) * 100.,
                                                         np.std([fold_scores[fld]['AP'] for fld in range(folds)],
                                                                axis=0) * 100.))
            else:
                f.write('PMRs: m-{}, std-{}\n'.format(
                    np.mean([fold_scores[fld]['PMR'] for fld in range(folds)], axis=0) * 100.,
                    np.std([fold_scores[fld]['PMR'] for fld in range(folds)], axis=0) * 100.))
                f.write('PAccs: m-{}, std-{}\n'.format(
                    np.mean([fold_scores[fld]['PAcc'] for fld in range(folds)], axis=0) * 100.,
                    np.std([fold_scores[fld]['PAcc'] for fld in range(folds)], axis=0) * 100.))
                f.write('taus: m-{}, std-{}\n'.format(
                    np.mean([fold_scores[fld]['tau'] for fld in range(folds)], axis=0) * 100.,
                    np.std([fold_scores[fld]['tau'] for fld in range(folds)], axis=0) * 100.))

            f.write('\n#####\n')
    print('Done!')


def plot(plot_iters, scores, model_fname):

    print('Write scores to files for plotting...')
    iters_file = os.path.join(WRKDIR, 'plots', 'iters.dat')
    with open(iters_file, 'a') as f:
        f.write(model_fname + ':\t' + '\t'.join([str(i) for i in plot_iters]) + '\n')

    score_list = list(scores[0].keys())
    avg_scores = {}
    for score in score_list:
        avg_scores[score] = np.mean([scores[f][score] for f in range(len(scores))], axis=0)
        fp = os.path.join(WRKDIR, 'plots', score + '.dat')
        with open(fp, 'a') as f:
            f.write(model_fname + ':\t' + '\t'.join(['{:2.3f}'.format(s) for s in avg_scores[score]]) + '\n')
    print('Done!')


def check_early_stop(metric, scores):

    if metric == 'loss' and scores[metric][-1] > scores[metric][-2] > scores[metric][-3]:
        return True
    elif metric != 'loss' and scores[metric][-1] < scores[metric][-2] < scores[metric][-3]:
        return True
    else:
        return False
