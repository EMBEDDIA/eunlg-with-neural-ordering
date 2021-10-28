
"""
implement the evaluation of ordering of whole paragraphs / articles. s
"""
import os

from scipy.stats import kendalltau
import torch
import numpy as np

from experiments.src.vars import WRKDIR


def predict_seq_order(model, seq, beam_size=10):
    """
    Given a sequence (paragraph / article), predict the ordering of the sentences in the sequence,
    using a pairwise model.
    :param model:
    :param seq sequence of sentences (paragraph / article)
    :param beam_size:
    :return:
    """
    if isinstance(seq, torch.nn.utils.rnn.PackedSequence):
        seq = torch.nn.utils.rnn.pad_packed_sequence(seq)[0]
        seq = torch.transpose(seq, 0, 1)

    seqlen = len(seq)
    beam = [([i], 0.0) for i in range(seqlen)]

    for i in range(seqlen - 1):
        new = []
        for tup in beam:
            for j in range(seqlen):
                o = tup[0] + [j]
                if len(set(o)) != len(o):
                    continue
                o_score = tup[1]
                for k in range(i + 1):
                    # add all pairwise scores s(i, j) where i < j, to total score
                    # model returns value from [0, 1] -> apply log
                    s1 = seq[o[k]].unsqueeze(0) if hasattr(model, 'conv_net') else seq[o[k]].unsqueeze(1)
                    s2 = seq[o[i + 1]].unsqueeze(0) if hasattr(model, 'conv_net') else seq[o[k]].unsqueeze(1)
                    o_score += torch.log(model(s1, s2))
                new += [(o, o_score)]
        # greater score is better, take only the <beam_size> best orders
        beam = sorted(new, key=lambda t: t[1], reverse=True)[:beam_size]
    return beam[0][0]   # best scoring order


def kendall_tau(pred_seqs, true_seqs):
    """
    Compute the mean of Kendaal's tau.

    For each paragraph, get all combinations (of sentence indices in correct order), then check whether
    pair of ind.pairs is concordant (x_i < x_j, y_i < y_j or x_i > x_j, y_i > y_j) or not.
    tau = #concordant - #discordant / n(n-1)/2
    :return:
    """
    assert len(pred_seqs) == len(true_seqs)
    n = len(pred_seqs)
    sum_tau = 0
    for pred_seq, true_seq in zip(pred_seqs, true_seqs):
        plen = len(pred_seq)
        assert plen == len(true_seq)
        # get combinations, count concordant pairs of sent.pairs
        pred_seq = np.array(pred_seq) if isinstance(pred_seq, list) else pred_seq.cpu().numpy()
        true_seq = np.array(true_seq) if isinstance(true_seq, list) else true_seq.cpu().numpy()
        tau = kendalltau(pred_seq, true_seq).correlation
        sum_tau += tau
    return sum_tau / n


def perfect_match_ratio(pred_seqs, true_seqs):
    """
    Perfect match ratio, i.e. the exactly matching orders across all predicted paragraphs.
    :return:
    """
    assert len(pred_seqs) == len(true_seqs)
    count = 0
    for pred_seq, true_seq in zip(pred_seqs, true_seqs):
        slen = len(pred_seq)
        if all(pred_seq[i] == true_seq[i] for i in range(slen)):
            count += 1
    return count / len(pred_seqs)


def positional_acc(pred_seqs, true_seqs):
    """
    For lists of predicted and correct sequences of sentences, get the ratio of matched sentence-level absolute
    positions between the lists.
    :param pred_seqs: list of torch Tensors
    :param true_seqs: list of torch Tensors
    :return:
    """
    assert len(pred_seqs) == len(true_seqs)
    n_sents = 0
    n_corr_pos = 0
    for pred_seq, true_seq in zip(pred_seqs, true_seqs):
        assert len(pred_seq) == len(true_seq)
        slen = len(pred_seq)
        # n_corr_pos += sum([1 if p == t else 0 for p, t in zip(pred_seq, true_seq)])
        s = sum(pred_seq[i] == true_seq[i] for i in range(slen))
        s = s.item() if isinstance(s, torch.Tensor) else s
        n_corr_pos += s
        n_sents += len(pred_seq)
    return n_corr_pos / n_sents


def log_test_scores(scores, model_name, args, te_args):

    args_to_print = {k: v for k, v in vars(args).items()}
    args_to_print['model_state_dict'] = list(args_to_print['model_state_dict'].keys())
    args_to_print['optimizer_state_dict'] = list(args_to_print['optimizer_state_dict'].keys())

    with open(os.path.join(WRKDIR, 'test_score_log.txt'), 'a') as f:
        f.write('\n --- Test set scores for model: {} ---\n'.format(model_name))
        f.write('Test args: {}\n'.format(vars(te_args)))
        f.write('Model args: {}\n'.format(args_to_print))
        if te_args.task == 'pw':
            line = '\nL: {:.4f}, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%, Acc.: {:.2f}%, AP: {:.2f}%\n\n'
            f.write(line.format(scores['loss'], scores['P'] * 100., scores['R'] * 100., scores['F1'] * 100.,
                                scores['A'] * 100., scores['AP'] * 100.))
        elif te_args.task == 'pos':
            line = '\nL: {:.4f}, P: {:.2f}%, R: {:.2f}%, F1: {:.2f}%, Acc.: {:.2f}%\n\n'
            f.write(line.format(scores['loss'], scores['P'] * 100., scores['R'] * 100., scores['F1'] * 100.,
                                scores['A'] * 100.))
        else:
            line = '\nL: {:.4f}, PMR: {:.2f}%, PAcc: {:.2f}%, tau: {:.3f}%\n\n'
            f.write(line.format(scores['loss'], scores['PMR'] * 100., scores['PAcc'] * 100., scores['tau']))

        f.write('\n###\n')



