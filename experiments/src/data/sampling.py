

import os
import itertools as it
import random
from collections import OrderedDict

from experiments.src.data.statfi import read_corpus
from experiments.src.vars import WRKDIR, SPLITS_DIR


def get_annotation_set(artlist, n=10):
    """
    pick a random set of articles, give a list of sents in random order, and annotate (by hand):
    param artlist: articles -> pars -> sents
    :param n:
    :return:
    """
    annset = random.sample(artlist, k=n)
    sent_set = []       # list of lists - sent_set[i] = sents of article i in random order
    shuff_pars = []
    for i in range(n):
        pars = annset[i]
        all_sents = [s for p in pars for s in p]
        sent_set += [random.sample(all_sents, k=len(all_sents))]
        shuff_pars += [random.sample(par, k=len(par)) for par in pars]
    i = len(os.listdir(os.path.join(WRKDIR, 'annotation'))) // 2
    fp = os.path.join(WRKDIR, 'annotation', 'annset_{}.txt'.format(i))
    fp_arts = os.path.join(WRKDIR, 'annotation', 'art_{}.txt'.format(i))
    sent_line = 'Sentence {}:\t{}\n'
    with open(fp, 'w', encoding='utf-8') as f:
        f.write('All sents shuffled:\n')
        for ai, a in enumerate(sent_set):
            f.write('Article {}:\n'.format(ai + 1))
            for si, s in enumerate(a):
                f.write(sent_line.format(si + 1, s))
        f.write('\n\n\nParagraphs in order, sents shuffled within:\n')
        for i in range(n):
            f.write('Article {}:\n'.format(i + 1))
            pars = [random.sample(par, k=len(par)) for par in annset[i]]
            for pi, p in enumerate(pars):
                f.write('Paragraph {}:\n'.format(pi + 1))
                for si, s in enumerate(p):
                    f.write(sent_line.format(si + 1, s))

    # write original articles also to file
    with open(fp_arts, 'w', encoding='utf-8') as f:
        for ai, a in enumerate(annset):
            f.write('Article {}:\n'.format(ai + 1))
            for pi, p in enumerate(a):
                f.write('Paragraph {}:\n'.format(pi + 1))
                for si, s in enumerate(p):
                    f.write(sent_line.format(si + 1, s))
                f.write('\n')


def get_ranges(artlist):
    """
    Get sentence index starting and ending paragraphs and articles; the last index is that of the last sentence + 1.
    :param artlist: List[List[List]], arts[pars[sents]]
    :return: pairs of indices of articles / pars in a single list of all sents
    """

    prev_ind = 0
    rngs = OrderedDict()
    for a in artlist:
        a_st = prev_ind
        par_inds = []
        for p in a:
            p_st = prev_ind
            prev_ind += len(p)
            par_inds += [(p_st, prev_ind)]
        rngs[(a_st, prev_ind)] = par_inds

    return rngs


def get_ind_pairs(inds, rngs, sample, split):
    """
    :param inds: paragraphs / articles to get ind pairs for.
    :param rngs: sentence index ranges of all articles: pars (dict)
    :param sample: flp, all, half
    :param split: whether pairs sampled from within arts or pars
    :return: pairs of inds
    """

    def half_pairs(rns):
        # get pairs between first and second half of all sentences in a paragraph
        ind_pairs = []
        for p_rns in rns.values():
            for rn in p_rns:
                half_i = rn[0] + (rn[1] - rn[0]) // 2
                lr_inds = [(*t, 1) for t in it.permutations(range(*rn), r=2) if t[0] < half_i >= t[1]]
                rl_inds = [(j, i, 0) for i, j, _ in lr_inds]
                ind_pairs += lr_inds + rl_inds
        return ind_pairs

    def get_all_pairs(rns):
        # get all left-right pairs from within paragraph
        ind_pairs = []
        for p_rns in rns.values():
            for rn in p_rns:
                lr_inds = [(*t, 1) for t in it.permutations(range(*rn), r=2) if t[0] < t[1]]
                rl_inds = [(j, i, 0) for i, j, _ in lr_inds]
                ind_pairs += lr_inds + rl_inds
        return ind_pairs

    def first_last_pairs(rns):
        # use sent inds of ARTICLES, not pars
        ind_pairs = []
        for ai, p_rns in enumerate(rns.values()):
            # check that article contains at least 2 paragraphs
            if len(p_rns) < 2:
                continue
            first_p, last_p = range(*p_rns[0]), range(*p_rns[-1])
            fl_inds = [(*t, 1) for t in list(it.product(first_p, last_p))]
            lf_inds = [(j, i, 0) for i, j, _ in fl_inds]
            ind_pairs += fl_inds + lf_inds
        return ind_pairs

    rns = {}
    if split == 'arts':                 # inds are article inds
        arts = list(rngs.keys())
        arts = [arts[i] for i in inds]
        rns = {a: rngs[a] for a in arts}
        assert len(rns) == len(inds)
    else:                               # inds are paragraph inds
        pi = 0
        for a, pars in rngs.items():
            val = []
            for p in pars:
                if pi in inds:
                    val += [p]
                pi += 1
            rns[a] = val
        assert len(inds) == sum([len(v) for v in rns.values()])

    # check that sample matches par / art (i.e. no index errors)
    if sample == 'all':
        return get_all_pairs(rns)  # rngs = par_rngs
    elif sample == 'flp':
        return first_last_pairs(rns)     # rngs = art_rngs
    elif sample == 'half':
        return half_pairs(rns)     # rngs


def split_dataset(spairs, test_ratio=0.2, s=100):

    random.seed(s)
    random.shuffle(spairs)
    i = int(len(spairs) * (1 - test_ratio))
    tr, te = spairs[:i], spairs[i:]
    return tr, te


def get_training_and_test_sents(test_inds_file):

    fp = os.path.join(SPLITS_DIR, test_inds_file)
    lang, ratio, seed, grp, = test_inds_file[:-4].split('-')
    artlist = read_corpus(lang)
    if not os.path.exists(fp):
        sample_test_set_inds(artlist, ratio, seed, grp, fp)

    with open(fp, 'r') as f:
        te_inds = [int(i) for i in f]

    if grp == 'arts':
        tr_inds = [i for i in range(len(artlist)) if i not in te_inds]
        tr_arts = [artlist[i] for i in tr_inds]
        te_arts = [artlist[i] for i in te_inds]
        return tr_arts, te_arts
    else:
        pars = [p for a in artlist for p in a]
        tr_inds = [i for i in range(len(pars)) if i not in te_inds]
        tr_pars = [pars[i] for i in tr_inds]
        te_pars = [pars[i] for i in te_inds]
        return tr_pars, te_pars


def sample_test_set_inds(artlist, ratio, seed, grp, fp):
    random.seed(seed)
    sents = artlist if grp == 'arts' else [p for a in artlist for p in a]
    test_art_inds = random.sample(range(len(sents)), k=int(len(sents) * float(ratio)))
    with open(fp, 'w') as f:
        for i in test_art_inds:
            f.write(str(i) + '\n')

