

import argparse
import os

import torch as t

from experiments.src.data.sampling import get_ind_pairs, get_training_and_test_sents, get_ranges
from experiments.src.data.datasets import SentPairDataset, SequenceDataset, PositionDataset
from experiments.src.data import encoder
from experiments.src.data.encoder import Encoder
from experiments.src.trainer import Trainer, log_scores, plot, check_early_stop
from experiments.src.models import pairnets, seq2seq
from experiments.src.vars import MODEL_DIR, OPTIMS, DEVICE
from experiments.src.utils import get_model_name, read_args


parser = argparse.ArgumentParser()

# general params
parser.add_argument('--task', nargs='?', default='order')       # pw / order / pos, determines dev dataset
parser.add_argument('--dev_ratio', nargs='?', type=float, default=0.2)
parser.add_argument('--cv_folds', nargs='?', type=int, default=5)
parser.add_argument('--seed', nargs='?', type=int)
parser.add_argument('--test_ind_file', nargs='?', default='english-0.2-1-arts.txt')     # lang-testsplit-seed-arts/pars
parser.add_argument('--sample_pars', nargs='*', default=['split=pars', 'tr-sample=half', 'dev-sample=all'])
parser.add_argument('--plot', action='store_true')
parser.add_argument('--final', action='store_true')
# params for getting pre-trained embeddings
parser.add_argument('--emb_pars', nargs='*', default=['enc=sbert', 'type=sents', 'dim=H', 'len=W', 'model=statfi'])
parser.add_argument('--model', nargs='?', default='PointerNet')
# training params
parser.add_argument('--n_epochs', nargs='?', type=int, default=20)
parser.add_argument('--batch_size', nargs='?', type=int, default=32)
parser.add_argument('--loss_fn', nargs='?')
parser.add_argument('--optim', nargs='?', default='adadelta')
parser.add_argument('--opt_params', nargs='*', default=['default'])
parser.add_argument('--early_stop', nargs='?')
# gen. NN params
parser.add_argument('--act_fns', nargs='*')
parser.add_argument('--model_pars', nargs='*')
parser.add_argument('--merge', nargs='*', default=['fn=bln', 'dim=100', 'pos=0'])
parser.add_argument('--h_units', nargs='*', type=int, default=[100])
parser.add_argument('--dropout', nargs='?', type=float, default=0.5)

args = parser.parse_args()

load_embs = False

# dataset divided to train / test articles
artlist, te_arts = get_training_and_test_sents(args.test_ind_file)
parlist, te_pars = [p for a in artlist for p in a], [p for a in te_arts for p in a]
sentlist, te_sents = [s for p in parlist for s in p], [s for p in te_pars for s in p]

# update args
args = read_args(args)

num_workers = 0

rngs = get_ranges(artlist)

# filter out pars with one sentence for ordering task
if args.task == 'order' and args.sample_pars['split'] == 'pars':
    rngs = {a_rn: [prn for prn in p_rns if prn[1] - prn[0] > 1] for a_rn, p_rns in rngs.items()}
    parlist = [par for par in parlist if len(par) > 1]

seqlist = artlist if args.sample_pars['split'] == 'arts' else parlist
n = len(seqlist)
n_dev = int(args.dev_ratio * n)
# if training with whole datast and saving model to file
if args.final:
    n_dev = 0
    args.cv_folds = 1

n_tr = n - n_dev
# get rand inds for tr / dev split
if args.seed:
    t.manual_seed(seed=args.seed)
rand_inds = t.randperm(n=n).tolist()

# init. encoder
enc = Encoder(emb_pars=args.emb_pars)
if enc.name in ['word2vec', 'glove', 'fasttext']:
    load_embs = True
# load embeddings, or set to None
# get also embs for the test set at the same time
embeddings, vocab = enc.load_embs(sentlist + te_sents) if load_embs else (None, None)

all_scores = {}
for fold in range(args.cv_folds):
    st_dev, end_dev = fold * n_dev, (fold + 1) * n_dev
    dev_inds = rand_inds[st_dev:end_dev]
    tr_inds = rand_inds[:st_dev] + rand_inds[end_dev:]
    assert len(tr_inds) == n_tr and len(dev_inds) == n_dev
    # get ranges
    a_rns, p_rns = list(rngs.keys()), [p for a in rngs.values() for p in a]

    model = getattr(pairnets, args.model) if args.model_type == 'pw' else getattr(seq2seq, args.model)
    n_sents = len(sentlist) if not load_embs else len(sentlist + te_sents)  # for Emb layer if using rand enc
    model = model(params=args, n_sents=n_sents, embeddings=embeddings)
    model = model.to(DEVICE)

    # get optimiser with params
    if args.opt_params[0] == 'default':
        opt = OPTIMS[args.optim](model.parameters())
    else:
        opt_params = {par.split('=')[0]: float(par.split('=')[1]) for par in args.opt_params}
        opt = OPTIMS[args.optim](model.parameters(), **opt_params)

        # sample is a range of sent indices (of sents in sentlist)
    tr_samples = [p_rns[i] for i in tr_inds] if args.sample_pars['split'] == 'pars' else [a_rns[i] for i in tr_inds]
    dev_samples = [p_rns[i] for i in dev_inds] if args.sample_pars['split'] == 'pars' else [a_rns[i] for i in dev_inds]
    if args.model_type == 'pw':
        # make sure split is always between paragraphs / articles (flp)
        # sample is a pair of indices (of sents in sentlist)
        tr_samples = get_ind_pairs(tr_inds, rngs, args.sample_pars['tr-sample'], args.sample_pars['split'])
        tr_dset = SentPairDataset(sentlist, ind_pairs=tr_samples, params=args, encoder=enc, vocab=vocab)
    else:
        if args.model_type == 'order':
            tr_dset = SequenceDataset(sentlist, iseqs=tr_samples, encoder=enc, params=args, seed=args.seed,
                                      seq_type=args.sample_pars['split'], vocab=vocab)
        else:
            tr_dset = PositionDataset(sentlist, iseqs=tr_samples, encoder=enc, params=args, vocab=vocab)

    if args.task == 'order':
        dev_dset = SequenceDataset(sentlist, iseqs=dev_samples, encoder=enc, params=args,
                                   seed=args.seed, seq_type=args.sample_pars['split'], vocab=vocab)
        scores = {'loss': [], 'tau': [], 'PMR': [], 'PAcc': []}
    elif args.task == 'pos':
        dev_dset = PositionDataset(sentlist, iseqs=dev_samples, encoder=enc, params=args, vocab=vocab)
        scores = {'loss': [], 'P': [], 'R': [], 'F1': [], 'A': []}      # weighted average for mulitclass case
        dev_samples = dev_dset.sent_inds
    else:   # pw
        dev_samples = get_ind_pairs(dev_inds, rngs, args.sample_pars['dev-sample'], args.sample_pars['split'])
        n_trp, n_devp = len(tr_samples), len(dev_samples)
        print('tr_pairs: #{} ({:.2f}%), dev_pairs: #{} ({:.2f}%)'.format(n_trp, n_trp / (n_trp + n_devp), n_devp,
                                                                         n_devp / (n_trp + n_devp)))
        scores = {'loss': [], 'P': [], 'R': [], 'F1': [], 'A': [], 'AP': []}
        dev_dset = SentPairDataset(sentlist, ind_pairs=dev_samples, params=args, encoder=enc, vocab=vocab)

    tr_collate = getattr(encoder, 'collate_' + args.model_type) if args.emb_pars['len'] == 'W' else None
    dev_collate = getattr(encoder, 'collate_' + args.task) if args.task != 'pw' or args.emb_pars['len'] == 'W' else None

    tr_loader = t.utils.data.DataLoader(dataset=tr_dset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                        num_workers=num_workers, collate_fn=tr_collate)
    if not args.final:
        # must be shuffle=False in order to get eyeball samples correctly
        # drop last if kernel size greater than last batch
        last_bs = n_dev % args.batch_size
        drop_last = False
        # drop_last = True if 'CNN' in args.model and \
        #                   any(int(k.split('x')[1]) > last_bs for k in args.model_pars['krn']) else False
        dev_loader = t.utils.data.DataLoader(dataset=dev_dset, batch_size=args.batch_size, shuffle=False,
                                             drop_last=drop_last, num_workers=num_workers, collate_fn=dev_collate)
    else:
        dev_loader = None

    trainer = Trainer(tr_loader, dev_loader, dev_samples, scores, opt, args.loss_fn, sentlist, args)

    # loop over epochs here, save to file after each epoch (if args.final)
    early_stop = False
    prev_path = ''
    for epoch in range(args.n_epochs):
        # test for early stopping (loss increasing / metric decreasing for 2 consec. epochs)
        if args.early_stop and len(scores[args.early_stop]) > 10:
            early_stop = check_early_stop(args.early_stop, scores)

        model = trainer.train(model, epoch)

        # stop if early stop condition true
        if early_stop:
            break

        # save last model, cannot save best without testing
        if args.final:

            if os.path.exists(prev_path):
                os.remove(prev_path)

            model_fname = get_model_name(args)
            model_path = os.path.join(MODEL_DIR, args.model, model_fname)  # model path
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            # save model to file (outside of git dir)

            args.epoch = epoch + 1
            args.model_state_dict = model.state_dict()
            args.optimizer_state_dict = opt.state_dict()
            chkpt = vars(args)
            t.save(chkpt, model_path)
            prev_path = model_path
            print('Model saved to path: {}'.format(model_path))

    all_scores[fold] = trainer.scores


if not args.final:
    log_scores(all_scores, args)
    if args.plot:
        plot(trainer.plot_iters, all_scores, trainer.model_fname)
