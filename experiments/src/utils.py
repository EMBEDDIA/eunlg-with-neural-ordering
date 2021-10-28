
from experiments.src.vars import LANGS, EMB_DIMS, LOSSES, ACTIVATIONS, MERGES, MODELS


def read_args(args):

    args.sample_pars = {p.split('=')[0]: p.split('=')[1] for p in args.sample_pars}

    emb_pars = {p.split('=')[0]: p.split('=')[1] for p in args.emb_pars}
    enc = emb_pars['enc']
    lang = args.test_ind_file.split('-')[0] if 'lang' not in emb_pars else emb_pars['lang']
    emb_pars['lang'] = lang

    emb_dim = emb_pars['dim']
    ed = EMB_DIMS[enc][lang] if enc in ['elmo', 'sbert'] else EMB_DIMS[enc]
    emb_pars['dim'] = int(emb_dim) if emb_dim != 'H' else ed
    emb_pars['len'] = int(emb_pars['len']) if emb_pars['len'] != 'W' else 'W'
    args.emb_pars = emb_pars

    model_pars = {par.split('=')[0]: par.split('=')[1] for par in args.model_pars} if args.model_pars else {}
    model_pars = {k: v.split('+') for k, v in model_pars.items()}
    args.model_pars = model_pars

    if args.cv_folds > 1:
        args.dev_ratio = 1 / args.cv_folds

    args.loss_fn = LOSSES[args.loss_fn]()
    act_fns = {p.split('=')[0]: p.split('=')[1] for p in args.act_fns} if args.act_fns else {}

    act_fns['fc'] = ACTIVATIONS['relu'] if 'fc' not in act_fns else ACTIVATIONS[act_fns['fc']]
    act_fns['out'] = ACTIVATIONS['sig'] if 'out' not in act_fns else ACTIVATIONS[act_fns['out']]
    act_fns['conv'] = ACTIVATIONS['relu'] if 'conv' not in act_fns else ACTIVATIONS[act_fns['conv']]
    args.act_fns = act_fns

    merge_pars = {p.split('=')[0]: p.split('=')[1] for p in args.merge} if args.merge else {}
    merge_pars['fn'] = MERGES[merge_pars['fn']] if args.merge else None
    merge_pars['dim'] = int(merge_pars['dim']) if merge_pars and 'dim' in merge_pars else None
    merge_pos = int(merge_pars['pos']) if merge_pars and 'pos' in merge_pars else 0
    merge_pars['pos'] = len(args.h_units) if args.h_units and merge_pos > len(args.h_units) else merge_pos
    args.merge = merge_pars

    for k, v in MODELS.items():
        if args.model in v:
            args.model_type = k
            break
    return args


def get_model_name(params, ext='.pt', add_time=True):

    from datetime import datetime

    lang = params.emb_pars['lang'][:2]
    lang = 'sv' if lang == 'sw' else lang
    model = params.model
    if model == 'PointerNet':
        mod = 'ptr+LSTM'
    elif 'Pair' in model:
        mod = model.replace('Pair', 'pw+')
    else:       # pos net
        mod = 'pos+LSTM'
    now = datetime.now()
    tstamp = now.strftime('%Y%m%d_%H%M')
    name = '_'.join([tstamp, mod, lang]) + ext if add_time else '_'.join([mod, lang])
    return name


def get_model_name_all_pars(epoch, params, ext='.pt'):

    sample_pars = {p.split('=')[0]: p.split('=')[1] for p in params.sample_pars}
    if params.final and 'dev-sample' in sample_pars:
        del sample_pars['dev-sample']

    emb_pars = {par.split('=')[0]: par.split('=')[1] for par in params.emb_pars}
    lang = params.test_ind_file.split('-')[0] if 'lang' not in emb_pars else emb_pars['lang']

    enc = emb_pars['enc'].replace('word2vec', 'w2v')
    emb_type = 'snt' if emb_pars['type'] == 'sents' else 'tok'
    edim = emb_pars['dim'] if 'dim' in emb_pars else None
    e_ps = 'ep_' + '+'.join([enc, emb_type, edim]) if edim else '+'.join([enc, emb_type])

    lang = lang[:2]
    mpars = 'mdp_' + '+'.join(params.model_pars) if params.model_pars else 'mdp_def'
    sh = 'in' + params.input_shape
    fns = 'fn_' + '+'.join(params.act_fns) if params.act_fns else 'fn_def'

    mgp = 'mgp_' + '+'.join(params.merge) if params.merge else 'mgp_def'
    d = 'd' + str(params.dropout)

    bs, ne = 'bs' + str(params.batch_size), 'ne' + str(epoch)
    op, ls = params.optim[:3], params.loss_fn
    op_pars = '+'.join(params.opt_params) if 'default' not in params.opt_params else 'def'
    tr_pars = [bs, ne, op, op_pars, ls]
    hu = 'h' + '+'.join([str(n) for n in params.h_units]) if params.task == 'pw' else None
    att = 'a_' + params.att if params.att else 'a_no'
    s_pars = 'smp_' + '+'.join(list(sample_pars.values()))

    if params.task == 'pw':
        task = 'pw'
        if params.model == 'Bilinear':
            mod = 'bilin'
            model_fname = '-'.join(map(str, [task, mod, lang, e_ps, sh, fns, hu, d, *tr_pars, s_pars]))
        elif params.model == 'PairFC':
            mod = 'FC'
            model_fname = '-'.join(map(str, [task, mod, lang, e_ps, sh, mgp, fns, hu, d, *tr_pars, s_pars]))
        else:
            mod = params.model.replace('Pair', '')
            model_fname = '-'.join(map(str, [task, mod, lang, e_ps, mpars, sh, mgp, fns, hu, d, *tr_pars, att, s_pars]))
    else:
        task = 'seq'
        mod = 'LSTM+Ptr'
        model_fname = '-'.join(map(str, [task, mod, lang, e_ps, mpars, *tr_pars, att, s_pars]))

    return model_fname + ext


def get_model_params_from_name(model_name):

    """
    To be used in test.py when getting model params from model name, for initialising model calss.
    :param model_name:
    :return:
    """
    import argparse

    pars = argparse.Namespace()
    args = model_name.replace('.pt', '').split('-')      # model_name is filename with .pt
    task, mod = args[:2]

    pars.merge = None
    pars.h_units = None
    if task == 'pw':
        if mod == 'bilin':
            lang, eps, sh, fns, hu, d, bs, ne, op, ops, ls, smp = args[2:]
            pars.model = 'Bilinear'
        elif mod == 'FC':
            lang, eps, sh, mgp, fns, hu, d, bs, ne, op, ops, ls, smp = args[2:]
            pars.model = 'PairFC'
            pars.merge = mgp.replace('mgp_', '').split('+')
        else:
            lang, eps, mps, sh, mgp, fns, hu, d, bs, ne, op, ops, ls, att, smp = args[2:]
            pars.model = 'Pair' + mod
            pars.merge = mgp.replace('mgp_', '').split('+')
        pars.h_units = hu.replace('h', '').split('+')
    else:
        if mod == 'LSTM+Ptr':
            lang, eps, mps, sh, fns, bs, ne, op, ops, ls, att, smp = args[2:]
            pars.model = 'PointerNet'

    eps = eps.replace('eps_', '').split('+')
    eps[0] = eps[0].replace('w2v', 'word2vec')
    eps[1] = 'sents' if eps[1] == 'snt' else 'tokens'
    pars.emb_pars = ['enc=', 'type=', 'dim='] if len(eps) == 3 else ['enc=', 'type=']
    pars.emb_pars = [p + eps[i] for i, p in enumerate(pars.emb_pars)]
    pars.act_fns = fns.replace('fns_', '').split('+')
    pars.lang = LANGS[lang]
    pars.input_shape = sh[2:]
    pars.model_pars = mps.replace('mdp_', '').split('+')

    pars.att = att.replace('a_', '')
    pars.loss_fn = ls

    pars.batch_size = int(bs[2:])
    pars.epohcs = int(ne[2:])

    return pars

