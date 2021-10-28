

import os
import nltk

from experiments.src.vars import STATFI_PATH, STATFI_FILES


def collect_pars_into_articles(pars):

    arts = []
    prev_art = []
    for par in pars:
        if par not in ['[EOF]', '[eof]']:
            prev_art += [par]
        else:
            arts += [prev_art]
            prev_art = []
    if prev_art:
        arts += [prev_art]

    return arts


def read_corpus(lang, filt_n=3):

    with open(os.path.join(STATFI_PATH, STATFI_FILES[lang]), 'r', encoding='utf-8') as f:
        pars = [line.strip() for line in f]
    pars = [par for par in pars if par != '']  # filter empty lines
    pars = [par.lower() for par in pars]

    artlist = collect_pars_into_articles(pars)
    artlist = [tokenise_pars_into_sents(a, lang=lang) for a in artlist]
    filtered = []
    for a in artlist:
        new_a = []
        for p in a:
            new_p = []
            for s in p:
                if len(tokenise_sent_into_words(s, lang=lang)) >= filt_n:
                    new_p += [s]
            new_a += [new_p]
        filtered += [new_a]
    return filtered


def tokenise_pars_into_sents(pars, lang='finnish'):
    """

    :param pars: list of strings, each string consists of 1 or more sents
    :param lang: for tokeniser
    :return:
    """

    par_sents = []
    for par in pars:
        if par == '[EOF]':
            continue
        sents = nltk.sent_tokenize(par, language=lang)
        par_sents += [sents]
    return par_sents


def tokenise_sent_into_words(sent, lang, filt=False):
    """
    Get a list of tokens from a sentence. Tokenise so that the tokens can be encoded with BERT/Elmo etc.
    :param sent: a string
    :param lang:
    :return:
    """
    toks = [tok for tok in nltk.word_tokenize(sent, language=lang)]
    if filt:
        toks = [tok for tok in toks if any(c.isalnum() for c in tok)]
    return toks



