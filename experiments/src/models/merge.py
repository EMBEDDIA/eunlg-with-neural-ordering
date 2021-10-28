

"""
Functions for merging two embeddings into one.
"""

import torch as t


def concatenate(left, right):
    return t.cat((left, right), dim=0)


def abs_diff(left, right):
    return t.abs(left - right)


def squared_diff(left, right):
    return (left - right)**2


def mean_emb(x):
    return t.mean(x, dim=0)



