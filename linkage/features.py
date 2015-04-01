"""Compute linkage features for use"""
import re

import numpy as np

from collections import defaultdict


def geometric_dists_mean(token_indices_list):
    dists = list(word_dists(token_indices_list))
    return geometric_mean(dists)


def geometric_mean(xs):
    if len(xs) == 0:
        return 0
    elif len(xs) == 1:
        return xs[0]
    else:
        mean = 1
        for x in xs:
            mean *= x
        mean **= (1 / len(xs))
        return mean


def lr_boundary(left, right, tokens):

    for l_offset in range(1, len(tokens) + 1):
        left -= 1
        if left < 0 or re.search(r'\W', tokens[left]) is not None:
            break

    for r_offset in range(1, len(tokens) + 1):
        right += 1
        if right == len(tokens) or re.search(r'\W', tokens[right]) is not None:
            break

    return l_offset, r_offset


def min_boundary(left, right, tokens):
    offset = 0
    while True:
        offset += 1
        left -= 1
        right += 1

        if (left < 0 or right == len(tokens) or
                re.search(r'\W', tokens[left]) is not None or
                re.search(r'\W', tokens[right]) is not None):
            return offset


def word_dists(token_indices_list):
    for a, b in zip(token_indices_list, token_indices_list[1:]):
        d = b[0] - a[-1]
        if d < 0:
            d = 0
        yield d


def token_offsets(indices):
    return indices[0], indices[-1]


def word_offsets(token_indices):
    return token_indices[0][0], token_indices[-1][-1]


def get_vector(i, pos_tokens, vectors):
    if i < 0 or i >= len(pos_tokens):
        return vectors.zeros()
    else:
        return vectors.get(pos_tokens[i])


def get_POS(s):
    return s.split('/')[1]


def load_features_table(path, tlabel_transform=lambda x: x):
    print('loading features table')

    feature_tbl = defaultdict(list)
    with open(path, 'r') as f:
        for l in f:
            label, tlabel, y, raw_features = l.rstrip().split('\t')
            feature_vector = np.array(list(map(float, raw_features.split())))
            feature_tbl[label].append((tlabel_transform(tlabel),
                                       int(y),
                                       feature_vector))

    return feature_tbl
