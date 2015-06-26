"""Compute linkage features for use"""
import re

import numpy as np

import corpus

from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing


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


_rB = re.compile(r'[!?:;,！？：；，。]')
_rE = re.compile(r'[！？；。]')


def num_of_sentences(tokens):
    num = 0
    for t in tokens:
        if _rE.search(t) is not None:
            num += 1
    return num


def lr_boundary(left, right, tokens):

    for l_offset in range(1, len(tokens) + 1):
        left -= 1
        if left < 0 or _rB.search(tokens[left]) is not None:
            break

    for r_offset in range(1, len(tokens) + 1):
        right += 1
        if right == len(tokens) or _rB.search(tokens[right]) is not None:
            break

    return l_offset, r_offset


def min_boundary(left, right, tokens):
    offset = 0
    while True:
        offset += 1
        left -= 1
        right += 1

        if (left < 0 or right == len(tokens) or
                _rB.search(tokens[left]) is not None or
                _rB.search(tokens[right]) is not None):
            return offset


def word_skips(token_indices_list, tokens):
    for a, b in zip(token_indices_list, token_indices_list[1:]):
        d = 0
        for i in range(a[-1] + 1, b[0]):
            if _rB.search(tokens[i]) is not None:
                d += 1
        yield d


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


def POS_feature_set(feature_vector, indices, pos_tokens):
    # POS tags involved
    for i in indices:
        pos_tag = get_POS(pos_tokens[i])
        feature_vector['in_pos_{}'.format(pos_tag)] = 1

    # left, right POS tags
    l_index, r_index = token_offsets(indices)
    for idx, label in ((l_index - 1, 'left'), (r_index + 1, 'right')):
        if 0 <= idx < len(pos_tokens):
            pos_tag = get_POS(pos_tokens[idx])
            feature_vector['{}_pos_{}'.format(label, pos_tag)] = 1


def PN_feature_set(feature_vector, parsed, l_index, r_index, *,
                   full=False, cnnct=None, exact=True):
    # self
    me = corpus.ParseHelper.self_category(
        parsed, [l_index, r_index], exact=exact)
    sf = corpus.ParseHelper.label(me)

    # parent
    p = corpus.ParseHelper.label(
        corpus.ParseHelper.parent_category(me))

    # left
    lsb = corpus.ParseHelper.label(
        corpus.ParseHelper.left_category(me))

    # right
    rsb = corpus.ParseHelper.label(
        corpus.ParseHelper.right_category(me))

    components = [
        'self_{}'.format(sf),
        'parent_{}'.format(p),
        'left_sb_{}'.format(lsb),
        'right_sb_{}'.format(rsb),
    ]

    if cnnct:
        components.append('cnnct_{}'.format(cnnct))

    for x in components:
        feature_vector[x] = 1

    if full:
        for x, y in zip(components, components[1:]):
            feature_vector['{}_{}'.format(x, y)] = 1


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


def vectorize(X, scale=False):
    X = DictVectorizer().fit_transform(X).toarray()
    if scale:
        X = preprocessing.scale(X)

    return X


def transform_features(X, Xext=None):

    # check if there are numerical features
    Xnum = []
    has_num = has_no_num = False
    for x in X:
        xnum = {}
        Xnum.append(xnum)

        for key, v in list(x.items()):
            if key.startswith('num_'):
                xnum[key] = v
                del x[key]
                has_num = True
            else:
                has_no_num = True

    assert(has_num or has_no_num)

    # merge vectors
    if has_no_num:
        X = vectorize(X)

    if has_num:
        Xnum = vectorize(Xnum, True)

        if has_no_num:
            X = np.concatenate((X, Xnum), axis=1)
        else:
            X = Xnum

    # merge additional vectors
    if Xext is not None:
        X = np.concatenate((X, Xext), axis=1)

    return X


def filter_features(X, r, reverse_select=False):
    for x in X:
        for k in list(x):
            m = r.match(k) is None
            if m ^ reverse_select:
                del x[k]
