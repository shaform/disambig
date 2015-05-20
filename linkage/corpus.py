"""I/O for corpus files"""
import numpy

import argument

from collections import defaultdict

from nltk.tree import ParentedTree


def load_corpus(path, postprocess=lambda x: x.split()):
    d = {}
    with open(path) as f:
        for l in f:
            label, tokens = l.rstrip('\n').split('\t')
            d[label] = postprocess(tokens)
    return d


class CorpusFile(object):

    def __init__(self, corpus_path, pos_path=None, parse_path=None):
        self.corpus = load_corpus(corpus_path)
        self.pos_corpus = {}
        if pos_path is not None:
            self.pos_corpus = load_corpus(pos_path)

        self.parse_corpus = {}
        if parse_path is not None:
            self.parse_corpus = load_corpus(
                parse_path,
                lambda x: ParentedTree.fromstring(x))

        self.edu_corpus = {}
        for l, tokens in self.corpus.items():
            self.edu_corpus[l] = argument.get_EDU_offsets(tokens)


class VectorFile(object):

    def __init__(self, path):
        self.vectors = {}
        with open(path, 'r') as f:
            for l in f:
                label, *vectors = l.rstrip('\n').split(' ')
                self.dim = len(vectors)
                vectors = numpy.array(list(float(n) for n in vectors))
                self.vectors[label] = vectors

    def zeros(self):
        return numpy.zeros(self.dim)

    def get(self, s):
        return self.vectors.get(s, self.zeros())


class FoldsHelper(object):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = [l.rstrip('\n').split() for l in f]
            self.folds_dict = {plabel: int(fold) for plabel, fold in lines}
            self.data_folds = defaultdict(set)
            for plabel, i in self.folds_dict.items():
                self.data_folds[i].add(plabel)

    def folds(self):
        return sorted(self.data_folds)

    def train_set(self, fold):
        for key, s in sorted(self.data_folds.items()):
            if key != fold:
                yield from sorted(s)

    def test_set(self, fold):
        for key, s in sorted(self.data_folds.items()):
            if key == fold:
                yield from sorted(s)

    def features(self, data_set, feature_tbl, extend=0):
        labels = []
        X = []
        Y = []

        for label in data_set:
            for l, y, x in feature_tbl[label]:
                lb = (label, l)
                if y == 1 and extend > 1:
                    labels.extend([lb] * extend)
                    X.extend([x] * extend)
                    Y.extend([y] * extend)
                else:
                    labels.append(lb)
                    X.append(x)
                    Y.append(y)

        return labels, X, Y


class ParseHelper(object):

    @staticmethod
    def parse(s):
        return ParentedTree.fromstring(s)

    @staticmethod
    def self_category(root, indices, exact=True, positions=False):
        l = min(indices)
        r = max(indices) + 1
        position = root.treeposition_spanning_leaves(l, r)
        node = root[position]

        pos = position
        for i in range(len(position)):
            p = position[:-i - 1]
            t = root[p]
            if type(node) is str or len(t.leaves()) == len(node.leaves()):
                node = t
                pos = p
            else:
                break

        if exact and len(node.leaves()) > len(indices):
            node = None
            pos = None

        if positions:
            return node, pos
        else:
            return node

    @staticmethod
    def parent_category(me):
        if me is not None:
            return me.parent()
        else:
            return None

    @staticmethod
    def left_category(me):
        if me is not None:
            return me.left_sibling()
        else:
            return None

    @staticmethod
    def right_category(me):
        if me is not None:
            return me.right_sibling()
        else:
            return None

    @staticmethod
    def label(node):
        return 'None' if node is None else node.label()
