"""I/O for corpus files"""
import numpy

import argument

from collections import defaultdict

from nltk.tree import ParentedTree


def load_linking(path):
    """load linking direction dictionary for each component"""
    link = defaultdict(set)
    with open(path) as f:
        for l in f:
            cnnct, tps = l.strip().split('\t')
            for tp in tps.split('/'):
                link[cnnct.split('(')[0]].add(tp)

    return link


def load_corpus(path,
                postprocess=lambda x: x.split(),
                preprocess=lambda x: x.rstrip('\n').split('\t')):
    d = {}
    with open(path) as f:
        for l in f:
            label, tokens = preprocess(l)
            d[label] = postprocess(tokens)
    return d


def preprocess_dep_entry(l):
    label, *entries = l.rstrip('\n').split('\t')
    return label, entries


def postprocess_dep_entry(entries):
    """split each dependency parse by @@@@"""
    return [set(entry.split('@@@@')) for entry in entries]


class CorpusFile(object):

    """tokens, POS, parse tree, dependency parse for each paragraph"""

    def __init__(self, corpus_path, pos_path=None, parse_path=None,
                 dep_path=None):
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

        self.dep_corpus = {}
        if dep_path is not None:
            self.dep_corpus = load_corpus(dep_path,
                                          postprocess_dep_entry,
                                          preprocess_dep_entry,
                                          )
            for l, dp in self.dep_corpus.items():
                assert(len(dp) == len(self.edu_corpus[l]))

    def EDUs(self, label, tokens=None):
        """get text for segments"""
        if tokens is None:
            tokens = self.corpus[label]
        edu = self.edu_corpus[label]
        segments = []
        for start, end in edu:
            segments.append(' '.join(tokens[start:end]))

        return segments


class VectorFile(object):

    def __init__(self, paths):
        self.vectors = {}
        self.dim = 0
        paths = paths.split(',')
        for path in paths:
            print('load {}...'.format(path))
            with open(path, 'r') as f:
                for l in f:
                    label, *vectors = l.rstrip('\n').split(' ')
                    dim = len(vectors)
                    vectors = numpy.array(list(float(n) for n in vectors))
                    if label in self.vectors:
                        old_vec = self.vectors[label]
                    else:
                        old_vec = numpy.zeros(self.dim)
                    self.vectors[label] = numpy.concatenate((old_vec,
                                                             vectors))

            self.dim += dim
            for k, v in self.vectors.items():
                if v.size == self.dim:
                    continue
                elif v.size == self.dim - dim:
                    self.vectors[k] = numpy.concatenate((v,
                                                         numpy.zeros(dim)))
                else:
                    print('wrong vector size')
                    assert(False)

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
        """return fold numbers"""
        return sorted(self.data_folds)

    def train_set(self, fold):
        """get train set for a fold number"""
        for key, s in sorted(self.data_folds.items()):
            if key != fold:
                yield from sorted(s)

    def test_set(self, fold):
        """get test set for a fold number"""
        for key, s in sorted(self.data_folds.items()):
            if key == fold:
                yield from sorted(s)

    def features(self, data_set, feature_tbl, extend=0):
        """get features for a fold number"""
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
        """get parse tree from Stanford parse string"""
        return ParentedTree.fromstring(s)

    @staticmethod
    def self_category(root, indices, exact=True, positions=False):
        """
        exact: must dominate exactly the indices, otherwise can dominate additional tokens
        """
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
