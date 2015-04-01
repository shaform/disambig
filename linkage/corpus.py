"""I/O for corpus files"""
import numpy

from collections import defaultdict

from nltk.tree import ParentedTree


class CorpusFile(object):

    def __init__(self, corpus_path, pos_path, parse_path):
        self.corpus = {}
        self.pos_corpus = {}
        for path, d in ((corpus_path, self.corpus), (pos_path, self.pos_corpus)):
            with open(path, 'r') as f:
                for l in f:
                    label, tokens = l.rstrip('\n').split('\t')
                    tokens = tokens.split()
                    d[label] = tokens

        self.parse_corpus = {}
        with open(parse_path, 'r') as f:
            for l in f:
                label, parsed = l.rstrip('\n').split('\t')
                self.parse_corpus[label] = ParentedTree.fromstring(parsed)


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
        return self.data_folds

    def train_set(self, fold):
        for key, s in self.data_folds.items():
            if key != fold:
                yield from s

    def test_set(self, fold):
        for key, s in self.data_folds.items():
            if key == fold:
                yield from s


class ParseHelper(object):

    @staticmethod
    def parse(s):
        return ParentedTree.fromstring(s)

    @staticmethod
    def self_category(root, indices):
        l = min(indices)
        r = max(indices) + 1
        position = root.treeposition_spanning_leaves(l, r)
        node = root[position]

        for i in range(len(position)):
            t = root[position[:-i - 1]]
            if type(node) is str or len(t.leaves()) == len(node.leaves()):
                node = t
            else:
                break

        if len(node.leaves()) > len(indices):
            return None
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
