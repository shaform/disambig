import linkage

from collections import defaultdict

_ENDs = ('?' '”', '…', '──', '、', '。', '」', '！', '，', '：', '；', '？')


def extract_features(tokens, pos_tokens, vectors, arg):
    cnnct, rtype, c_indices, a_indices = arg
    tlen = len(tokens)

    tfeatures = [set() for _ in range(tlen)]

    # set labels
    tlabels = ['PRE'] * tlen
    for idx in range(a_indices[-1][-1] + 1, tlen):
        assert(tlabels[idx] == 'PRE')
        tlabels[idx] = 'END'
    for indices in a_indices:
        for idx in indices:
            assert(tlabels[idx] == 'PRE')
            tlabels[idx] = 'I'
        assert(tlabels[idx] == 'I')
        tlabels[indices[0]] = 'B'

    # set features
    for s in tfeatures:
        s.add('CNNCT-' + cnnct)
        s.add('RTYPE-{}'.format(rtype))
    for idx in c_indices:
        tfeatures[idx].add('IS_CNNCT')
    for s, t, pt in zip(tfeatures, tokens, pos_tokens):
        s.add(t.replace('\\', r'\\').replace(':', r'\:'))
        if t in _ENDs:
            s.add('IS_END')
        for i, v in enumerate(vectors.get(pt)):
            s.add('VEC-{}:{}'.format(i, v))

    return tlabels, tfeatures


class ArgumentFile(object):

    def __init__(self, argument_path):
        self.argument = defaultdict(dict)

        with open(argument_path, 'r') as f:
            items = [l.rstrip().split('\t') for l in f]

        for plabel, cnnct, cnnct_indices, rtype, arg_indices in items:
            cnnct_token_indices = []
            for lst in linkage.list_of_token_indices(
                    cnnct_indices.split('-')):
                for i in lst:
                    cnnct_token_indices.append(i)
            cnnct_token_indices = tuple(cnnct_token_indices)
            arg_token_indices = linkage.list_of_token_indices(
                arg_indices.split('-'))

            assert(cnnct_token_indices not in self.argument[plabel])
            self.argument[plabel][cnnct_token_indices] = (
                cnnct, rtype, arg_token_indices)

    def arguments(self, plabel):
        for c_indices, (cnnct, rtype, a_indices) in self.argument[plabel].items():
            yield cnnct, rtype, c_indices, a_indices

    def __getitem__(self, plabel):
        """Get arguments of a paragraph."""
        return self.argument[plabel]
