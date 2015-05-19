import corpus
import linkage

from collections import defaultdict

_ENDs = ('?', '”', '…', '──', '、', '。', '」', '！', '，', '：', '；', '？')

_BEFORE, _BEGIN, _INSIDE, _AFTER = range(4)


def is_argument_label(l):
    return l in (_BEGIN, _INSIDE)


def get_argument_offsets(arg_indices):
    offsets = []
    for indices in arg_indices:
        for i, idx in enumerate(indices):
            assert(i == 0 or indices[i - 1] + 1 == idx)
        offsets.append((indices[0], indices[-1] + 1))

    for prev, curr in zip(offsets, offsets[1:]):
        assert(prev[1] == curr[0])
    return offsets


def get_EDU_offsets(tokens):
    start = 0
    tlen = len(tokens)
    offsets = []
    for i, token in enumerate(tokens):
        if token in _ENDs and (i + 1 == tlen or tokens[i + 1] not in _ENDs):
            offsets.append((start, i + 1))
            start = i + 1
            if start == tlen:
                break
    else:
        offsets.append((start, tlen))
    return offsets


def get_EDU_labels(EDUs, arg_indices):
    labels = []
    arg_indices_ = list(reversed(arg_indices))
    for start, end in EDUs:
        if len(arg_indices_) == 0:
            labels.append(_AFTER)
        elif arg_indices_[-1][0] > start:
            assert(arg_indices_[-1][-1] >= end)
            labels.append(_BEFORE)
        else:
            if arg_indices_[-1][0] == start:
                labels.append(_BEGIN)
            else:
                labels.append(_INSIDE)

            if arg_indices_[-1][-1] + 1 == end:
                arg_indices_.pop()
    check_continuity(labels)
    return labels


def get_end_index(span, tokens):
    start, end = span
    end -= 1
    while end >= start:
        if tokens[end] not in _ENDs:
            return end
        else:
            end -= 1
    return start


def collect_cnnct_positions(c_indices):
    start_indices = set()
    end_indices = set()

    for indices in c_indices:
        start_indices.add(indices[0])
        end_indices.add(indices[-1])
    return start_indices, end_indices


def connective_features(tfeatures, EDUs, c_indices):
    c_start = c_indices[0][0]
    c_end = c_indices[-1][-1]
    c_start_EDU = None
    c_end_EDU = None
    for i, edu in enumerate(EDUs):
        start, end = edu
        if start > c_end:
            break
        c_end_EDU = i
        if end > c_start and c_start_EDU is None:
            c_start_EDU = i

    for i, s in enumerate(tfeatures):
        if i < c_start_EDU:
            s.add('BEFORE_CNNCT')
            s.add('BEFORE_CNNCT-{}'.format(c_start_EDU - i))
        if i > c_end_EDU:
            s.add('AFTER_CNNCT')
            s.add('AFTER_CNNCT-{}'.format(i - c_end_EDU))


def path_features(s, parsed, from_pos, to_pos):
    lca = -1
    for i, (a, b) in enumerate(zip(from_pos, to_pos)):
        if a == b:
            lca = i
        else:
            break

    fname = 'PATH-'
    items = []
    for i in range(len(from_pos), lca, -1):
        items.append(parsed[from_pos[:i]].label())
    fname += '^'.join(items)

    items = []
    for i in range(lca + 2, len(to_pos) + 1):
        items.append(parsed[to_pos[:i]].label())
    if len(items) > 0:
        fname += 'v' + 'v'.join(items)

    s.add(fname)


def extract_EDU_features(EDUs, tokens, pos_tokens, parsed, arg):
    cnnct, rtype, c_indices, a_indices = arg
    cnncts = cnnct.split('-')
    tlen = len(EDUs)
    tlabels = get_EDU_labels(EDUs, a_indices)
    arg_offsets = get_argument_offsets(a_indices)
    assert(len(arg_offsets) == tlabels.count(_BEGIN))
    tfeatures = [set() for _ in range(tlen)]

    # set features
    for s in tfeatures:
        s.add('CNNCT-' + cnnct)
        s.add('RTYPE-{}'.format(rtype))
        s.add('CNNCT_NUM:{}'.format(cnnct.count('-') + 1))

    connective_features(tfeatures, EDUs, c_indices)
    c_spans = []
    if len(c_indices) > 1:
        c_spans.append((c_indices[0][0], c_indices[-1][-1]))
    for indices in c_indices:
        c_spans.append((indices[0], indices[-1]))

    c_poses = []
    for c_start, c_end in c_spans:
        _, c_pos = corpus.ParseHelper.self_category(
            parsed, [c_start, c_end], exact=False, positions=True)
        c_poses.append(c_pos)

    sindices, eindices = collect_cnnct_positions(c_indices)
    for i, s in enumerate(tfeatures):
        span = EDUs[i]
        start = span[0]
        end = get_end_index(span, tokens)
        if start in sindices:
            s.add('CNNCT_START')
        if end in sindices:
            s.add('CNNCT_END')
        for indices, cnnct_comp in zip(c_indices, cnncts):
            if span[0] <= indices[0] < span[1]:
                s.add('HAS_CONNCT')
                break

        for j in range(span[0], span[1]):
            t = tokens[j]
            pt = pos_tokens[j]
            s.add(
                'TOKEN_POS-{}'.format(
                    pt.replace('\\', r'\\').replace(':', r'\:')))

        # self
        me, me_pos = corpus.ParseHelper.self_category(
            parsed, [start, end], exact=False, positions=True)
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

        s.add('CONTEXT-{}-{}-{}-{}'.format(sf, p, lsb, rsb))

        for c_pos in c_poses:
            path_features(s, parsed, c_pos, me_pos)

    return tlabels, tfeatures


def check_continuity(labels):
    last = _BEFORE
    total_transit = 0
    for l in labels:
        if l == _BEGIN:
            # first begin
            if not is_argument_label(last):
                assert(last == _BEFORE)
                total_transit += 1
        # middle
        elif l == _INSIDE:
            assert(is_argument_label(last))
        # end of arguments
        elif not is_argument_label(l) and is_argument_label(last):
            assert(l == _AFTER)
            total_transit += 1
        # continue outside
        else:
            assert(last == l)
        last = l
    if is_argument_label(last):
        total_transit += 1
    assert(total_transit == 2)


def extract_features(tokens, pos_tokens, arg):
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
    for lst in c_indices:
        for idx in lst:
            tfeatures[idx].add('IS_CNNCT')
    for i, (s, t, pt) in enumerate(zip(tfeatures, tokens, pos_tokens)):
        s.add(t.replace('\\', r'\\').replace(':', r'\:'))
        s.add('POS-{}'.format(pt.split('/')[-1]))
        if t in _ENDs:
            s.add('IS_END')
            if i + 1 < tlen:
                tfeatures[i + 1].add('PREV_END')

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
                cnnct_token_indices.append(tuple(lst))
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
