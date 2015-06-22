import re

from collections import defaultdict


def all_indices(s, target, offset=0):
    listindex = []
    i = s.find(target, offset)
    while i >= 0:
        listindex.append(i + len(target))
        i = s.find(target, i + 1)
    return listindex

ITEM_FORMAT = '{}[{}:{}]'


def extract_item(connective, tokens, it, continuous=False):
    item = []
    text = ''
    for i in range(it, len(tokens)):
        text += tokens[i]
        if connective.startswith(text):
            item.append(ITEM_FORMAT.format(i, 0, len(tokens[i])))
            if connective == text:
                return i, item
            elif not continuous:
                break
        else:
            break
    return it, None


def offsets_to_items(offsets, tokens, text):
    items = []
    it = 0
    it_start = 0
    for start, end in offsets:
        item = []

        while end > it_start:
            it_end = it_start + len(tokens[it])

            if start < it_end:
                item.append(ITEM_FORMAT.format(
                    it,
                    max(start - it_start, 0),
                    min(end - it_start, len(tokens[it]))
                ))

            it += 1
            it_start = it_end

        items.append(item)

    return items


def items_to_tuple(items):
    return tuple(','.join(item) for item in items)


def token_indices(word):
    """Get indices of tokens in a word of linkage"""
    return (int(i.split('[')[0]) for i in word.split(','))


def list_of_token_indices(words):
    return [list(token_indices(x)) for x in words]


class LinkageFile(object):

    def __init__(self, linkage_path):
        self.linkage = defaultdict(set)
        self.linkage_with_types = set()
        self.linkage_type = defaultdict(dict)
        self.linkage_type2 = defaultdict(dict)
        self.structure_type = defaultdict(dict)
        self.type_stats = defaultdict(set)
        self.type_stats2 = defaultdict(set)
        self.type_counts = defaultdict(int)

        with open(linkage_path, 'r') as f:
            items = [l.rstrip().split('\t') for l in f]

        for plabel, words, indices, tp, tp2, sp in items:
            cnnct = tuple(indices.split('-'))
            self.linkage[plabel].add(cnnct)
            self.linkage_type[plabel][cnnct] = int(tp)
            self.linkage_type2[plabel][cnnct] = int(tp2)
            self.structure_type[plabel][cnnct] = int(sp)
            self.type_stats[words].add(tp)
            self.type_stats2[words].add(tp2)
            self.type_counts[words] += 1

        for plabel, words, indices, _, _, _ in items:
            cnnct = tuple(indices.split('-'))
            if len(self.type_stats[words]) > 1:
                self.linkage_with_types.add((plabel, cnnct))

    def internal_print_type_stats(self, type_stats):
        d = defaultdict(int)
        dinst = defaultdict(int)
        for w, s in type_stats.items():
            d[len(s)] += 1
            dinst[len(s)] += self.type_counts[w]

        print('Type stats')
        for v, c in sorted(d.items()):
            print('{}: {}'.format(v, c))

        print('Type instances stats')
        for v, c in sorted(dinst.items()):
            print('{}: {}'.format(v, c))

    def print_type_stats(self):
        print('\n1-level')
        self.internal_print_type_stats(self.type_stats)

        print('\n2-level')
        self.internal_print_type_stats(self.type_stats2)

    def all_words(self):
        """Get identities of all single words appeared."""
        words = set()
        for plabel, links in self.linkage.items():
            for link in links:
                for index in link:
                    words.add((plabel, index))

        return words

    def __getitem__(self, plabel):
        """Get linkages of a paragraph."""
        return self.linkage[plabel]


class LinkageDetector(object):

    def __init__(self, connective_path):
        """Creates likage detector by connective token file"""
        with open(connective_path, 'r') as f:
            self.connectives = {tuple(l.rstrip().split('\t')) for l in f}
            self.components = set()
            for connective in self.connectives:
                for component in connective:
                    self.components.add(component)

    def all_tokens(self, tokens, *, continuous=True, cross=False, truth=None):
        list_of_tokens = list(self.detect_by_tokens(tokens,
                                                    continuous=continuous,
                                                    cross=cross))
        if truth is not None:
            s = {indices for _, indices in list_of_tokens}
            for indices in truth:
                if indices not in s:
                    list_of_tokens.append(
                        (tuple('' for _ in range(len(indices))), indices))

        return list_of_tokens

    def detect_by_tokens(self, tokens, *, continuous=True, cross=False):
        for connective in self.connectives:
            for indices in self.extract_connective(0, connective, 0, tokens,
                                                   continuous=continuous,
                                                   cross=cross):
                yield connective, indices

    def detect_all_components(self, tokens):
        for component in self.components:
            for indices in self.extract_all_connective(0,
                                                       (component,),
                                                       0,
                                                       tokens,
                                                       ''.join(tokens)):
                yield component, indices[0]

    def detect_all(self, tokens):
        for connective in self.connectives:
            for indices in self.extract_all_connective(0, connective, 0, tokens,
                                                       ''.join(tokens)):
                yield connective, indices

    def extract_connective(self, idx, connective, it, tokens, *,
                           items=None, continuous=False, cross=False):
        """
        continuous: include continuous tokens
        cross: must cross boundary
        """
        if items is None:
            items = []

        if idx >= len(connective):
            yield items_to_tuple(items)
        else:
            for i in range(it, len(tokens)):
                offset, item = extract_item(
                    connective[idx], tokens, i, continuous)
                if item is not None:
                    items.append(item)
                    if cross:
                        while offset < len(tokens):
                            if re.search(r'\W', tokens[offset]) is not None:
                                break
                            else:
                                offset += 1
                    yield from self.extract_connective(
                        idx + 1, connective, offset + 1, tokens,
                        items=items, continuous=continuous)
                    items.pop()

    def extract_all_connective(self, idx, connective, start, tokens, text, *,
                               offsets=None):
        if offsets is None:
            offsets = []

        if idx >= len(connective):
            items = offsets_to_items(offsets, tokens, text)
            yield items_to_tuple(items)
        else:
            component = connective[idx]
            while True:
                offset = text.find(component, start)
                if offset != -1:
                    end = offset + len(component)
                    offsets.append((offset, end))
                    yield from self.extract_all_connective(
                        idx + 1, connective, end, tokens, text,
                        offsets=offsets)
                    offsets.pop()

                    start = offset + 1
                else:
                    break
