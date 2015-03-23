import re

from collections import defaultdict


def all_indices(s, target, offset=0):
    listindex = []
    i = s.find(target, offset)
    while i >= 0:
        listindex.append(i + len(target))
        i = s.find(target, i + 1)
    return listindex


def extract_item(tag, tokens, it, continuous=False):
    item = []
    text = ''
    for i in range(it, len(tokens)):
        text += tokens[i]
        if tag.startswith(text):
            item.append('{}[{}:{}]'.format(i, 0, len(tokens[i])))
            if tag == text:
                return i, item
            elif not continuous:
                break
        else:
            break
    return it, None


def token_indices(word):
    """Get indices of tokens in a word of linkage"""
    return (int(i.split('[')[0]) for i in word.split(','))


def list_of_token_indices(words):
    return [list(token_indices(x)) for x in words]


class LinkageFile(object):

    def __init__(self, linkage_path):
        self.linkage = defaultdict(set)

        with open(linkage_path, 'r') as f:
            for l in f:
                plabel, _, indices = l.rstrip().split('\t')
                self.linkage[plabel].add(tuple(indices.split('-')))

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

    def __init__(self, tag_path):
        """Creates likage detector by connective token file"""
        with open(tag_path, 'r') as f:
            self.tags = {tuple(l.rstrip().split('\t')) for l in f}

    def detect_by_tokens(self, tokens, *, continuous=True, cross=True):
        for tag in self.tags:
            for indices in self.extract_tag(0, tag, 0, tokens,
                                            continuous=continuous,
                                            cross=cross):
                yield tag, indices

    def extract_tag(self, idx, tag, it, tokens, *,
                    items=None, continuous=False, cross=False):
        """
        continuous: include continuous tokens
        cross: moust cross boundary
        """
        if items is None:
            items = []

        if idx >= len(tag):
            yield tuple(','.join(item) for item in items)
        else:
            for i in range(it, len(tokens)):
                offset, item = extract_item(
                    tag[idx], tokens, i, continuous)
                if item is not None:
                    items.append(item)
                    if cross:
                        while offset < len(tokens):
                            if re.search(r'\W', tokens[offset]) is not None:
                                break
                            else:
                                offset += 1
                    yield from self.extract_tag(
                        idx + 1, tag, offset + 1, tokens,
                        items=items, continuous=continuous)
                    items.pop()
