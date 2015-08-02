"""
connective experiments

connective disambiguation for ambiguous connectives extracted
from large corpus
"""
import argparse

import numpy as np

import evaluate

from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfTransformer(object):

    def __init__(self, tfidfvz, start=1):
        self.tfidfvz = tfidfvz
        self.start = start

    def transform(self, text):
        dok = self.tfidfvz.transform([text]).todok()
        d = {}
        for key, value in dok.items():
            d[self.start + key[1]] = value

        return d


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='connective corpus file')
    parser.add_argument('--output', required=True,
                        help='output svm file')

#    parser.add_argument('--train', required=True,
#                        help='path to liblinear train')
#    parser.add_argument('--scale', required=True,
#                        help='path to liblinear scale')
#    parser.add_argument('--threads', type=int, default=24)
#    parser.add_argument('action', choices=('extract', 'train'))

    return parser.parse_args()


def tf_idf_vectorizer(path, *, start=1):
    tfidfvz = TfidfVectorizer(analyzer='word',
                              ngram_range=(1, 2),
                              token_pattern=r'[^ ]+')
    with open(path) as f:
        Ts = (l.split('\t')[3] for l in f)
        tfidfvz.fit(Ts)

    return tfidfvz


def to_feature_string(d):
    items = []
    for key, value in sorted(d.items()):
        items.append('{}:{}'.format(key, value))

    return ' '.join(items)


def output_features(path, output, ftr):
    print('extract features')
    counter = evaluate.ProgressCounter()

    with open(path) as f, open(output, 'w') as of:
        for l in f:
            counter.step()
            num, cnnct, sense, tokens = l.strip().split('\t')
            d = ftr.transform(tokens)
            of.write('{} {}\n'.format(sense, to_feature_string(d)))


def main():
    args = process_commands()

    tfidftr = TfidfTransformer(tf_idf_vectorizer(args.corpus))
    output_features(args.corpus, args.output, tfidftr)


if __name__ == '__main__':
    main()
