"""Extract features for words in corpus"""
import argparse
import re

import numpy as np

import evaluate
import features
import linkage
import corpus

from collections import defaultdict


PN = 'PN'
POS = 'POS'
NUM = 'NUM'
GLOVE = 'GLOVE'


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True,
                        help='connective file')
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')
    parser.add_argument('--corpus_pos', required=True,
                        help='pos-tagged raw corpus file')
    parser.add_argument('--corpus_parse', required=True,
                        help='syntax-parsed raw corpus file')
    parser.add_argument('--vector', required=True,
                        help='vector file')
    parser.add_argument('--output', required=True,
                        help='output file')
    parser.add_argument('--output_ambig',
                        help='output word ambiguity file')
    parser.add_argument('--select',
                        help='only select a feature set',
                        choices=(PN, POS, NUM, GLOVE))

    return parser.parse_args()


FILTER_SET = {
    PN: ('self_', 'parent_', 'left_sb_', 'right_sb_'),
    POS: ('in_pos_', 'left_pos_', 'right_pos_'),
    NUM: ('num_of_choices', 'left_boundary', 'right_boundary'),
}

for k, v in FILTER_SET.items():
    FILTER_SET[k] = re.compile('({})'.format('|'.join(v)))


def get_features(detector, corpus_file, vectors, truth, ambig_path,
                 select=None):

    cands = []
    Y = []
    X = []
    Xext = []

    total_choices = 0
    total_words = 0

    counter = evaluate.ProgressCounter()
    t_ambig = defaultdict(int)
    for l, tokens in corpus_file.corpus.items():
        counter.step()

        pos_tokens = corpus_file.pos_corpus[l]
        parsed = corpus_file.parse_corpus[l]
        features_of = defaultdict(lambda: defaultdict(int))

        for tags, poss in detector.detect_by_tokens(tokens,
                                                    continuous=True,
                                                    cross=False):
            # count ambiguity by truth
            for word in poss:
                if (l, word) not in truth:
                    break
            else:
                for word in poss:
                    t_ambig[(l, word)] += 1

            for cnnct, pos in zip(tags, poss):
                cand = (l, pos)
                feature_vector = features_of[cand]

                # how many connectives use this token
                total_choices += 1
                feature_vector['num_of_choices'] += 1

                # except for the above, only do once
                if feature_vector['num_of_choices'] == 1:
                    total_words += 1

                    indices = list(linkage.token_indices(pos))
                    l_index, r_index = features.token_offsets(indices)

                    lbound, rbound = features.lr_boundary(
                        indices[0], indices[-1], tokens)

                    feature_vector['left_boundary'] = lbound
                    feature_vector['right_boundary'] = rbound

                    token_vectors = []
                    for i in indices:
                        token_vectors.append(
                            vectors.get(pos_tokens[i]))
                    token_vector = np.mean(token_vectors, axis=0)

                    # left vector
                    left_vector = features.get_vector(
                        l_index - 1, pos_tokens, vectors)

                    # right vector
                    right_vector = features.get_vector(
                        r_index + 1, pos_tokens, vectors)
                    Xext.append(
                        np.concatenate((token_vector, left_vector,
                                        right_vector)))

                    # POS tag involved
                    for i in indices:
                        pos_tag = features.get_POS(pos_tokens[i])
                        feature_vector['in_pos_{}'.format(pos_tag)] = 1

                    # left POS
                    if l_index > 0:
                        pos_tag = features.get_POS(pos_tokens[l_index - 1])
                        feature_vector['left_pos_{}'.format(pos_tag)] = 1

                    # right POS
                    if r_index < len(pos_tokens) - 1:
                        pos_tag = features.get_POS(pos_tokens[r_index + 1])
                        feature_vector['right_pos_{}'.format(pos_tag)] = 1

                    # P & N
                    sf, p, lsb, rsb = features.PN_feature_set(
                        parsed, l_index, r_index)
                    feature_vector['self_{}'.format(sf)] = 1
                    feature_vector['parent_{}'.format(p)] = 1
                    feature_vector['left_sb_{}'.format(lsb)] = 1
                    feature_vector['right_sb_{}'.format(rsb)] = 1

                    X.append(feature_vector)
                    cands.append(cand)
                    Y.append(1 if cand in truth else 0)

    # statistics

    print('\nTotal words: {}\nTotal choices: {}'.format(
        total_words, total_choices))

    print('\nChoice distribution:')
    d = defaultdict(int)
    for x in X:
        d[x['num_of_choices']] += 1
    for n, v in sorted(d.items()):
        print('{}:\t{}'.format(n, v))

    print('\nChoice distribution in truth:')
    d = defaultdict(int)
    for x, y in zip(X, Y):
        if y == 1:
            d[x['num_of_choices']] += 1
    for n, v in sorted(d.items()):
        print('{}:\t{}'.format(n, v))

    print('\nChoice distribution in truth by truth:')
    d = defaultdict(int)
    for v in t_ambig.values():
        d[v] += 1
    for n, v in sorted(d.items()):
        print('{}:\t{}'.format(n, v))

    if ambig_path:
        print('\noutput file to {}'.format(ambig_path))
        with open(ambig_path, 'w') as f:
            for (l, word), x, y in zip(cands, X, Y):
                if y == 1:
                    f.write('{}\t{}\t{}\t1\n'.format(
                        l, word, x['num_of_choices']))

            for l, word in truth - set(cands):
                f.write('{}\t{}\t0\t0\n'.format(
                    l, word))

    if select == GLOVE:
        X = Xext
    elif select is None:
        X = features.transform_features(X, Xext)
    else:
        # PN, POS, NUM
        features.filter_features(X, FILTER_SET[select])
        X = features.transform_features(X)

    return cands, Y, X


def output_file(path, cands, Y, X):
    print('\noutput file to {}'.format(path))
    with open(path, 'w') as f:
        for (label, tlabel), y, x in sorted(zip(cands, Y, X)):
            f.write('{}\t{}\t{}\t{}\n'.format(
                label, tlabel, y, ' '.join(str(n) for n in x)))


def main():
    args = process_commands()

    # load data

    truth = linkage.LinkageFile(args.linkage).all_words()
    detector = linkage.LinkageDetector(args.tag)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    vectors = corpus.VectorFile(args.vector)

    cands, Y, X = get_features(
        detector, corpus_file, vectors, truth, args.output_ambig,
        select=args.select)

    output_file(args.output, cands, Y, X)

if __name__ == '__main__':
    main()
