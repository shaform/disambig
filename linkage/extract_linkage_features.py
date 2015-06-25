"""Extract features for linkages in corpus"""
import argparse
import re

import numpy as np

import evaluate
import features
import linkage
import corpus

from collections import defaultdict

from sklearn import svm
from sklearn import cross_validation

PN = 'PN'
POS = 'POS'
NUM = 'NUM'
GLOVE = 'GLOVE'


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connective', required=True,
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
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--output', required=True,
                        help='output file')
    parser.add_argument('--perfect_output',
                        help='perfect output file')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')
    parser.add_argument('--select',
                        help='only select a feature set',
                        choices=(PN, POS, NUM, GLOVE))

    return parser.parse_args()


FILTER_SET = {
    PN: ('self_', 'parent_', 'left_sb_', 'right_sb_'),
    POS: ('in_pos_', 'left_pos_', 'right_pos_'),
    NUM: ('num_of_overlapped', 'num_of_crossed',
          'num_left_boundary', 'num_right_boundary',
          'num_dist', 'num_dist_to_boundary',
          'num_geo_mean',
          'num_of_words_'),
}


for k, v in FILTER_SET.items():
    FILTER_SET[k] = re.compile('({})'.format('|'.join(v)))


def get_linkage_features(corpus_file, detector, vectors, truth, *,
                         select=None, perfect=False):
    print('get linkage features')
    cands = []
    Y = []
    X = []
    Xext = []

    counter = evaluate.ProgressCounter()
    correct_count = 0
    for label, tokens in corpus_file.corpus.items():
        counter.step()

        if perfect:
            truth_connectives = truth[label]
        else:
            truth_connectives = None

        pos_tokens = corpus_file.pos_corpus[label]
        parsed = corpus_file.parse_corpus[label]

        # grab overlapped statistics
        overlapped_at = defaultdict(set)
        crossed_at = defaultdict(set)
        for _, indices in detector.all_tokens(tokens,
                                              continuous=True,
                                              cross=False,
                                              truth=truth_connectives):
            for x in indices:
                token_indices = list(linkage.token_indices(x))
                for i in token_indices:
                    overlapped_at[i].add(indices)

            for x, y in zip(indices, indices[1:]):
                x_indices = list(linkage.token_indices(x))
                y_indices = list(linkage.token_indices(y))

                for i in range(x_indices[-1] + 1, y_indices[0]):
                    crossed_at[i].add(indices)

        # start construct features
        for tags, indices in detector.all_tokens(tokens,
                                                 continuous=True,
                                                 cross=False,
                                                 truth=truth_connectives):
            t_indices = linkage.list_of_token_indices(indices)
            feature_vector = defaultdict(int)

            gm = features.geometric_dists_mean(t_indices)
            feature_vector['num_geo_mean'] = gm

            token_vectors = []
            left_vectors = []
            right_vectors = []

            dist_to_boundary = len(tokens)
            overlapped = set()
            crossed = set()
            l_index = len(tokens)
            r_index = 0
            for x, token_indices in zip(indices, t_indices):
                dist_to_bundary = min(features.min_boundary(
                    token_indices[0], token_indices[-1], tokens),
                    dist_to_boundary)

                for i in token_indices:
                    token_vectors.append(
                        vectors.get(pos_tokens[i]))
                    overlapped |= overlapped_at[i]
                    crossed |= crossed_at[i]
                    l_index = min(l_index, i)
                    r_index = max(r_index, i)

                # left vector
                left_vectors.append(features.get_vector(
                    l_index - 1, pos_tokens, vectors))

                # right vector
                right_vectors.append(features.get_vector(
                    r_index + 1, pos_tokens, vectors))

                # POS
                features.POS_feature_set(
                    feature_vector,
                    token_indices,
                    pos_tokens)

            # GLOVE

            token_vector = np.mean(token_vectors, axis=0)
            left_vector = np.mean(left_vectors, axis=0)
            right_vector = np.mean(right_vectors, axis=0)

            Xext.append(np.concatenate((token_vector,
                                        left_vector,
                                        right_vector)))

            # NUM

            feature_vector['num_of_overlapped'] = len(overlapped)
            feature_vector['num_of_crossed'] = len(crossed)

            feature_vector['num_of_words_{}'.format(len(indices))] = 1

            # dist features
            feature_vector['num_dist'] = r_index - l_index

            # boundary features
            feature_vector['num_dist_to_boundary'] = dist_to_boundary

            lbound, rbound = features.lr_boundary(
                l_index, r_index, tokens)

            feature_vector['num_left_boundary'] = lbound
            feature_vector['num_right_boundary'] = rbound

            # P & N
            for token_indices in t_indices:
                l_l_index = token_indices[0]
                l_r_index = token_indices[-1]
                features.PN_feature_set(
                    feature_vector, parsed, l_l_index, l_r_index)
            # features.PN_feature_set(
            #     feature_vector, parsed, l_index, r_index)

            X.append(feature_vector)
            if indices in truth[label]:
                Y.append(1)
                correct_count += 1
            else:
                Y.append(0)

            cands.append((label, indices))

    if select == GLOVE:
        X = Xext
    elif select is None:
        X = features.transform_features(X, Xext)
    else:
        # PN, POS, NUM
        features.filter_features(X, FILTER_SET[select])
        X = features.transform_features(X)

    print('detect {} correct linkages'.format(correct_count))

    return cands, Y, X


def check_accuracy(X, Y):
    print('training SVM for accuracy checking')
    clf = svm.SVC(kernel='linear', C=1)
    print('cross validation')
    scores = cross_validation.cross_val_score(
        clf, X, Y, cv=5, scoring='f1')
    print('F1', scores)


def output_file(path, cands, Y, X):
    with open(path, 'w') as f:
        for (label, indices), y, x in sorted(zip(cands, Y, X)):
            f.write('{}\t{}\t{}\t{}\n'.format(
                label, '-'.join(indices), y, ' '.join(str(n) for n in x)))


def main():
    args = process_commands()

    # loading data

    truth = linkage.LinkageFile(args.linkage)
    fhelper = corpus.FoldsHelper(args.folds)
    detector = linkage.LinkageDetector(args.connective)
    vectors = corpus.VectorFile(args.vector)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)

    print('process file')

    cands, Y, X = get_linkage_features(corpus_file,
                                       detector,
                                       vectors,
                                       truth,
                                       select=args.select)

    output_file(args.output, cands, Y, X)

    if args.check_accuracy:
        check_accuracy(X, Y)

    # extract perfect features for sense experiments

    if args.perfect_output:
        print('process perfect file')

        cands, Y, X = get_linkage_features(corpus_file,
                                           detector,
                                           vectors,
                                           truth,
                                           select=args.select,
                                           perfect=True)

        output_file(args.perfect_output, cands, Y, X)

        if args.check_accuracy:
            check_accuracy(X, Y)

if __name__ == '__main__':
    main()
