"""Extract features for linkages in corpus"""
import argparse
import os
import re
import sys

import numpy as np

import evaluate
import features
import linkage
import corpus

from collections import defaultdict

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import svm
from sklearn import cross_validation


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
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--output', required=True,
                        help='output file')
    parser.add_argument('--perfect_output', required=True,
                        help='perfect output file')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')

    return parser.parse_args()


def get_linkage_features(corpus_file, detector, vectors, truth, perfect=False):
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
            feature_vector['geo_mean'] = gm
            # feature_vector['-'.join(tags)] = 1

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

                # POS tag involved
                for i in token_indices:
                    pos_tag = features.get_POS(pos_tokens[i])
                    feature_vector['in_pos_{}'.format(pos_tag)] = 1

                # neighbor POSes
                for i, side in ((token_indices[0] - 1, 'left'),
                               (token_indices[-1] + 1, 'right')):
                    if i >= 0 and i < len(pos_tokens):
                        pos_tag = features.get_POS(pos_tokens[i])
                        feature_vector[
                            '{}_out_pos_{}'.format(side, pos_tag)] = 1

            # averaged word2vec vectors
            token_vector = np.mean(token_vectors, axis=0)
            left_vector = np.mean(left_vectors, axis=0)
            right_vector = np.mean(right_vectors, axis=0)

            Xext.append(np.concatenate((token_vector,
                                        left_vector,
                                        right_vector)))

            feature_vector['num_of_overlapped'] = len(overlapped)
            feature_vector['num_of_crossed'] = len(crossed)

            feature_vector['num_of_words_{}'.format(len(indices))] = 1

            # dist features
            feature_vector['dist'] = r_index - l_index

            # boundary features
            feature_vector['dist_to_boundary'] = dist_to_boundary

            lbound, rbound = features.lr_boundary(
                l_index, r_index, tokens)

            feature_vector['left_bundary'] = lbound
            feature_vector['right_bundary'] = rbound

            # syntactic features

            # self
            me = corpus.ParseHelper.self_category(
                parsed, [l_index, r_index])
            sf = corpus.ParseHelper.label(me)
            feature_vector['self_{}'.format(sf)] = 1

            # parent
            p = corpus.ParseHelper.label(
                corpus.ParseHelper.parent_category(me))
            feature_vector['parent_{}'.format(p)] = 1

            # left
            sb = corpus.ParseHelper.label(
                corpus.ParseHelper.left_category(me))
            feature_vector['left_sb_{}'.format(sb)] = 1

            # right
            sb = corpus.ParseHelper.label(
                corpus.ParseHelper.right_category(me))
            feature_vector['right_sb_{}'.format(sb)] = 1

            X.append(feature_vector)
            if indices in truth[label]:
                Y.append(1)
                correct_count += 1
            else:
                Y.append(0)

            cands.append((label, indices))

    # transform features
    X = DictVectorizer().fit_transform(X).toarray()
    X = preprocessing.scale(X)
    X = np.concatenate((X, Xext), axis=1)

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
    detector = linkage.LinkageDetector(args.tag)
    vectors = corpus.VectorFile(args.vector)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)

    print('process file')

    cands, Y, X = get_linkage_features(corpus_file,
                                       detector,
                                       vectors,
                                       truth)

    output_file(args.output, cands, Y, X)

    if args.check_accuracy:
        check_accuracy(X, Y)

    print('process perfect file')

    cands, Y, X = get_linkage_features(corpus_file,
                                       detector,
                                       vectors,
                                       truth,
                                       perfect=True)

    output_file(args.perfect_output, cands, Y, X)

    if args.check_accuracy:
        check_accuracy(X, Y)

if __name__ == '__main__':
    main()
