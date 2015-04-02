"""Main program for linkage experiments"""
import argparse

import numpy as np

import corpus
import evaluate
import features
import linkage

from collections import defaultdict

from sklearn.svm import SVR


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--tag', required=True,
                        help='connective token file')
    parser.add_argument('--word_ambig', required=True,
                        help='word ambiguity file')
    parser.add_argument('--word_probs', required=True,
                        help='word probability file')
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')
    parser.add_argument('--corpus_pos', required=True,
                        help='pos-tagged raw corpus file')
    parser.add_argument('--corpus_parse', required=True,
                        help='syntax-parsed raw corpus file')
    parser.add_argument('--linkage_probs', required=True,
                        help='linkage probability file')
    parser.add_argument('--perfect', action='store_true',
                        help='whether to do perfect experiment')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')

    return parser.parse_args()


def count_linkage(path):
    with open(path, 'r') as f:
        return len([l for l in f])


def load_linkage_probs(path):
    linkage_probs = {}
    with open(path, 'r') as f:
        for l in f:
            label, tlabel, truth, prob = l.split('\t')
            words = tuple(tlabel.split('-'))
            linkage_probs[(label, words)] = float(prob)
    return linkage_probs


def load_word_probs(path):
    word_probs = {}
    truth_probs = {}
    with open(path, 'r') as f:
        for l in f:
            label, indices, truth, prob = l.split('\t')
            word_probs[(label, indices)] = float(prob)
            truth_probs[(label, indices)] = int(truth)
    return word_probs, truth_probs


def compute_ranking_probs(linkage_probs, key=None):
    if key is None:
        key = lambda x: linkage_probs[x]
    ranking_probs = {}
    for label in linkage_probs:
        words = label[1]
        token_lst = linkage.list_of_token_indices(words)
        ranking_probs[label] = (
            len(words),
            key(label))

    return ranking_probs


def detect_wrong(indices, visited, crossed):
    '''
    if len(indices) > 1:
        pure_indices = tuple(next(linkage.token_indices(token))
                             for token in indices)
        for l, r in zip(pure_indices, pure_indices[1:]):
            for prev_indices in crossed:
                if any((l < pl < r < pr or
                        pl < l < pr < r)
                        for pl, pr in zip(prev_indices,
                                          prev_indices[1:])):
                    return True

        crossed.add(pure_indices)
    '''

    all_indices = set()

    for token in indices:
        token_indices = tuple(linkage.token_indices(token))

        for idx in token_indices:
            if idx in visited:
                return True
            else:
                all_indices.add(idx)

    # correct for these indices
    visited |= all_indices

    return False


def cross_validation(corpus_file, fhelper, truth, detector,
                     linkage_counts, linkage_probs, word_ambig,
                     cut, *, perfect=False):
    stats = evaluate.FoldStats(show_fold=True)
    pstats = evaluate.FoldStats(show_fold=True)
    rejected_ov = defaultdict(int)
    rejected_s = defaultdict(int)

    for i in fhelper.folds():
        print('\npredict for fold', i)

        # compute linkage statistics
        labels = []
        Yp = []
        Y = []

        # compute paragraph statistics
        plabels = list(fhelper.test_set(i))
        pYp = []
        pY = [1] * len(plabels)

        for label in plabels:
            tokens = corpus_file.corpus[label]

            if perfect:
                truth_connectives = truth[label]
            else:
                truth_connectives = None

            markers = []
            for _, indices in detector.all_tokens(tokens,
                                                  continuous=True,
                                                  cross=False,
                                                  truth=truth_connectives):
                markers.append(indices)

            markers.sort(key=lambda x: linkage_probs[(label, x)], reverse=True)

            visited = set()
            crossed = set()
            correct = 0
            for indices in markers:
                if indices in truth[label]:
                    Y.append(1)
                else:
                    Y.append(0)
                labels.append((label, indices))

                if cut((label, indices), linkage_probs):
                    Yp.append(0)
                    rejected_s[len(indices)] += Y[-1] == 1
                    continue
                if detect_wrong(indices, visited, crossed):
                    Yp.append(0)
                    rejected_ov[len(indices)] += Y[-1] == 1
                    continue

                Yp.append(1)

                if Yp[-1] == Y[-1] == 1:
                    correct += 1

                ilen = len(indices)

            if correct == len(truth[label]):
                pYp.append(1)
            else:
                pYp.append(0)

        print('\nLinkage stats:')
        stats.compute_fold(labels, Yp, Y)

        print('\nParagraph stats:')
        pstats.compute_fold(plabels, pYp, pY)

    print('== done ==')

    print('\nLinkage stats:')
    stats.print_total(truth_count=linkage_counts)
    stats.print_distribution(
        word_ambig,
        function=lambda x: {(l, w) for (l, ws) in x for w in ws})
    stats.count_by(label='length')
    print('rejected overlapped:', rejected_ov, 'rejected scores:', rejected_s)

    print('\nParagraph stats:')
    pstats.print_total()


def main():
    args = process_commands()

    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    fhelper = corpus.FoldsHelper(args.folds)
    truth = linkage.LinkageFile(args.linkage)
    words = truth.all_words()
    detector = linkage.LinkageDetector(args.tag)

    linkage_counts = count_linkage(args.linkage)
    linkage_probs = load_linkage_probs(args.linkage_probs)

    word_ambig = evaluate.WordAmbig(args.word_ambig)

    '''
    print('score model')
    cross_validation(
        corpus_file,
        fhelper,
        truth,
        detector,
        linkage_counts,
        linkage_probs,
        word_ambig,
        cut=lambda x, y: linkage_probs[x] < 0.7)
    '''

    ranking_probs = compute_ranking_probs(linkage_probs)

    if args.perfect:
        cut = lambda x, _: any((x[0], w) not in words for w in x[1])
    else:
        cut = lambda x, _: linkage_probs[x] < 0.7

    print('ranking model')
    cross_validation(
        corpus_file,
        fhelper,
        truth,
        detector,
        linkage_counts,
        ranking_probs,
        word_ambig,
        cut=cut,
        perfect=args.perfect)

    '''
    baseline_probs = compute_ranking_probs(linkage_probs, key=lambda x: 1)
    word_probs, word_truth = load_word_probs(args.word_probs)

    print('baseline model')
    cross_validation(
        corpus_file,
        fhelper,
        truth,
        detector,
        linkage_counts,
        baseline_probs,
        word_ambig,
        cut=lambda x, y: any(word_probs[(x[0], w)] < 0.5 for w in x[1]))
    '''

if __name__ == '__main__':
    main()
