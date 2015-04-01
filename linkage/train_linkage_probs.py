"""Train linkage probs"""
import argparse

import corpus
import evaluate
import features
import linkage

from collections import defaultdict

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--linkage_features', required=True,
                        help='linkage features file')
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--output', required=True,
                        help='output file')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')

    return parser.parse_args()


def train_linkage_probs(fhelper, feature_tbl, linkage_counts,
                        check_accuracy=False):
    print('training linkage probability')

    lr = SVR(C=1.0, epsilon=0.2)
    # lr = SVC(kernel='linear')
    # lr = LinearSVC()
    linkage_probs = {}

    stats = evaluate.FoldStats(threshold=0.7)
    for i in fhelper.folds():
        print('\ntraining for fold', i, '...')
        X = []
        Y = []

        for label in fhelper.train_set(i):
            for _, y, x in feature_tbl[label]:
                if y == 1:
                    X.extend([x for _ in range(4)])
                    Y.extend([y for _ in range(4)])
                else:
                    X.append(x)
                    Y.append(y)

        Xt = []
        Yt_truth = []
        labels = []
        for label in fhelper.test_set(i):
            for l, y, x in feature_tbl[label]:
                labels.append((label, tuple(l)))
                Xt.append(x)
                Yt_truth.append(y)

        lr.fit(X, Y)
        Yt = lr.predict(Xt)

        if check_accuracy:
            stats.compute_fold(labels, Yt, Yt_truth)

        for label, y_truth, y in zip(labels, Yt_truth, Yt):
            linkage_probs[label] = (y_truth, y)

    if check_accuracy:
        print('== done ==')

        stats.print_total(truth_count=linkage_counts)

    print('linkage trained:', len(linkage_probs))
    return linkage_probs


def count_linkage(path):
    with open(path, 'r') as f:
        return len([l for l in f])


def output_file(path, linkage_probs):
    with open(path, 'w') as f:
        for (label, words), (truth, prob) in sorted(linkage_probs.items()):
            f.write('{}\t{}\t{}\t{}\n'.format(
                label, '-'.join(words), truth, prob))


def main():
    args = process_commands()

    fhelper = corpus.FoldsHelper(args.folds)
    feature_tbl = features.load_features_table(
        args.linkage_features, lambda x: tuple(x.split('-')))
    linkage_counts = count_linkage(args.linkage)

    linkage_probs = train_linkage_probs(fhelper, feature_tbl, linkage_counts,
                                        check_accuracy=args.check_accuracy)

    output_file(args.output, linkage_probs)

if __name__ == '__main__':
    main()
