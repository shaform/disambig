"""Train word probs"""
import argparse

import corpus
import features
import evaluate

from collections import defaultdict

from sklearn.linear_model import LogisticRegression


TOTAL_WORDS = 12523


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_features', required=True,
                        help='word features file')
    parser.add_argument('--word_ambig', required=True,
                        help='word ambiguity file')
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--output', required=True,
                        help='output file')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')

    return parser.parse_args()


def train_word_probs(fhelper, feature_tbl, ambig_path, check_accuracy=False):
    lr = LogisticRegression()
    word_probs = {}

    if check_accuracy and ambig_path is not None:
        word_ambig = evaluate.WordAmbig(ambig_path)

    stats = evaluate.FoldStats()
    for i in fhelper.folds():
        print('\ntraining word probability for fold', i, '...')

        _, X, Y = fhelper.features(fhelper.train_set(i), feature_tbl)

        labels, Xt, Yt_truth = fhelper.features(
            fhelper.test_set(i), feature_tbl)

        lr.fit(X, Y)
        Yt = lr.predict(Xt)

        truth_count_for_fold = None
        if word_ambig is not None:
            truth_count_for_fold = word_ambig.count_fold(fhelper.test_set(i))

        if check_accuracy:
            stats.compute_fold(labels, Yt, Yt_truth,
                               truth_count=truth_count_for_fold)

        for label, y_truth, y in zip(labels, Yt_truth, Yt):
            word_probs[label] = (y_truth, y)

    if check_accuracy:
        print('== done ==')

        if word_ambig is not None:
            truth_count = len(word_ambig)
        else:
            truth_count = None

        stats.print_total(truth_count=truth_count, total=TOTAL_WORDS)

        if word_ambig is not None:
            stats.print_distribution(word_ambig)

    print('word trained:', len(word_probs))
    return word_probs


def output_file(path, word_probs):
    with open(path, 'w') as f:
        for (label, indices), (truth, prob) in sorted(word_probs.items()):
            f.write('{}\t{}\t{}\t{}\n'.format(label, indices, truth, prob))


def main():
    args = process_commands()

    # loading data

    fhelper = corpus.FoldsHelper(args.folds)
    feature_tbl = features.load_features_table(args.word_features)

    word_probs = train_word_probs(
        fhelper, feature_tbl, ambig_path=args.word_ambig,
        check_accuracy=args.check_accuracy)

    output_file(args.output, word_probs)

if __name__ == '__main__':
    main()
