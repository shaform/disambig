"""Train word probs"""
import argparse

import corpus
import features
import evaluate

from collections import defaultdict
from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_features', required=True,
                        help='word features file')
    parser.add_argument('--word_ambig', required=True,
                        help='word ambiguity file')
    parser.add_argument('--word_count', required=True,
                        help='word count file')
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--output', required=True,
                        help='output file')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')

    return parser.parse_args()


def predict(args):
    i, X, Y, Xt = args
    lr = RandomForestClassifier()
    lr = SVC()
    lr = LinearSVC()
    lr = GaussianNB()
    lr = DecisionTreeClassifier()
    lr = LogisticRegression()
    lr.fit(X, Y)
    Yt = lr.predict(Xt)
    print('completed training word probability for fold', i, '...')
    return Yt


def train_word_probs(fhelper, feature_tbl, ambig_path, count_path,
                     check_accuracy=False):
    word_probs = {}

    word_ambig = None
    if check_accuracy and ambig_path is not None:
        word_ambig = evaluate.WordAmbig(ambig_path)

    word_count = None
    if check_accuracy and count_path is not None:
        word_count = evaluate.WordCount(count_path)

    stats = evaluate.FoldStats()

    # extract training data
    num_of_folds = len(fhelper.folds())
    print('\nextract data for {} folds'.format(num_of_folds))
    input_data = []
    helper_data = []
    for i in fhelper.folds():
        _, X, Y = fhelper.features(fhelper.train_set(i), feature_tbl)

        labels, Xt, Yt_truth = fhelper.features(
            fhelper.test_set(i), feature_tbl)

        input_data.append((i, X, Y, Xt))
        helper_data.append({
            'i': i,
            'labels': labels,
            'Yt_truth': Yt_truth
        })

    # spawn processes to train
    print('\nstart training')
    with Pool(num_of_folds) as p:
        results = p.map(predict, input_data)

    # join all data
    for helper, Yt in zip(helper_data, results):
        i = helper['i']
        labels = helper['labels']
        Yt_truth = helper['Yt_truth']

        truth_count = None
        if word_ambig is not None:
            truth_count = word_ambig.count_fold(fhelper.test_set(i))

        total_count = None
        if word_count is not None:
            total_count = word_count.count_fold(fhelper.test_set(i))

        if check_accuracy:
            stats.compute_fold(labels, Yt, Yt_truth,
                               truth_count=truth_count,
                               total_count=total_count)

        for label, y_truth, y in zip(labels, Yt_truth, Yt):
            word_probs[label] = (y_truth, y)

    if check_accuracy:
        print('== done ==')

        if word_ambig is not None:
            truth_count = len(word_ambig)
        else:
            truth_count = None

        stats.print_total(truth_count=truth_count)

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
        count_path=args.word_count,
        check_accuracy=args.check_accuracy)

    output_file(args.output, word_probs)

if __name__ == '__main__':
    main()
