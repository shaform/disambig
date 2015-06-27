"""Train linkage probs"""
import argparse

import corpus
import evaluate
import features
import linkage

from collections import defaultdict
from multiprocessing import Pool

# regressors
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


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
    parser.add_argument('--output_classify', required=True,
                        help='output classification file')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')

    return parser.parse_args()


class LogisticRegressor():

    def __init__(self):
        self.lr = LogisticRegression()

    def fit(self, X, Y):
        self.lr.fit(X, Y)

    def predict(self, X):
        return self.lr.predict_proba(X)[:, 1]


def predict(args):
    i, X, Y, Xt = args
    lr = SVR()
    #lr = DecisionTreeRegressor()
    lr = LogisticRegressor()
    lr = LinearRegression()
    lr.fit(X, Y)
    Yt = lr.predict(Xt)
    print('completed training linkage probability for fold', i, '...')
    return Yt


def classify(args):
    i, X, Y, Xt = args
    lr = LogisticRegression()
    lr.fit(X, Y)
    Yt = lr.predict(Xt)
    print('completed training linkage classification for fold', i, '...')
    return Yt


def train_linkage_probs(fhelper, feature_tbl, linkage_counts,
                        check_accuracy=False):
    print('training linkage probability')

    probs = {}
    classes = {}

    stats = evaluate.FoldStats(threshold=0.7)
    cstats = evaluate.FoldStats()

    # extract training data
    num_of_folds = len(fhelper.folds())
    print('\nextract data for {} folds'.format(num_of_folds))
    input_data = []
    helper_data = []
    for i in fhelper.folds():
        _, X, Y = fhelper.features(
            fhelper.train_set(i), feature_tbl)

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
    with Pool(num_of_folds * 2) as p:
        results = p.map_async(predict, input_data)
        cresults = p.map_async(classify, input_data)
        results = results.get()
        cresults = cresults.get()

    # join all data
    for helper, Yt, cYt in zip(helper_data, results, cresults):
        labels = helper['labels']
        Yt_truth = helper['Yt_truth']

        if check_accuracy:
            stats.compute_fold(labels, Yt, Yt_truth)
            cstats.compute_fold(labels, cYt, Yt_truth)

        for label, y_truth, y, cy in zip(labels, Yt_truth, Yt, cYt):
            probs[label] = (y_truth, y)
            classes[label] = (y_truth, cy)

    if check_accuracy:
        print('== done ==')
        print('\n== regression test ==')
        stats.print_total(truth_count=linkage_counts)
        print('\n== classification test ==')
        cstats.print_total(truth_count=linkage_counts)

    print('linkage trained:', len(probs))
    return probs, classes


def count_linkage(path):
    with open(path, 'r') as f:
        return len([l for l in f])


def output_file(path, linkage_probs):
    print('== output to ' + path)
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

    probs, classes = train_linkage_probs(fhelper, feature_tbl, linkage_counts,
                                         check_accuracy=args.check_accuracy)

    output_file(args.output, probs)
    output_file(args.output_classify, classes)

if __name__ == '__main__':
    main()
