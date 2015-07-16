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
    parser.add_argument('--word_ambig', required=True,
                        help='word ambiguity file')
    parser.add_argument('--word_count', required=True,
                        help='word count file')
    parser.add_argument('--output',
                        help='output file')
    parser.add_argument('--output_classify',
                        help='output classification file')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')
    parser.add_argument('--classifier', default='LR',
                        choices=('SVM', 'DT', 'NB', 'LR',))

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
    lr = LogisticRegressor()
    lr.fit(X, Y)
    Yt = lr.predict(Xt)
    print('completed training linkage probability for fold', i, '...')
    return Yt

global_classifier = {'key': None}


def classify(args):
    i, X, Y, Xt = args
    lr = global_classifier['key']()
    lr.fit(X, Y)
    Yt = lr.predict(Xt)
    print('completed training linkage classification for fold', i, '...')
    return Yt


def train_linkage_probs(fhelper, feature_tbl, linkage_counts,
                        ambig_path,
                        count_path,
                        check_accuracy=False,
                        train_prob=True,
                        train_classify=True):

    word_ambig = None
    if check_accuracy:
        word_ambig = evaluate.WordAmbig(ambig_path)

    word_count = None
    if check_accuracy:
        word_count = evaluate.WordCount(count_path)

    print('training linkage probability')

    probs = {}
    classes = {}

    stats = evaluate.FoldStats(show_fold=True, label='linkage stats')
    cstats = evaluate.FoldStats(show_fold=True, label='word stats')

    # extract training data
    num_of_folds = len(fhelper.folds())
    print('\nextract data for {} folds'.format(num_of_folds))
    input_data = []
    helper_data = []
    for i in fhelper.folds():
        _, X, Y = fhelper.features(
            fhelper.train_set(i), feature_tbl, extend=3)

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
        if train_prob:
            results = p.map_async(predict, input_data)
        if train_classify:
            cresults = p.map_async(classify, input_data)

        if train_prob:
            results = results.get()
        if train_classify:
            cresults = cresults.get()

        if not train_prob:
            results = cresults

        if not train_classify:
            cresults = results

    # join all data
    for helper, Yt, cYt in zip(helper_data, results, cresults):
        i = helper['i']
        labels = helper['labels']
        Yt_truth = helper['Yt_truth']

        if check_accuracy:
            truth_count = word_ambig.count_fold(fhelper.test_set(i))
            total_count = word_count.count_fold(fhelper.test_set(i))

            cclabels = []
            ccYt = []
            ccYt_truth = []

            predictions = set()
            for (l, indices_lst), cy in zip(labels, cYt):
                if cy > 0.5:
                    for indices in indices_lst:
                        predictions.add((l, indices))

            for label in predictions:
                cclabels.append(label)
                ccYt.append(1)
                cy = 1 if label in word_ambig.ambig else 0
                ccYt_truth.append(cy)

            stats.compute_fold(labels, cYt, Yt_truth)

            cstats.compute_fold(cclabels, ccYt, ccYt_truth,
                                truth_count=truth_count,
                                total_count=total_count)

        for label, y_truth, y, cy in zip(labels, Yt_truth, Yt, cYt):
            probs[label] = (y_truth, y)
            classes[label] = (y_truth, cy)

    if check_accuracy:
        print('== done ==')
        print('\n== classification test on connective ==')
        stats.print_total(truth_count=linkage_counts)
        print('\n== classification test on component ==')
        cstats.print_total(truth_count=len(word_ambig))

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
    train_prob = args.output is not None
    train_classify = args.output_classify is not None
    assert(train_prob or train_classify)

    if args.classifier == 'SVM':
        global_classifier['key'] = SVC
    elif args.classifier == 'DT':
        global_classifier['key'] = DecisionTreeClassifier
    elif args.classifier == 'RF':
        global_classifier['key'] = RandomForestClassifier
    elif args.classifier == 'NB':
        global_classifier['key'] = GaussianNB
    elif args.classifier == 'LSVM':
        global_classifier['key'] = LinearSVC
    elif args.classifier == 'LR':
        global_classifier['key'] = LogisticRegressor
    else:
        assert(False)

    fhelper = corpus.FoldsHelper(args.folds)
    feature_tbl = features.load_features_table(
        args.linkage_features, lambda x: tuple(x.split('-')))
    linkage_counts = count_linkage(args.linkage)

    probs, classes = train_linkage_probs(fhelper, feature_tbl, linkage_counts,
                                         ambig_path=args.word_ambig,
                                         count_path=args.word_count,
                                         check_accuracy=args.check_accuracy,
                                         train_prob=train_prob,
                                         train_classify=train_classify)

    if train_prob:
        output_file(args.output, probs)
    if train_classify:
        output_file(args.output_classify, classes)

if __name__ == '__main__':
    main()
