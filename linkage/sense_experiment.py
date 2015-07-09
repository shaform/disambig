"""linkage sense experiments"""
import argparse

import numpy as np

import evaluate
import features
import linkage

from sklearn import cross_validation

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

NOT_COUNTED = (1, 5, 16)

TRANS = {}

__count = 0
for i in range(17):
    if i in NOT_COUNTED:
        continue
    else:
        TRANS[i] = __count
        __count += 1


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--linkage_features', required=True,
                        help='linkage features file')
    parser.add_argument('--word_features', required=True,
                        help='word features file')

    return parser.parse_args()


def extract_indices(lst, indices):
    extracted = []
    for i in indices:
        extracted.append(lst[i])
    return extracted


def main():
    args = process_commands()

    truth = linkage.LinkageFile(args.linkage)

    truth.print_type_stats()

    feature_tbl = features.load_features_table(
        args.linkage_features, lambda x: tuple(x.split('-')))

    #word_feature_tbl = features.load_features_table(args.word_features)

    X = []
    Y = []
    labels = []
    X2 = []
    Y2 = []
    labels2 = []

    for label, pset in sorted(truth.linkage.items()):
        feature_set = feature_tbl[label]
        x_set = {key: tbl for key, _, tbl in feature_set}

        for indices in sorted(pset):
            X.append(x_set[indices])
            Y.append(truth.linkage_type[label][indices])
            labels.append((label, indices))

            ctype2 = truth.linkage_type2[label][indices]
            if ctype2 not in NOT_COUNTED:
                X2.append(x_set[indices])
                Y2.append(TRANS[ctype2])
                labels2.append((label, indices))

    X = np.array(X)
    X2 = np.array(X2)
    Y = np.array(Y)
    Y2 = np.array(Y2)

    lr = SVC()
    lr = LogisticRegression()
    lr = GaussianNB()
    lr = LogisticRegressionCV()

    print('predict 1-level...')
    folds = cross_validation.StratifiedKFold(
        Y, 10, shuffle=True, random_state=np.random.RandomState(1))
    Yp = cross_validation.cross_val_predict(
        lr, X, Y, cv=folds, n_jobs=10)

    print('predict 2-level...')
    folds2 = cross_validation.StratifiedKFold(
        Y2, 10, shuffle=True, random_state=np.random.RandomState(1))
    Y2p = cross_validation.cross_val_predict(
        lr, X2, Y2, cv=folds2, n_jobs=10)

    print('collect type predictions...')
    Ys, Yps = [], []
    wYs, wYps = [], []
    for _, test_idx in folds:
        ys = list(Y[test_idx])
        yps = list(Yp[test_idx])
        ls = extract_indices(labels, test_idx)

        Ys.append(ys)
        Yps.append(yps)

        wys, wyps = [], []

        for y, yp, l in zip(ys, yps, ls):
            length = len(l[1])
            wys.extend([y] * length)
            wyps.extend([yp] * length)

        wYs.append(wys)
        wYps.append(wyps)

    print('collect 2-level type predictions...')
    Y2s, Y2ps = [], []
    wY2s, wY2ps = [], []
    for _, test_idx in folds2:
        ys = list(Y2[test_idx])
        yps = list(Y2p[test_idx])
        ls = extract_indices(labels2, test_idx)

        Y2s.append(ys)
        Y2ps.append(yps)

        wys, wyps = [], []

        for y, yp, l in zip(ys, yps, ls):
            length = len(l[1])
            wys.extend([y] * length)
            wyps.extend([yp] * length)

        wY2s.append(wys)
        wY2ps.append(wyps)

    evaluate.print_sense_scores(Ys, Yps, 'Overall', print_accuracy=True)
    evaluate.print_sense_scores(
        Y2s, Y2ps, 'Overall for 2nd-level', print_accuracy=True)

    print('\n== word stats ==')

    evaluate.print_sense_scores(wYs, wYps, 'Overall', print_accuracy=True)
    evaluate.print_sense_scores2(wY2s, wY2ps, 'Overall for 2nd-level')

if __name__ == '__main__':
    main()
