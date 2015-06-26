"""linkage sense experiments"""
import argparse

import numpy as np

import evaluate
import features
import linkage


from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--linkage_features', required=True,
                        help='linkage features file')

    return parser.parse_args()


def main():
    args = process_commands()

    truth = linkage.LinkageFile(args.linkage)

    truth.print_type_stats()

    feature_tbl = features.load_features_table(
        args.linkage_features, lambda x: tuple(x.split('-')))

    X = []
    Y = []
    Y2 = []
    labels = []

    for label, pset in sorted(truth.linkage.items()):
        for indices in sorted(pset):
            feature_set = feature_tbl[label]
            x_set = {key: tbl for key, _, tbl in feature_set}
            X.append(x_set[indices])
            Y.append(truth.linkage_type[label][indices])
            Y2.append(truth.linkage_type2[label][indices])
            labels.append((label, indices))

    X = np.array(X)
    Y = np.array(Y)
    Y2 = np.array(Y2)

    lr = LogisticRegression()

    print('predict 1-level...')
    folds = cross_validation.StratifiedKFold(
        Y, 10, shuffle=True, random_state=np.random.RandomState(1))
    Yp = cross_validation.cross_val_predict(
        lr, X, Y, cv=folds, n_jobs=10)

    print('predict 2-level...')
    folds2 = cross_validation.StratifiedKFold(
        Y2, 10, shuffle=True, random_state=np.random.RandomState(1))
    Y2p = cross_validation.cross_val_predict(
        lr, X, Y2, cv=folds2, n_jobs=10)

    Ys, Yps = [], []
    for _, test_idx in folds:
        Ys.append(list(Y[test_idx]))
        Yps.append(list(Yp[test_idx]))

    Y2s, Y2ps = [], []
    for _, test_idx in folds2:
        Y2s.append(list(Y2[test_idx]))
        Y2ps.append(list(Y2p[test_idx]))

    evaluate.print_sense_scores(Ys, Yps, 'Overall', print_accuracy=True)
    evaluate.print_sense_scores2(Y2s, Y2ps, 'Overall for 2nd-level')

    # Y = [y for i, y in enumerate(Y) if labels[i] in truth.linkage_with_types]
    # Yp = [y for i, y in enumerate(Yp) if labels[i] in truth.linkage_with_types]
    # evaluate.print_sense_scores(Y, Yp, 'Only ambiguous')
    # print('Total cases: {}'.format(len(Y)))

if __name__ == '__main__':
    main()
