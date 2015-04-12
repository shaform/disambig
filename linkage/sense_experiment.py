"""linkage sense experiments"""
import argparse

import numpy as np

import features
import linkage


from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics


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
    labels = []

    for label, pset in sorted(truth.linkage.items()):
        for indices in sorted(pset):
            feature_set = feature_tbl[label]
            x_set = {key: tbl for key, _, tbl in feature_set}
            X.append(x_set[indices])
            Y.append(truth.linkage_type[label][indices])
            labels.append((label, indices))

    X = np.array(X)
    Y = np.array(Y)

    lr = LogisticRegression()

    folds = cross_validation.StratifiedKFold(
        Y, 10, shuffle=True, random_state=np.random.RandomState(1))
    Yp = cross_validation.cross_val_predict(
        lr, X, Y, cv=folds, n_jobs=10)

    print('Overall')
    scores = metrics.precision_recall_fscore_support(Y, Yp)
    print(scores)
    scores = metrics.f1_score(Y, Yp, average='micro')
    print(scores)

    print('Only ambiguous')
    Y = [y for i, y in enumerate(Y) if labels[i] in truth.linkage_with_types]
    Yp = [y for i, y in enumerate(Yp) if labels[i] in truth.linkage_with_types]
    print('Total cases: {}'.format(len(Y)))
    scores = metrics.precision_recall_fscore_support(Y, Yp)
    print(scores)
    scores = metrics.f1_score(Y, Yp, average='micro')
    print(scores)

if __name__ == '__main__':
    main()
