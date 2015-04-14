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


HEADERS = [
    'causality',
    'coordination',
    'transition',
    'explanation',
    'micro-AVG',
    'macro-AVG',
]


def print_scores(Y, Yp, label):
    print()
    print(label, ':')

    print('Relation\tPrecision\tRecall\tF1\tcases')

    scores = list(metrics.precision_recall_fscore_support(Y, Yp)[:3])
    scores.append([0, 0, 0, 0])
    scores = [list(ss) for ss in scores]
    scores = list(list(ss) for ss in zip(*scores))

    for y in Y:
        scores[y][-1] += 1

    scores.extend([
        [
        metrics.precision_score(Y, Yp, average='micro'),
        metrics.recall_score(Y, Yp, average='micro'),
        metrics.f1_score(Y, Yp, average='micro'),
        ],
        [
            metrics.precision_score(Y, Yp, average='macro'),
            metrics.recall_score(Y, Yp, average='macro'),
            metrics.f1_score(Y, Yp, average='macro'),
        ]
    ])

    for header, score in zip(HEADERS, scores):
        score_line = '\t'.join('{:.02}'.format(s) for s in score[:3])
        if len(score) > 3:
            score_line += '\t{}'.format(score[3])
        print('{}\t{}'.format(header, score_line))


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

    print_scores(Y, Yp, 'Overall')

    Y = [y for i, y in enumerate(Y) if labels[i] in truth.linkage_with_types]
    Yp = [y for i, y in enumerate(Yp) if labels[i] in truth.linkage_with_types]
    print_scores(Y, Yp, 'Only ambiguous')
    print('Total cases: {}'.format(len(Y)))

if __name__ == '__main__':
    main()
