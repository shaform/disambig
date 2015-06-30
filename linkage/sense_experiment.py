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

    word_feature_tbl = features.load_features_table(args.word_features)

    X = []
    Y = []
    Y2 = []
    labels = []

    cX = []
    cY = []
    cY2 = []

    for label, pset in sorted(truth.linkage.items()):
        feature_set = feature_tbl[label]
        x_set = {key: tbl for key, _, tbl in feature_set}
        cfeature_set = word_feature_tbl[label]
        cx_set = {key: tbl for key, _, tbl in cfeature_set}
        for indices in sorted(pset):
            X.append(x_set[indices])
            Y.append(truth.linkage_type[label][indices])
            Y2.append(truth.linkage_type2[label][indices])
            labels.append((label, indices))

            for cindices in indices:
                cX.append(cx_set[cindices])
                cY.append(truth.linkage_type[label][indices])
                cY2.append(truth.linkage_type2[label][indices])

    X = np.array(X)
    Y = np.array(Y)
    Y2 = np.array(Y2)

    cX = np.array(cX)
    cY = np.array(cY)
    cY2 = np.array(cY2)

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

    print('predict word 1-level...')
    cfolds = cross_validation.StratifiedKFold(
        cY, 10, shuffle=True, random_state=np.random.RandomState(1))
    cYp = cross_validation.cross_val_predict(
        lr, cX, cY, cv=cfolds, n_jobs=10)

    print('predict word 2-level...')
    cfolds2 = cross_validation.StratifiedKFold(
        cY2, 10, shuffle=True, random_state=np.random.RandomState(1))
    cY2p = cross_validation.cross_val_predict(
        lr, cX, cY2, cv=cfolds2, n_jobs=10)

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
        ls = extract_indices(labels, test_idx)

        Y2s.append(ys)
        Y2ps.append(yps)

        wys, wyps = [], []

        for y, yp, l in zip(ys, yps, ls):
            length = len(l[1])
            wys.extend([y] * length)
            wyps.extend([yp] * length)

        wY2s.append(wys)
        wY2ps.append(wyps)

    print('collect type predictions...')
    cYs, cYps = [], []
    for _, test_idx in cfolds:
        ys = list(cY[test_idx])
        yps = list(cYp[test_idx])

        cYs.append(ys)
        cYps.append(yps)

    print('collect 2-level type predictions...')
    cY2s, cY2ps = [], []
    for _, test_idx in cfolds2:
        ys = list(cY2[test_idx])
        yps = list(cY2p[test_idx])

        cY2s.append(ys)
        cY2ps.append(yps)

    evaluate.print_sense_scores(Ys, Yps, 'Overall', print_accuracy=True)
    evaluate.print_sense_scores2(Y2s, Y2ps, 'Overall for 2nd-level')

    print('\n== word stats ==')

    evaluate.print_sense_scores(wYs, wYps, 'Overall', print_accuracy=True)
    evaluate.print_sense_scores2(wY2s, wY2ps, 'Overall for 2nd-level')

    print('\n== predict by word stats ==')
    evaluate.print_sense_scores(cYs, cYps, 'Overall', print_accuracy=True)
    evaluate.print_sense_scores2(cY2s, cY2ps, 'Overall for 2nd-level')

    # Y = [y for i, y in enumerate(Y) if labels[i] in truth.linkage_with_types]
    # Yp = [y for i, y in enumerate(Yp) if labels[i] in truth.linkage_with_types]
    # evaluate.print_sense_scores(Y, Yp, 'Only ambiguous')
    # print('Total cases: {}'.format(len(Y)))

if __name__ == '__main__':
    main()
