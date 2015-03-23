"""Randomly split corpus into folds"""
import argparse

from collections import defaultdict

from sklearn import cross_validation


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5,
                        help='how many folds to use')
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--output', required=True,
                        help='output folds file')

    return parser.parse_args()


if __name__ == '__main__':
    args = process_commands()

    # load file
    with open(args.corpus, 'r') as f:
        counts = {l.split('\t')[0]: 0 for l in f}
    with open(args.linkage, 'r') as f:
        for l in f:
            counts[l.split('\t')[0]] += 1

    # random split
    labels, y = zip(*list(sorted(counts.items())))
    maxi = 1
    errors = 0
    while True:
        skf = cross_validation.StratifiedKFold(
            y, n_folds=args.folds, shuffle=True)
        folds = list(skf)

        indices = [f[1] for f in folds]

        try:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    d = abs(len(indices[i]) - len(indices[j]))
                    if d > maxi:
                        raise ValueError
            else:
                break
        except ValueError:
            errors += 1

            if errors % 1000 == 0:
                maxi += 1
                print('maxi:', maxi)
            continue

    # process labels
    folds = {}
    for i, idx in enumerate(indices):
        for j in idx:
            folds[labels[j]] = i

    # write file
    with open(args.output, 'w') as f:
        for label in labels:
            f.write('{}\t{}\n'.format(label, folds[label]))
