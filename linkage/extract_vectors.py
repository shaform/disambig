"""Extract word2vec vectors with tokens"""
import argparse

from collections import defaultdict


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', required=True,
                        help='POS-tagged raw corpus file')
    parser.add_argument('--vectors', required=True,
                        help='vectors file')
    parser.add_argument('--output', required=True,
                        help='output folds file')

    return parser.parse_args()


if __name__ == '__main__':
    args = process_commands()

    tokens = set()
    # load file
    with open(args.pos, 'r') as f:
        for l in f:
            for token in l.split('\t')[1].rstrip('\n').split():
                tokens.add(token)

    vectors = {}
    with open(args.vectors, 'r') as f, open(args.output, 'w') as of:
        for l in f:
            label, line = l.split(' ', 1)
            if label in tokens:
                of.write(l)
                tokens.remove(label)

    print('not found', tokens)
