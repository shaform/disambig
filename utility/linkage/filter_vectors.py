import argparse

from collections import defaultdict


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--corpus_pos', required=True)
    parser.add_argument('--output', required=True,
                        help='output folds file')

    return parser.parse_args()


def main():
    args = process_commands()

    tokens = set()
    with open(args.corpus_pos, 'r') as f:
        for l in f:
            for t in l.rstrip('\n').split('\t', 1)[1].split(' '):
                tokens.add(t)

    remain_count = len(tokens)
    with open(args.vectors, 'r') as vf, open(args.output, 'w') as of:
        for l in vf:
            l = l.rstrip()
            if l.split(' ', 1)[0] in tokens:
                remain_count -= 1
                of.write(l + '\n')
                if remain_count == 0:
                    break
    print('{} tokens not found'.format(remain_count))

if __name__ == '__main__':
    main()
