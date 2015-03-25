"""Filter sentences for use"""
import argparse

from collections import defaultdict


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='unique clueweb corpus file')
    parser.add_argument('--cnnct', required=True,
                        help='connectives file generated by select.py')
    parser.add_argument('--output', required=True,
                        help='output connectives files')

    return parser.parse_args()


def load_connectives(path):
    connectives = defaultdict(set)
    with open(path, 'r') as f:
        for l in f:
            word, sense, surface = l.strip().split('\t')
            connectives[surface].add((word, sense))
    return connectives


def filter_corpus(input_path, output_path, connectives):
    print('filtering corpus...')
    filtered_dict = defaultdict(list)
    with open(input_path, 'r') as f, open(output_path, 'w') as out:
        num = 1
        for l in f:
            header, tail = l.strip().split(';', 1)
            pair = ' '.join(header.split(' ', 2)[:2])
            if pair in connectives:
                x, y = pair.split(' ')
                tokens = tail.split(' ')
                for key in connectives[pair]:
                    if x == key[0]:
                        eliminated = y + '/'
                    elif y == key[0]:
                        eliminated = x + '/'
                    else:
                        assert(False)
                    xtokens = list(
                        filter(lambda x: not x.startswith(eliminated), tokens))
                    if len(xtokens) + 1 == len(tokens):
                        filtered_dict[key].append(' '.join(xtokens) + '\n')
            else:
                out.write('@@SENT-{} '.format(num))
                num += 1
                out.write(tail + '\n')

    return filtered_dict


def append_filtered(path, filtered_dict):
    with open(path, 'a') as f:
        for (w, s), tails in sorted(filtered_dict.items()):
            print('process {}-{} with {} sentences'.format(w, s, len(tails)))
            for i, tail in enumerate(tails):
                f.write('@@SSENT-{}-{}-{} '.format(w, s, i + 1))
                f.write(tail)


def main():
    args = process_commands()
    connectives = load_connectives(args.cnnct)
    filtered_dict = filter_corpus(args.corpus, args.output, connectives)
    append_filtered(args.output, filtered_dict)

if __name__ == '__main__':
    main()