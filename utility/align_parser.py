"""Align raw Stanford parser outputs with origin labels"""
import argparse


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')
    parser.add_argument('--parsed', required=True,
                        help='raw parsed file')
    parser.add_argument('--output', required=True,
                        help='output file')

    return parser.parse_args()

def read_parsed(path):
    with open(path, 'r') as f:
        lines = ''
        for l in f:
            l = l.strip()
            if l == '':
                yield lines.strip()
                lines = ''
            else:
                lines += ' ' + l

def read_labels(path):
    with open(path, 'r') as f:
        yield from (l.split('\t')[0] for l in f)

def main():
    args = process_commands()
    with open(args.output, 'w') as f:
        for label, parsed in zip(read_labels(args.corpus),
                                 read_parsed(args.parsed)):
            f.write('{}\t{}\n'.format(label, parsed))

if __name__ == '__main__':
    main()
