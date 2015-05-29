import argparse

import corpus

from collections import defaultdict


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')
    parser.add_argument('--input', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('action', choices=('preprocess', 'postprocess'))

    return parser.parse_args()


def preprocess(args):
    corpus_file = corpus.CorpusFile(args.corpus)

    labels = []
    with open(args.output, 'w') as f:
        for l, tokens in corpus_file.corpus.items():
            labels.append(l)
            for s, e in corpus_file.edu_corpus[l]:
                f.write('{}\n'.format(' '.join(tokens[s:e])))

    with open(args.label, 'w') as f:
        for l in labels:
            f.write(l + '\n')


def postprocess(args):
    corpus_file = corpus.CorpusFile(args.corpus)
    labels = []
    with open(args.label) as f:
        for l in f:
            labels.append(l.strip())
    with open(args.input) as f, open(args.output, 'w') as out:
        for lb in labels:
            lb = lb.strip()
            edus = []
            for _ in corpus_file.edu_corpus[lb]:
                items = []
                for l in f:
                    if l == '\n':
                        break
                    else:
                        items.append(l.strip())
                edus.append('@@@@'.join(items))
            out.write('{}\t{}\n'.format(lb, '\t'.join(edus)))


def main():
    args = process_commands()
    if args.action == 'preprocess':
        preprocess(args)
    else:
        postprocess(args)


if __name__ == '__main__':
    main()
