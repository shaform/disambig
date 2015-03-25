"""Select the connectives to use by corpus"""
import argparse

from collections import defaultdict


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='unique clueweb corpus file')
    parser.add_argument('--pairs', required=True,
                        help='raw pair connectives sql dump file')
    parser.add_argument('--output', required=True,
                        help='output connectives files')
    parser.add_argument('--threshold', default=500,
                        help='minimum sentences found to be kept')

    return parser.parse_args()


def load_words(path):
    pairs = []
    with open(path, 'r') as f:
        for l in f:
            tokens = l.strip().split()
            pairs.append((tokens[2], tokens[3], int(tokens[-1])))
    print('{} pairs'.format(len(pairs)))
    pairs = {(a, b, c) for a, b, c in pairs if a != b}
    print('{} unique pairs without POS'.format(len(pairs)))

    d = defaultdict(int)
    for a, b, _ in pairs:
        d[a] += 1
        d[b] += 1
    pairs = {(a, b, c) for a, b, c in pairs if d[a] > 1 or d[b] > 1}
    print('{} overlapped pairs'.format(len(pairs)))

    words = defaultdict(lambda: defaultdict(set))
    for a, b, c in pairs:
        words[a][c].add((a, b))
        words[b][c].add((a, b))
    print('{} words'.format(len(words)))

    for k in list(words):
        if len(words[k]) <= 1:
            del words[k]
    print('{} ambiguous words'.format(len(words)))

    pairs = {(a, b, c) for a, b, c in pairs if a in words or b in words}
    print('{} ambiguous pairs'.format(len(pairs)))

    return words


def filter_words(path, words, threshold):
    counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for w, tag_dict in words.items():
        for tag, pairs in tag_dict.items():
            for a, b in pairs:
                pair_to_words[a + ' ' + b].add((w, tag))

    with open(path, 'r') as f:
        for i, l in enumerate(f):
            pair = ' '.join(l.split(' ', 2)[:2])
            if pair in pair_to_words:
                for wt in pair_to_words[pair]:
                    counts[wt] += 1
            if i % 1000000 == 0:
                print('{} processed'.format(i))

    print('{} meanings'.format(len(counts)))
    for wt in list(counts):
        if counts[wt] < threshold:
            del counts[wt]
    print('{} meanings > {} sentences'.format(len(counts), threshold))

    for w, tags in words.items():
        for tag in list(tags):
            if (w, tag) not in counts:
                del tags[tag]
    for w in list(words):
        if len(words[w]) <= 1:
            del words[w]

    print('after filtering')
    print('{} ambiguous words'.format(len(words)))
    print('{} ambiguous meanings'.format(sum(len(x) for x in words.values())))


def output(path, words):
    with open(path, 'w') as f:
        for w, tag_dict in words.items():
            for tag, pairs in tag_dict.items():
                for a, b in pairs:
                    f.write('{}\t{}\t{}\n'.format(w, tag, a + ' ' + b))


def main():
    args = process_commands()
    words = load_words(args.pairs)
    filter_words(args.corpus, words, args.threshold)
    output(args.output, words)

if __name__ == '__main__':
    main()
