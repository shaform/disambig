"""Align CDTB connective positions with segmented text CDTB."""
import argparse

from collections import defaultdict

from corpus import load_corpus
from statistics import print_distribution

_TP = {
    # causality
    '因果关系': 0,
    '推断关系': 0,
    '假设关系': 0,
    '目的关系': 0,
    '条件关系': 0,
    '背景关系': 0,

    # coordination
    '并列关系': 1,
    '顺承关系': 1,
    '递进关系': 1,
    '选择关系': 1,
    '对比关系': 1,

    # transition
    '转折关系': 2,
    '让步关系': 2,

    # explanation
    '解说关系': 3,
    '总分关系': 3,
    '例证关系': 3,
    '评价关系': 3,
}


def extract_span(text, index, tokens, stats=None):
    start = 0
    text_span = ''

    if stats is None:
        stats = {}

    extracted_tokens = []
    for i, x in enumerate(tokens):
        end = start + len(x)
        if start >= index[0] and start < index[1] or end > index[0] and end <= index[1]:
            text_span += x
            extracted_tokens.append((i, len(x)))
        start = end

    if text in text_span:
        if text != text_span:
            stats['not fit boundary'] += 1
        else:
            stats['{:03d}'.format(len(extracted_tokens))] += 1

        offset = text_span.find(text)
        offset_end = offset + len(text)

        offsets = []
        for i, length in extracted_tokens:
            if offset < length:
                offsets.append('{}[{}:{}]'.format(i,
                                                  max(0, offset),
                                                  min(length,
                                                      offset_end)
                                                  ))
                if offset_end <= length:
                    break
            offset -= length
            offset_end -= length

        return offsets, None
    else:
        return None, text_span


def extract_indices(indices, sep='-'):
    indices = [x.split(':') for x in indices.split(sep)]
    indices = [(int(x), int(y)) for x, y in indices]
    return indices


def align_connectives(corpus, cnnct_path, linkage_output):
    stats = defaultdict(int)

    with open(cnnct_path, 'r') as f, open(linkage_output, 'w') as of:
        for l in f:
            label, conncts, indices, tp = l.rstrip('\n').split('\t')
            conncts = conncts.split('-')
            indices = extract_indices(indices)

            # extract a linkage
            detected_indices = []
            for connct, index in zip(conncts, indices):
                detected_index, text_span = extract_span(
                    connct,
                    index,
                    corpus[label],
                    stats)
                if detected_index is None:
                    print('wrong')
                    print(corpus[label])
                    print(label, text_span, connct)
                    print(index)
                    break
                else:
                    detected_indices.append(','.join(detected_index))
            else:
                # successful
                of.write('{}\t{}\t{}\t{}\n'.format(label, '-'.join(conncts),
                                                   '-'.join(detected_indices),
                                                   _TP[tp]))

    print('connective components:')
    print_distribution(stats)


def align_arguments(corpus, arg_path, argument_output):
    stats = defaultdict(int)

    with open(arg_path, 'r') as f, open(argument_output, 'w') as of:
        for l in f:
            label, sents, indices = l.rstrip('\n').split('\t')
            sents = sents.split('|')
            indices = extract_indices(indices, sep='|')

            # extract arguments
            detected_indices = []
            for sent, index in zip(sents, indices):
                detected_index, text_span = extract_span(
                    sent,
                    index,
                    corpus[label],
                    stats)
                if detected_index is None:
                    print('wrong')
                    print(corpus[label])
                    print(label, text_span, sent)
                    print(index)
                    break
                else:
                    detected_indices.append(','.join(detected_index))
            else:
                # successful
                of.write('{}\t{}\n'.format(
                    label,
                    '-'.join(detected_indices)
                ))

    print('arguments:')
    print_distribution(stats)


def main():
    args = process_commands()
    print('\n== start align ==')

    corpus = load_corpus(args.corpus)

    align_connectives(corpus, args.connective, args.linkage_output)
    align_arguments(corpus, args.arg, args.argument_output)


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True,
                        help='segmented text')
    parser.add_argument('--connective', required=True,
                        help='linkage text')
    parser.add_argument('--linkage_output', required=True)
    parser.add_argument('--arg', required=True,
                        help='argument text')
    parser.add_argument('--argument_output', required=True)

    return parser.parse_args()


if __name__ == '__main__':
    main()
