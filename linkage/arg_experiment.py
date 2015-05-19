"""linkage argument experiments"""
import argparse

import numpy as np

import argument
import corpus
import evaluate
import features
import linkage

from collections import defaultdict


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--argument', required=True,
                        help='argument ground truth file')
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')
    parser.add_argument('--corpus_pos', required=True,
                        help='pos-tagged raw corpus file')
    parser.add_argument('--corpus_parse', required=True,
                        help='syntax-parsed raw corpus file')
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--crfsuite', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)

    return parser.parse_args()


def extract_features(corpus_file, arguments):
    data_set = defaultdict(list)
    counter = evaluate.ProgressCounter()

    for l, tokens in corpus_file.corpus.items():
        counter.step()

        pos_tokens = corpus_file.pos_corpus[l]
        parsed = corpus_file.parse_corpus[l]
        EDUs = argument.get_EDU_offsets(tokens)

        for arg in arguments.arguments(l):
            tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, arg)
            data_set[l].append((tlabels, tfeatures))
    return data_set


def output_crf(fout, labels, data_set):
    for l in labels:
        for tlabels, tfeatures in data_set[l]:
            for i, label in enumerate(tlabels):
                # if not argument.is_argument_label(label):
                #     continue
                fout.write('{}\t{}\n'.format(
                    label,
                    '\t'.join(tfeatures[i])
                ))
            fout.write('\n')


def test(fhelper, arguments, corpus_file, train_path, test_path):

    data_set = extract_features(corpus_file, arguments)
    for i in fhelper.folds():
        with open('{}.{}'.format(train_path, i), 'w') as train_file:
            output_crf(
                train_file,
                fhelper.train_set(i),
                data_set
            )
        with open('{}.{}'.format(test_path, i), 'w') as test_file:
            output_crf(
                test_file,
                fhelper.test_set(i),
                data_set
            )


def main():
    args = process_commands()
    arguments = argument.ArgumentFile(args.argument)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    fhelper = corpus.FoldsHelper(args.folds)

    test(fhelper, arguments, corpus_file, args.train, args.test)


if __name__ == '__main__':
    main()
