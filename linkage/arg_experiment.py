"""linkage argument experiments"""
import argparse

import numpy as np

import argument
import corpus
import evaluate
import features
import linkage


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
    parser.add_argument('--vector', required=True,
                        help='vector file')
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--crfsuite', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)

    return parser.parse_args()


def output_crf(fout, data_set, arguments, corpus_file, vectors):
    for l in data_set:
        tokens = corpus_file.corpus[l]
        pos_tokens = corpus_file.pos_corpus[l]

        for arg in arguments.arguments(l):
            tlabels, tfeatures = argument.extract_features(
                tokens,
                pos_tokens,
                vectors,
                arg)

            # output
            for idx in range(len(tokens)):
                fout.write('{}\t{}\n'.format(
                    tlabels[idx],
                    '\t'.join(tfeatures[idx])
                ))
            fout.write('\n')


def test(fhelper, arguments, corpus_file, vectors, train_file, test_file):

    for i in fhelper.folds():
        output_crf(
            train_file,
            fhelper.train_set(i),
            arguments,
            corpus_file,
            vectors
        )
        output_crf(
            test_file,
            fhelper.test_set(i),
            arguments,
            corpus_file,
            vectors
        )
        break


def main():
    args = process_commands()
    arguments = argument.ArgumentFile(args.argument)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    vectors = corpus.VectorFile(args.vector)
    fhelper = corpus.FoldsHelper(args.folds)

    with open(args.train, 'w') as train_file, open(args.test, 'w') as test_file:
        test(fhelper, arguments, corpus_file, vectors, train_file, test_file)


if __name__ == '__main__':
    main()
