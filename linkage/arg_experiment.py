"""linkage argument experiments"""
import argparse
import subprocess

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
    parser.add_argument('--argument_test', required=True,
                        help='argument test file')
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
    parser.add_argument('--model', required=True)

    return parser.parse_args()


def extract_features(corpus_file, arguments, test_arguments):
    data_set = defaultdict(list)
    test_set = defaultdict(list)
    counter = evaluate.ProgressCounter()

    for l, tokens in corpus_file.corpus.items():
        counter.step()

        pos_tokens = corpus_file.pos_corpus[l]
        parsed = corpus_file.parse_corpus[l]
        EDUs = argument.get_EDU_offsets(tokens)

        for arg in test_arguments.arguments(l):
            a_indices = arguments.get_a_indices(l, arg)
            arg = list(arg)
            arg[-1] = a_indices
            tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, arg)
            test_set[l].append((tlabels, tfeatures))

        for arg in arguments.arguments(l):
            tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, arg)
            data_set[l].append((tlabels, tfeatures))

    return data_set, test_set


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


def test(fhelper, args, test_args, corpus_file,
         train_path, test_path, model_path, crf):

    data_set, test_set = extract_features(corpus_file, args, test_args)

    processes = []
    for i in fhelper.folds():
        path = '{}.{}'.format(train_path, i)
        mpath = '{}.{}'.format(model_path, i)
        with open(path, 'w') as train_file:
            output_crf(
                train_file,
                fhelper.train_set(i),
                data_set
            )
        processes.append(subprocess.Popen([crf, 'learn', '-m', mpath, path]))
        with open('{}.{}'.format(test_path, i), 'w') as test_file:
            output_crf(
                test_file,
                fhelper.test_set(i),
                test_set
            )
    for p in processes:
        p.wait()


def main():
    args = process_commands()
    arguments = argument.ArgumentFile(args.argument)
    test_arguments = argument.ArgumentFile(args.argument_test)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    fhelper = corpus.FoldsHelper(args.folds)

    test(fhelper, arguments, test_arguments,
         corpus_file,
         args.train, args.test, args.model, args.crfsuite)


if __name__ == '__main__':
    main()
