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
    parser.add_argument('--keep_boundary', action='store_true')

    return parser.parse_args()


def extract_features(corpus_file, arguments, test_arguments):
    data_set = defaultdict(list)
    test_set = defaultdict(list)
    counter = evaluate.ProgressCounter()

    for l, tokens in corpus_file.corpus.items():
        counter.step()

        pos_tokens = corpus_file.pos_corpus[l]
        parsed = corpus_file.parse_corpus[l]
        EDUs = corpus_file.edu_corpus[l]

        for arg in test_arguments.arguments(l):
            a_indices = arguments.get_a_indices(l, arg)
            arg = list(arg)
            arg[-1] = a_indices
            c_indices, tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, arg)
            test_set[l].append((c_indices, tlabels, tfeatures))

        for arg in arguments.arguments(l):
            c_indices, tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, arg)
            data_set[l].append((c_indices, tlabels, tfeatures))

    return data_set, test_set


def extract_crf_data(labels, data_set):
    crf_data = []
    for l in labels:
        for c_indices, tlabels, tfeatures in data_set[l]:
            crf_data.append((l, c_indices, tlabels, tfeatures))
    return crf_data


def output_crf(fout, crf_data):
    for _, _, tlabels, tfeatures in crf_data:
        for i, label in enumerate(tlabels):
            # if not argument.is_argument_label(label):
            #     continue
            fout.write('{}\t{}\n'.format(
                label,
                '\t'.join(tfeatures[i])
            ))
        fout.write('\n')


def load_predict(fin):
    arg_spans = []
    labels = []
    for l in fin:
        l = l.strip()
        if l.startswith('@'):
            prob = float(l.split()[1])

            last = argument._BEFORE
            idx = 0
        elif l:
            label = int(l.split(':')[0])

            labels.append(label)

            last = label
            idx += 1
        else:
            argument.correct_labels(labels)
            argument.check_continuity(labels)
            arg_spans.append(argument.labels_to_offsets(labels))
            labels = []
    else:
        assert(len(labels) == 0)
    return arg_spans


def test(fhelper, args, test_args, corpus_file,
         train_path, test_path, model_path, crf,
         keep_boundary):

    args.init_truth(corpus_file)
    data_set, test_set = extract_features(corpus_file, args, test_args)

    processes = []
    test_crf_data = []
    for i in fhelper.folds():
        path = '{}.{}'.format(train_path, i)
        mpath = '{}.{}'.format(model_path, i)
        crf_data = extract_crf_data(fhelper.train_set(i), data_set)
        test_crf_data.append(extract_crf_data(fhelper.test_set(i), test_set))
        with open(path, 'w') as train_file:
            output_crf(
                train_file,
                crf_data
            )
        processes.append(subprocess.Popen([crf, 'learn', '-m', mpath, path]))
    for p in processes:
        p.wait()

    print('start testing')
    processes = []
    for i in fhelper.folds():
        mpath = '{}.{}'.format(model_path, i)
        path = '{}.{}'.format(test_path, i)
        with open(path, 'w') as test_file:
            output_crf(
                test_file,
                test_crf_data[i]
            )

        processes.append(
            subprocess.Popen([crf, 'tag', '-pi', '-m', mpath, path],
                             stdout=subprocess.PIPE, universal_newlines=True))
    cv_stats = defaultdict(list)
    for crf_data, p in zip(test_crf_data, processes):
        p.wait()
        arg_spans = load_predict(p.stdout)
        assert(len(arg_spans) == len(crf_data))

        correct = 0
        for arg_span, item in zip(arg_spans, crf_data):
            s = args.edu_truth[item[0]][item[1]]
            correct += (s == arg_span)
        cv_stats['Accuracy'].append(correct / len(arg_spans))
    print('Accuracy:', np.mean(cv_stats['Accuracy']))


def main():
    args = process_commands()
    arguments = argument.ArgumentFile(args.argument)
    test_arguments = argument.ArgumentFile(args.argument_test)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    fhelper = corpus.FoldsHelper(args.folds)

    test(fhelper, arguments, test_arguments,
         corpus_file,
         args.train, args.test, args.model, args.crfsuite,
         args.keep_boundary)


if __name__ == '__main__':
    main()
