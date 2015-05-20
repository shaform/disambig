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
            c_indices, cEDU, tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, arg)
            test_set[l].append((c_indices, cEDU, tlabels, tfeatures))

        for arg in arguments.arguments(l):
            c_indices, cEDU, tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, arg)
            data_set[l].append((c_indices, cEDU, tlabels, tfeatures))

    return data_set, test_set


def extract_crf_data(labels, data_set):
    crf_data = []
    for l in labels:
        for c_indices, cEDU, tlabels, tfeatures in data_set[l]:
            crf_data.append((l, c_indices, cEDU, tlabels, tfeatures))
    return crf_data


def get_ranges(crf_data):
    ranges = []
    for _, _, _, labels, _ in crf_data:
        start = labels.index(argument._BEGIN)
        try:
            end = labels.index(argument._AFTER)
        except:
            end = len(labels)
        ranges.append([(start, end)])
    return ranges


def get_hierarchy_ranges(crf_data, edu_spans):
    ranges = []
    for l, c_indices, _, _, _ in crf_data:
        ranges.append(edu_spans[l][c_indices])
    return ranges


def output_crf(fout, crf_data, ranges=None):
    for i, (_, _, _, tlabels, tfeatures) in enumerate(crf_data):
        if ranges is not None:
            rs = ranges[i]
        else:
            rs = [(0, len(tlabels))]
        for start, end in rs:
            for j, label in enumerate(tlabels):
                if j < start:
                    continue
                elif j >= end:
                    break
                fout.write('{}\t{}\n'.format(
                    label,
                    '\t'.join(tfeatures[j])
                ))
            fout.write('\n')


def load_predict(fin, ranges):
    arg_spans = []
    probs = []
    labels = []
    for l in fin:
        l = l.strip()
        if l.startswith('@'):
            prob = float(l.split()[1])
            probs.append(prob)

            last = argument._BEFORE
        elif l:
            label = int(l.split(':')[0])

            labels.append(label)

            last = label
        else:
            argument.correct_labels(labels)
            argument.check_continuity(labels)
            idx = len(arg_spans)
            start = ranges[idx][0][0] if ranges is not None else 0
            arg_spans.append(argument.labels_to_offsets(
                labels,
                start=start))
            labels = []
    else:
        assert(len(labels) == 0)
        if ranges is not None:
            assert(len(arg_spans) == len(ranges))
    return arg_spans, probs


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
        if keep_boundary:
            crf_ranges = get_ranges(crf_data)
        else:
            crf_ranges = get_hierarchy_ranges(crf_data, args.edu_spans)
        with open(path, 'w') as train_file:
            output_crf(
                train_file,
                crf_data,
                crf_ranges
            )
        test_crf_data.append(extract_crf_data(fhelper.test_set(i), test_set))

        processes.append(subprocess.Popen([crf, 'learn', '-m', mpath, path]))
    for p in processes:
        p.wait()

    print('start testing')
    processes = []
    test_crf_ranges = [None] * len(test_crf_data)
    if keep_boundary:
        test_crf_ranges = [get_ranges(crf_data) for crf_data in test_crf_data]
    for i in fhelper.folds():
        mpath = '{}.{}'.format(model_path, i)
        path = '{}.{}'.format(test_path, i)
        with open(path, 'w') as test_file:
            output_crf(
                test_file,
                test_crf_data[i],
                test_crf_ranges[i]
            )

        processes.append(
            subprocess.Popen([crf, 'tag', '-pi', '-m', mpath, path],
                             stdout=subprocess.PIPE, universal_newlines=True))

    preds = []
    pred_probs = []
    for crf_data, ranges, p in zip(test_crf_data, test_crf_ranges, processes):
        p.wait()
        arg_spans, probs = load_predict(p.stdout, ranges)
        assert(len(arg_spans) == len(crf_data))
        preds.append(arg_spans)
        pred_probs.append(probs)

    if not keep_boundary:
        for crf_data, pds, probs in zip(test_crf_data, preds, pred_probs):
            hierarchy_adjust(crf_data, pds, probs)

    # evaluation
    cv_stats = defaultdict(list)
    sum_of_total = 0
    for i, crf_data, arg_spans in zip(fhelper.folds(), test_crf_data, preds):
        correct = 0
        tp = fp = total = 0
        for arg_span, item in zip(arg_spans, crf_data):
            s = args.edu_truth[item[0]][item[1]]
            tp += len(s & arg_span)
            fp += len(arg_span - s)
            correct += (s == arg_span)

        for l in fhelper.test_set(i):
            d = args.edu_truth[l]
            for s in d.values():
                total += len(s)
        sum_of_total += total
        print('Totally {} arguments'.format(total))

        accuracy = correct / len(arg_spans) if len(arg_spans) > 0 else 1
        recall = tp / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1
        f1 = evaluate.f1(recall, prec)
        cv_stats['Accuracy'].append(accuracy)
        cv_stats['Recall'].append(recall)
        cv_stats['Prec'].append(prec)
        cv_stats['F1'].append(f1)

    print('Accuracy:', np.mean(cv_stats['Accuracy']))
    print('Recall:', np.mean(cv_stats['Recall']))
    print('Prec:', np.mean(cv_stats['Prec']))
    print('F1:', np.mean(cv_stats['F1']))
    print('Totally {} arguments for all'.format(sum_of_total))


def pop_max(items, probs):
    max_idx = items[0]
    maxi = 0
    maxp = probs[max_idx]
    for i, idx in enumerate(items):
        if probs[idx] > maxp:
            maxp = probs[idx]
            max_idx = idx
            maxi = i
    items.pop(maxi)
    return max_idx


def get_items(items, lst):
    elems = []
    for i in items:
        elems.append(lst[i])
    return elems


# cd = (l, c_indices, cEDUs, tlabels, tfeatures)
def reduce_inner(item, preds, cds):
    for pd, cd in zip(preds, cds):
        # if connectives are insie of item
        # if span is less than item
        # then use rule to restrict and then CRF predict again
        pass


def hierarchy_adjust(crf_data, preds, probs):
    instances = defaultdict(list)
    for i, item in enumerate(crf_data):
        instances[item[0]].append(i)

    for items in instances.values():
        while len(items) > 0:
            idx = pop_max(items, probs)
            pds = get_items(items, preds)
            cds = get_items(items, crf_data)
            reduce_inner(preds[idx], pds, cds)


def main():
    args = process_commands()
    arguments = argument.ArgumentFile(args.argument)
    test_arguments = argument.ArgumentFile(args.argument_test)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    fhelper = corpus.FoldsHelper(args.folds)

    keep_boundary = args.keep_boundary and args.argument == args.argument_test

    test(fhelper, arguments, test_arguments,
         corpus_file,
         args.train, args.test, args.model, args.crfsuite,
         keep_boundary)


if __name__ == '__main__':
    main()
