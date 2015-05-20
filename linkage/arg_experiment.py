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


class Predictor(object):

    def __init__(self, crf_path, model_path, train_path, test_path):
        self.crf = crf_path
        self.model_path = model_path
        self.train_path = train_path
        self.test_path = test_path

    def write_file(self, path, crf_data, crf_ranges):
        with open(path, 'w') as f:
            output_crf(
                f,
                crf_data,
                crf_ranges
            )

    def train(self, i, crf_data, crf_ranges):
        mpath = '{}.{}'.format(self.model_path, i)
        path = '{}.{}'.format(self.train_path, i)

        self.write_file(path, crf_data, crf_ranges)

        return subprocess.Popen([self.crf, 'learn', '-m', mpath, path])

    def test(self, i, crf_data, crf_ranges, wait=False):
        mpath = '{}.{}'.format(self.model_path, i)
        path = '{}.{}'.format(self.test_path, i)

        self.write_file(path, crf_data, crf_ranges)

        p = subprocess.Popen([self.crf, 'tag', '-pi', '-m', mpath, path],
                             stdout=subprocess.PIPE, universal_newlines=True)
        if wait:
            p.wait()

        return p


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
    parser.add_argument('--hierarchy_ranges', action='store_true')
    parser.add_argument('--hierarchy_adjust', action='store_true')

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


def get_hierarchy_ranges(crf_data, edu_spans, hierarchy_ranges=False):
    ranges = []
    for l, c_indices, _, _, _ in crf_data:
        rs = edu_spans[l][c_indices]
        if not hierarchy_ranges:
            rs = rs[-1:]
        ranges.append(rs)
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
         keep_boundary=False, hierarchy_ranges=False,
         hierarchy_adjust=False):

    args.init_truth(corpus_file)
    data_set, test_set = extract_features(corpus_file, args, test_args)

    predictor = Predictor(crf, model_path, train_path, test_path)

    processes = []
    test_crf_data = []
    for i in fhelper.folds():
        crf_data = extract_crf_data(fhelper.train_set(i), data_set)
        if keep_boundary:
            crf_ranges = get_ranges(crf_data)
        else:
            crf_ranges = get_hierarchy_ranges(crf_data, args.edu_spans)
        test_crf_data.append(extract_crf_data(fhelper.test_set(i), test_set))

        processes.append(predictor.train(i, crf_data, crf_ranges))
    for p in processes:
        p.wait()

    print('start testing')
    processes = []
    test_crf_ranges = [None] * len(test_crf_data)
    if keep_boundary:
        test_crf_ranges = [get_ranges(crf_data) for crf_data in test_crf_data]

    for i in fhelper.folds():
        processes.append(
            predictor.test(i, test_crf_data[i], test_crf_ranges[i]))

    preds = []
    pred_probs = []
    for crf_data, ranges, p in zip(test_crf_data, test_crf_ranges, processes):
        p.wait()
        arg_spans, probs = load_predict(p.stdout, ranges)
        assert(len(arg_spans) == len(crf_data))
        preds.append(arg_spans)
        pred_probs.append(probs)

    if not keep_boundary and hierarchy_adjust:
        for i, crf_data, pds, probs in zip(fhelper.folds(),
                                           test_crf_data,
                                           preds,
                                           pred_probs):
            handle_hierarchy_adjust(crf_data, pds, probs, i, predictor)

    # evaluation
    cv_stats = defaultdict(list)
    sum_of_total = 0
    sum_of_i_total = 0
    for i, crf_data, arg_spans in zip(fhelper.folds(), test_crf_data, preds):
        correct = 0
        tp = fp = total = 0
        i_tp = i_fp = i_total = 0
        for arg_span, item in zip(arg_spans, crf_data):
            s = args.edu_truth[item[0]][item[1]]
            tp += len(s & arg_span)
            fp += len(arg_span - s)
            if s == arg_span:
                correct += 1
                if len(s) > 0:
                    i_tp += 1
            else:
                i_fp += 1

        for l in fhelper.test_set(i):
            d = args.edu_truth[l]
            for s in d.values():
                total += len(s)
                if len(s) > 0:
                    i_total += 1
        sum_of_total += total
        sum_of_i_total += i_total
        print('Totally {} arguments'.format(total))
        print('Totally {} instances'.format(i_total))

        accuracy = correct / len(arg_spans) if len(arg_spans) > 0 else 1
        recall = tp / total
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1
        f1 = evaluate.f1(recall, prec)
        cv_stats['Accuracy'].append(accuracy)
        cv_stats['Recall'].append(recall)
        cv_stats['Prec'].append(prec)
        cv_stats['F1'].append(f1)

        recall = i_tp / i_total
        prec = i_tp / (i_tp + i_fp) if (i_tp + i_fp) > 0 else 1
        f1 = evaluate.f1(recall, prec)
        cv_stats['iRecall'].append(recall)
        cv_stats['iPrec'].append(prec)
        cv_stats['iF1'].append(f1)

    print('Accuracy:', np.mean(cv_stats['Accuracy']))
    print('Recall:', np.mean(cv_stats['Recall']))
    print('Prec:', np.mean(cv_stats['Prec']))
    print('F1:', np.mean(cv_stats['F1']))
    print('Instance Recall:', np.mean(cv_stats['iRecall']))
    print('Instance Prec:', np.mean(cv_stats['iPrec']))
    print('Instance F1:', np.mean(cv_stats['iF1']))
    print('Totally {} arguments for all'.format(sum_of_total))
    print('Totally {} instances for all'.format(sum_of_i_total))


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


def get_pred_span(pd):
    return min(pd)[0], max(pd)[-1]


def is_inner(cEDUs, pd, start, end, outer):
    # check if span is less than item
    i_start, i_end = get_pred_span(pd)
    if i_end - i_start >= end - start:
        return None

    # check if connectives are inside of an argument
    for a_start, a_end in outer:
        if all(a_start <= i < a_end for i in cEDUs):
            # check if it exceed the span
            if i_start < a_start or i_end > a_end:
                return (a_start, a_end)
            else:
                return None

    return None


def strip_inner(pd, span):
    for s, e in list(pd):
        if e <= span[0] or s >= span[1]:
            pd.remove((s, e))
        else:
            if s < span[0]:
                pd.remove((s, e))
                s = span[0]
                pd.add((s, e))
            if e > span[1]:
                pd.remove((s, e))
                pd.add((s, span[1]))


def strip_outer(pd, span):
    for s, e in list(pd):
        if s > span[0] and e <= span[1]:
            pd.remove((s, e))
        else:
            if span[0] < s < span[1]:
                pd.remove((s, e))
                pd.add((span[1], e))
            elif span[0] < e < span[1]:
                pd.remove((s, e))
                pd.add((s, span[1]))


def repredict_inner(pd, span, cd, fold, predictor):
    p = predictor.test(fold, [cd], [[span]], wait=True)
    arg_spans, probs = load_predict(p.stdout, [[span]])
    assert(len(arg_spans) == 1)
    assert(len(probs) == 1)
    pd.clear()
    pd.update(arg_spans[0])
    return probs[0]


def reduce_inner(item, preds, cds, items, probs, fold, predictor):
    start, end = get_pred_span(item)
    for pd, cd, j in zip(preds, cds, items):
        cEDUs = cd[2]
        span = is_inner(cEDUs, pd, start, end, item)
        if span is not None:
            strip_inner(pd, span)
            # pr = repredict_inner(pd, span, cd, fold, predictor)
            # if pr is not None:
            #     probs[j] = pr


def reduce_outer(item, preds):
    start, end = get_pred_span(item)
    for pd in preds:
        o_start, o_end = get_pred_span(pd)
        if o_start <= start and o_end >= end:
            strip_outer(pd, (start, end))


def handle_hierarchy_adjust(crf_data, preds, probs, fold, predictor):
    instances = defaultdict(list)
    for i, item in enumerate(crf_data):
        instances[item[0]].append(i)

    for items in instances.values():
        while len(items) > 0:
            idx = pop_max(items, probs)
            pds = get_items(items, preds)
            cds = get_items(items, crf_data)
            reduce_inner(
                preds[idx], pds, cds, items, probs, fold, predictor)
            # reduce_outer(
            #     preds[idx], pds)


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
         keep_boundary, args.hierarchy_ranges, args.hierarchy_adjust)


if __name__ == '__main__':
    main()
