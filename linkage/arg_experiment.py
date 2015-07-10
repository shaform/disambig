"""linkage argument experiments"""
import argparse
import re
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

        return subprocess.Popen([self.crf, 'learn', '-m', mpath, path],
                                stdout=subprocess.DEVNULL)

    def test(self, i, crf_data, crf_ranges, wait=False):
        mpath = '{}.{}'.format(self.model_path, i)
        path = '{}.{}'.format(self.test_path, i)

        self.write_file(path, crf_data, crf_ranges)

        p = subprocess.Popen([self.crf, 'tag', '-pi', '-m', mpath, path],
                             stdout=subprocess.PIPE, universal_newlines=True)
        if wait:
            p.wait()

        return p

CONTEXT = 'CONTEXT'

FILTER_SET = {
    CONTEXT: ('CONTEXT-',)
}

for k, v in FILTER_SET.items():
    FILTER_SET[k] = re.compile('({})'.format('|'.join(v)))


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
    parser.add_argument('--corpus_dep', required=True,
                        help='dep-parsed corpus file')
    parser.add_argument('--linking', required=True)
    parser.add_argument('--bounded', type=int)
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--log', required=True,
                        help='log file to output error analysis')
    parser.add_argument('--crfsuite', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--keep_boundary', action='store_true')
    parser.add_argument('--hierarchy_ranges', action='store_true')
    parser.add_argument('--hierarchy_adjust', action='store_true')
    parser.add_argument('--use_baseline', action='store_true')
    parser.add_argument('--select',
                        help='only select a feature set',
                        choices=tuple(FILTER_SET))
    parser.add_argument('--reverse_select',
                        help='reverse selection',
                        action='store_true')

    return parser.parse_args()


def extract_features(corpus_file, linkings, arguments, test_arguments,
                     use_feature=None, reverse_select=False):
    data_set = defaultdict(list)
    test_set = defaultdict(list)
    counter = evaluate.ProgressCounter(inline=True)

    for l, tokens in corpus_file.corpus.items():
        counter.step()

        pos_tokens = corpus_file.pos_corpus[l]
        parsed = corpus_file.parse_corpus[l]
        deps = corpus_file.dep_corpus[l]
        EDUs = corpus_file.edu_corpus[l]

        for arg in test_arguments.arguments(l):
            a_indices = arguments.get_a_indices(l, arg)
            arg = list(arg)
            arg[-1] = a_indices
            c_indices, cEDU, tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, deps, linkings, arg)
            if use_feature is not None:
                features.filter_features(tfeatures, FILTER_SET[use_feature],
                                         reverse_select=reverse_select)
            test_set[l].append((c_indices, cEDU, tlabels, tfeatures))

        for arg in arguments.arguments(l):
            c_indices, cEDU, tlabels, tfeatures = argument.extract_EDU_features(
                EDUs, tokens, pos_tokens, parsed, deps, linkings, arg)
            if use_feature is not None:
                features.filter_features(tfeatures, FILTER_SET[use_feature],
                                         reverse_select=reverse_select)
            data_set[l].append((c_indices, cEDU, tlabels, tfeatures))
    print()

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


def get_bounded_ranges(crf_data, bounded=None):
    if bounded is None:
        return None

    ranges = []
    for _, _, cEDU, labels, _ in crf_data:
        start = max(0, min(cEDU) - bounded)
        end = min(len(labels), max(cEDU) + 1 + bounded)
        rs = [(start, end)]

        ranges.append(rs)
    return ranges


def get_hierarchy_ranges(crf_data, edu_spans):
    ranges = []
    for l, c_indices, _, _, _ in crf_data:
        rs = edu_spans[l][c_indices]
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


def load_predict(fin, ranges, use_baseline=False, datas=None):
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
            if use_baseline:
                cEDU = datas[idx][3]
                labels = argument.cEDU_to_labels(cEDU, labels)
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


def log_error(log, true_span, predict_span, item, corpus_file, *,
              stats=defaultdict(int)):
    l, c_indices, cEDU, tlabels, tfeatures = item
    tokens = list(corpus_file.corpus[l])
    for indices in c_indices:
        start = indices[0]
        end = indices[-1]
        tokens[start] = '_@' + tokens[start]
        tokens[end] = tokens[start] + '@_'
    edus = corpus_file.EDUs(l, tokens)

    if len(true_span) > 0 and true_span != predict_span:
        log.write('== instance == \n')
        log.write('label: {}\n'.format(l))
        log.write('true: {}\n'.format(list(sorted(true_span))))
        log.write('predict: {}\n'.format(list(sorted(predict_span))))
        log.write('cEDU: {}\n'.format(str(cEDU)))

        starts = set()
        ends = set()
        pstarts = set()
        pends = set()

        for s, e in true_span:
            starts.add(s)
            ends.add(e)

        for s, e in predict_span:
            pstarts.add(s)
            pends.add(e)

        annotated_edus = []
        for i, item in enumerate(edus):
            if i in starts:
                item = '_<' + item
            if i in pstarts:
                item = '_[' + item
            if i + 1 in ends:
                item = item + '_>'
            if i + 1 in pends:
                item = item + '_]'
            annotated_edus.append(item)

        log.write('{}\n'.format(' // '.join(annotated_edus)))

    if len(true_span) > 0 and len(predict_span) > 0:
        stats['length', len(c_indices)] += 1
        d = min(cEDU) - min(min(true_span))
        stats['left expand', len(c_indices), d] += 1
        d = max(max(true_span)) - (max(cEDU) + 1)
        d = (max(max(true_span)) - 1) - max(cEDU)
        stats['right expand', len(c_indices), d] += 1
        stats['max left', 0] = max(stats['max left', 0], min(cEDU))
        stats['max right', 0] = max(
            stats['max right', 0], len(edus) - max(cEDU) - 1)

        if len(c_indices) > 1:
            if min(min(true_span)) < min(min(predict_span)):
                stats['left true < predict', len(c_indices)] += 1

            if min(min(true_span)) > min(min(predict_span)):
                stats['left true > predict', len(c_indices)] += 1

            if max(max(true_span)) < max(max(predict_span)):
                stats['right true < predict', len(c_indices)] += 1

            if max(max(true_span)) > max(max(predict_span)):
                stats['right true > predict', len(c_indices)] += 1


def rule_based(arg_spans, crf_data, corpus_file):
    for arg_span, cdata in zip(arg_spans, crf_data):
        if len(arg_span) == 0:
            continue

        if len(arg_span) == 1:
            arg_span.clear()
            continue

        # l, c_indices, cEDU, tlabels, tfeatures = cdata
        # edus = corpus_file.EDUs(l)
        # start, end = max(arg_span)
        # if end == len(edus) - 1 and min(arg_span)[0] <= 1:
        #     if edus[end].endswith('。') and edus[end - 1].endswith('，'):
        #         arg_span.remove((start, end))
        #         arg_span.add((start, end + 1))

    for arg_span in arg_spans:
        assert(len(arg_span) != 1)

    return arg_spans


def test(fhelper, train_args, test_args, corpus_file,
         linkings,
         train_path, test_path, model_path, crf,
         log_path,
         bounded=None,
         keep_boundary=False, hierarchy_ranges=False,
         hierarchy_adjust=False,
         use_baseline=False,
         use_feature=None,
         reverse_select=False):

    train_args.init_truth(corpus_file)
    data_set, test_set = extract_features(corpus_file,
                                          linkings, train_args, test_args,
                                          use_feature=use_feature,
                                          reverse_select=reverse_select)

    predictor = Predictor(crf, model_path, train_path, test_path)

    processes = []
    test_crf_data = []
    for i in fhelper.folds():
        crf_data = extract_crf_data(fhelper.train_set(i), data_set)
        if keep_boundary:
            crf_ranges = get_ranges(crf_data)
        elif hierarchy_ranges:
            crf_ranges = get_hierarchy_ranges(crf_data, train_args.edu_spans)
        else:
            crf_ranges = get_bounded_ranges(crf_data, bounded)
        test_crf_data.append(extract_crf_data(fhelper.test_set(i), test_set))

        processes.append(predictor.train(i, crf_data, crf_ranges))

    print('training...', end='', flush=True)
    for i, p in enumerate(processes):
        p.wait()
        print(i, end='', flush=True)
    print()

    print('start testing')
    processes = []
    if keep_boundary:
        test_crf_ranges = [get_ranges(crf_data) for crf_data in test_crf_data]
    else:
        test_crf_ranges = [get_bounded_ranges(crf_data, bounded)
                           for crf_data in test_crf_data]

    for i in fhelper.folds():
        processes.append(
            predictor.test(i, test_crf_data[i], test_crf_ranges[i]))

    preds = []
    pred_probs = []
    wow_count = 0
    for crf_data, ranges, p in zip(test_crf_data, test_crf_ranges, processes):
        p.wait()
        arg_spans, probs = load_predict(p.stdout, ranges,
                                        use_baseline, crf_data)
        assert(len(arg_spans) == len(crf_data))
        preds.append(arg_spans)
        pred_probs.append(probs)

    # if not keep_boundary and hierarchy_adjust:
    #    for i, crf_data, pds, probs in zip(fhelper.folds(),
    #                                       test_crf_data,
    #                                       preds,
    #                                       pred_probs):
    #        handle_hierarchy_adjust(crf_data, pds, probs, i, predictor)

    # for arg_spans, crf_data in zip(preds, test_crf_data):
    #    arg_spans = rule_based(arg_spans, crf_data, corpus_file)

    # evaluation
    cv_stats = defaultdict(list)
    log_stats = defaultdict(int)
    with open(log_path, 'w') as log_out:
        for i, crf_data, pds in zip(fhelper.folds(), test_crf_data, preds):
            correct = 0
            tp = fp = total = 0
            i_tp = i_fp = i_total = 0
            ''' use each arg to evaluate
            for arg_span, item in zip(pds, crf_data):
                s = train_args.edu_truth[item[0]][item[1]]
                log_error(log_out, s, arg_span, item, corpus_file,
                          stats=log_stats)
                tp += len(s & arg_span)
                fp += len(arg_span - s)
                if s == arg_span:
                    correct += 1
                    if len(s) > 0:
                        i_tp += 1
                elif len(arg_span) > 0:
                    i_fp += 1

            for l in fhelper.test_set(i):
                d = train_args.edu_truth[l]
                for s in d.values():
                    total += len(s)
                    if len(s) > 0:
                        i_total += 1
            '''

            for arg_span, item in zip(pds, crf_data):
                label, cindices, *_ = item

                s = set()
                if cindices in train_args.argument[label]:
                    rtype = train_args.argument[label][cindices][1]
                    pd_rtype = test_args.argument[label][cindices][1]
                    if rtype == pd_rtype:
                        s = train_args.edu_truth[label][cindices]

                right_rtype = train_args
                log_error(log_out, s, arg_span, item, corpus_file,
                          stats=log_stats)
                truth_boundaries = set()
                for start, end in s:
                    truth_boundaries.add(start)
                    truth_boundaries.add(end)
                assert(len(truth_boundaries) == len(s) + 1
                       or len(s) == len(truth_boundaries) == 0)

                pd_boundaries = set()
                for start, end in arg_span:
                    pd_boundaries.add(start)
                    pd_boundaries.add(end)
                assert(len(pd_boundaries) == len(arg_span) + 1
                       or len(arg_span) == len(pd_boundaries) == 0)

                tp += len(truth_boundaries & pd_boundaries)
                fp += len(pd_boundaries - truth_boundaries)
                if s == arg_span:
                    correct += 1
                    if len(s) > 0:
                        i_tp += 1
                elif len(arg_span) > 0:
                    i_fp += 1

            for l in fhelper.test_set(i):
                d = train_args.edu_truth[l]
                for s in d.values():
                    if len(s) == 0:
                        continue
                    total += len(s) + 1
                    i_total += 1

            assert(
                sum(len(pd) + 1 if len(pd) != 0 else 0 for pd in pds) == tp + fp)
            cv_stats['Total'].append(total)
            cv_stats['iTotal'].append(i_total)
            cv_stats['Propose'].append(tp + fp)
            cv_stats['iPropose'].append(i_tp + i_fp)

            accuracy = correct / len(pds) if len(pds) > 0 else 1
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

    print('Accuracy: {:.2%}'.format(np.mean(cv_stats['Accuracy'])))
    print('Recall: {:.2%}'.format(np.mean(cv_stats['Recall'])))
    print('Prec: {:.2%}'.format(np.mean(cv_stats['Prec'])))
    print('F1: {:.2%}'.format(np.mean(cv_stats['F1'])))
    print('Instance Recall: {:.2%}'.format(np.mean(cv_stats['iRecall'])))
    print('Instance Prec: {:.2%}'.format(np.mean(cv_stats['iPrec'])))
    print('Instance F1: {:.2%}'.format(np.mean(cv_stats['iF1'])))
    print('Totally {} arguments for all'.format(sum(cv_stats['Total'])))
    print('Totally {} instances for all'.format(sum(cv_stats['iTotal'])))
    print('Totally {} arguments predicted for all'.format(
        sum(cv_stats['Propose'])))
    print('Totally {} instances predicted for all'.format(
        sum(cv_stats['iPropose'])))
    print('log stats:')
    # evaluate.print_counts(log_stats)


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
    if len(pd) == 0:
        return None
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
            if len(preds[idx]) == 0:
                continue
            pds = get_items(items, preds)
            cds = get_items(items, crf_data)
            reduce_inner(
                preds[idx], pds, cds, items, probs, fold, predictor)
            # reduce_outer(
            #     preds[idx], pds)


def main():
    args = process_commands()
    linkings = corpus.load_linking(args.linking)
    arguments = argument.ArgumentFile(args.argument)
    test_arguments = argument.ArgumentFile(args.argument_test)
    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse,
        args.corpus_dep)
    fhelper = corpus.FoldsHelper(args.folds)

    keep_boundary = args.keep_boundary and args.argument == args.argument_test

    test(fhelper, arguments, test_arguments,
         corpus_file,
         linkings,
         args.train, args.test, args.model, args.crfsuite,
         args.log,
         args.bounded,
         keep_boundary, args.hierarchy_ranges, args.hierarchy_adjust,
         use_baseline=args.use_baseline,
         use_feature=args.select,
         reverse_select=args.reverse_select
         )


if __name__ == '__main__':
    main()
