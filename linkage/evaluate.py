"""routines for evaluation and display"""
import numpy as np

from collections import defaultdict

from sklearn import metrics


def f1(recall, prec):
    if recall + prec == 0:
        return 0
    else:
        return 2 * recall * prec / (recall + prec)


def print_counts(d):
    for l, v in sorted(d.items()):
        print(l, v)


def print_scores(recall, prec, *, accuracy=None, fscore=None, label=None):
    if label is not None:
        print(label + '\t', end='')

    if fscore is None:
        fscore = f1(recall, prec)

    print('prec: {:.04}\trecall: {:.04}\tf1: {:.04}'.format(
          prec, recall, fscore), end='')

    if accuracy is not None:
        print('\taccuracy: {:.04}'.format(accuracy))
    else:
        print()

_HEADERS = [
    [
        'hierarchy',
        'parallel',
        'micro-AVG',
        'macro-AVG',
    ],
    [
        'hierarchy',
        'parallel',
        'non-discourse',
        'micro-AVG',
        'macro-AVG',
    ],
    [
        'causality',
        'coordination',
        'transition',
        'explanation',
        'micro-AVG',
        'macro-AVG',
    ],
    [
        'causality',
        'coordination',
        'transition',
        'explanation',
        'non-discourse',
        'micro-AVG',
        'macro-AVG',
    ],
    [
        # 因果关系
        'cause-result',
        # 推断关系
        #'inference',
        # 假设关系
        'hypothetical',
        # 目的关系
        'purpose',
        # 条件关系
        'condition',
        # 背景关系
        #'background',
        # 并列关系
        'coordination',
        # 顺承关系
        'continue',
        # 递进关系
        'progressive',
        # 选择关系
        'selectional',
        # 对比关系
        'inverse',
        # 转折关系
        'transition',
        # 让步关系
        'concessive',
        # 解说关系
        'explanation',
        # 总分关系
        'summary-elaboration',
        # 例证关系
        'example',
        # 评价关系
        #'evaluation',
        'micro-AVG',
        'macro-AVG',
    ],
]


def compute_cv_sense_scores(Ys, Yps, labels=None, non_dis=None):
    total_scores = []
    raw_scores = []
    for Y, Yp in zip(Ys, Yps):
        # get list of p, r, f, each list contains score for each class
        scores = list(metrics.precision_recall_fscore_support(
            Y,
            Yp,
            labels=labels)[:3])
        # convert to list
        scores = [list(ss) for ss in scores]
        # convert to list of (p, r, f)
        if non_dis is not None:
            assert(non_dis + 1 == len(scores[0]))
            raw_score = [np.mean(ss[:-1]) for ss in scores]
        else:
            raw_score = [np.mean(ss) for ss in scores]
        scores = list(list(ss) for ss in zip(*scores))
        total_scores.append(scores)
        raw_scores.append(raw_score)

    total_scores = [list(ss) for ss in list(np.mean(total_scores, axis=0))]

    # add one column for num
    for scores in total_scores:
        scores.append(0)

    return total_scores, raw_scores


def compute_indomain_micro(Y, Yp, non_dis):
    total_pred = total_positive = correct = 0
    for y, yp in zip(Y, Yp):
        if yp != non_dis:
            total_pred += 1
        if y != non_dis:
            total_positive += 1
            if y == yp:
                correct += 1
    prec = correct / total_pred
    recall = correct / total_positive
    fscore = f1(prec, recall)

    return prec, recall, fscore


def compute_indomain_macro(Y, Yp, non_dis):
    pred, positive, correct = range(3)

    scores = [[0, 0, 0] for _ in range(non_dis)]
    for y, yp in zip(Y, Yp):
        if yp != non_dis:
            scores[yp][pred] += 1
        if y != non_dis:
            scores[y][positive] += 1
            if y == yp:
                scores[y][correct] += 1

    prec = recall = fscore = 0
    for pd, pos, c in scores:
        p = c / pd if pd != 0 else 0
        r = c / pos if pos != 0 else 0
        prec += p
        recall += r
        fscore += f1(p, r)

    return prec / non_dis, recall / non_dis, fscore / non_dis


def compute_cv_average(Ys, Yps, average='micro', non_dis=None):
    prec = recall = fscore = 0
    n = len(Ys)
    for Y, Yp in zip(Ys, Yps):
        if non_dis is None:
            p, r, f, _ = metrics.precision_recall_fscore_support(Y, Yp,
                                                                 average=average)
        elif average == 'micro':
            p, r, f = compute_indomain_micro(Y, Yp, non_dis)
        else:
            p, r, f = compute_indomain_macro(Y, Yp, non_dis)
        prec += p
        recall += r
        fscore += f

    prec /= n
    recall /= n
    fscore /= n

    return prec, recall, fscore


def compute_cv_accuracy(Ys, Yps):
    accuracy = 0
    n = len(Ys)
    for Y, Yp in zip(Ys, Yps):
        accuracy += metrics.accuracy_score(Y, Yp)
    accuracy /= n
    return accuracy


def print_sense_scores2(Ys, Yps, label, non_dis=None):
    print()
    print(label, ':')

    # compute macro average
    macro_p, macro_r, macro_f = compute_cv_average(Ys, Yps, 'macro')
    micro_p, micro_r, micro_f = compute_cv_average(Ys, Yps, 'micro')
    accuracy = compute_cv_accuracy(Ys, Yps)

    print('Accuracy = {:.04}'.format(accuracy))
    print('length', sum(len(y) for y in Ys))
    print('Relation\tPrec\tRecall\tF1')
    print('Macro   \t{:.04}\t{:.04}\t{:.04}'.format(macro_p, macro_r, macro_f))
    print('Micro   \t{:.04}\t{:.04}\t{:.04}'.format(micro_p, micro_r, micro_f))

    if non_dis is not None:
        micro_p, micro_r, micro_f = compute_cv_average(Ys, Yps, 'micro',
                                                       non_dis=non_dis)
        macro_p, macro_r, macro_f = compute_cv_average(Ys, Yps, 'macro',
                                                       non_dis=non_dis)
        print('Macro*  \t{:.04}\t{:.04}\t{:.04}'.format(
            macro_p, macro_r, macro_f))
        print('Micro*  \t{:.04}\t{:.04}\t{:.04}'.format(
            micro_p, micro_r, micro_f))


def print_sense_scores(Ys, Yps, label,
                       print_accuracy=False, non_dis=None, labels=None):
    print()
    print(label, ':')

    # calculate accuracy
    if print_accuracy:
        accuracy = compute_cv_accuracy(Ys, Yps)
        print('Accuracy = {:.04}'.format(accuracy))

    micro_p, micro_r, micro_f = compute_cv_average(Ys, Yps, 'micro')

    scores, raw_scores = compute_cv_sense_scores(Ys, Yps, labels=labels,
                                                 non_dis=non_dis)
    Y = list(np.concatenate(Ys, axis=0))
    Yp = list(np.concatenate(Yps, axis=0))

    print('length', len(Y))
    print('Relation\tPrec\tRecall\tF1\tcases')

    # count num of cases
    for y in Y:
        scores[y][-1] += 1

    scores.extend([
        # micro averaging
        [
            micro_p, micro_r, micro_f
        ],
        # macro averaging
        np.mean(scores, axis=0)[:3]
    ])

    headers = None
    for headers_ in _HEADERS:
        if len(scores) == len(headers_):
            headers = headers_

    assert(headers is not None)

    if non_dis is not None:
        headers = list(headers)
        headers.extend(['micro-AVG*', 'macro-AVG*'])

        # calculate average scores for true connectives only
        rel_scores = scores[:-3]
        p, r, f = compute_cv_average(Ys, Yps, 'micro', non_dis)

        scores.extend([
                      # micro averaging
                      [p, r, f],
                      # macro averaging
                      np.mean(rel_scores, axis=0)[:3]
                      ])

    for header, score in zip(headers, scores):
        score_line = '\t'.join('{:.04}'.format(s) for s in score[:3])
        # append num of cases
        if len(score) > 3:
            score_line += '\t{}'.format(score[3])
        print('{}\t{}'.format(header, score_line))
    print('\nraw scores')
    print(raw_scores)


class ProgressCounter(object):

    def __init__(self, inline=False):
        self.n = 0
        self.inline = inline
        if inline:
            print('processing...', end='', flush=True)

    def step(self):
        self.n += 1
        if self.n % 200 == 0:
            if self.inline:
                print('', self.n, '/', end='', flush=True)
            else:
                print('{} processed'.format(self.n))


def print_stats(tp, tn, fp, fn, label=None, print_prf=True):
    if label is not None:
        print('\n' + label)
    print('negative: {}/{}, positive: {}/{}, accuracy: {}'.format(
        tn, tn + fp, tp, tp + fn, (tp + tn) / (tp + tn + fp + fn)))
    print('total predicted: {}'.format(tp + fp))

    if print_prf:
        print_scores(tp / (tp + fn), tp / (tp + fp))


class WordCount(object):

    def __init__(self, path):
        self.counts = defaultdict(int)
        with open(path) as f:
            for l in f:
                label, v = l.split('\t')
                self.counts[label] = int(v)

    def count_fold(self, labels):
        total = 0
        for l in labels:
            total += self.counts[l]
        return total


class WordAmbig(object):

    def __init__(self, path):
        self.ambig = {}
        self.total_ambig = {}
        with open(path, 'r') as f:
            for l in f:
                label, word, ambig, inside = l.split('\t')
                if int(inside) == 1:
                    self.ambig[(label, word)] = int(ambig)
                self.total_ambig[(label, word)] = int(ambig)

    def __len__(self):
        return len(self.total_ambig)

    def count_fold(self, labels):
        total = 0
        labels = set(labels)
        for label, _ in self.total_ambig:
            if label in labels:
                total += 1
        return total


class FoldStats(object):

    """Compute scores for each fold and output average"""

    def __init__(self, threshold=0.5, show_fold=False, label=None):
        self.stats = defaultdict(int)
        self.cv_stats = defaultdict(list)
        self.tp_labels = set()
        self.fp_labels = set()
        self.fn_labels = set()
        self.threshold = threshold
        self.show_fold = show_fold
        self.label = label

    def compute_fold(self, labels, Yp, Y, truth_count=None, total_count=None):
        f_stats = defaultdict(int)

        tp = tn = fp = fn = 0
        for l, yp, y in zip(labels, Yp, Y):
            yp = yp >= self.threshold

            if yp == y:
                if yp == 1:
                    tp += 1
                    self.tp_labels.add(l)
                else:
                    tn += 1
            else:
                if yp == 1:
                    fp += 1
                    self.fp_labels.add(l)
                else:
                    fn += 1
                    self.fn_labels.add(l)

        if truth_count is not None:
            fn = truth_count - tp

        if total_count is not None:
            tn = total_count - (tp + fp + fn)

        if self.show_fold:
            if self.label is not None:
                print('\n{}'.format(self.label))
            print_stats(tp, tn, fp, fn)

        self.stats['tp'] += tp
        self.stats['tn'] += tn
        self.stats['fp'] += fp
        self.stats['fn'] += fn

        recall = tp / (tp + fn) if tp + fn != 0 else 0
        prec = tp / (tp + fp) if tp + fp != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.cv_stats['prec'].append(prec)
        self.cv_stats['recall'].append(recall)
        self.cv_stats['f1'].append(f1(recall, prec))
        self.cv_stats['accuracy'].append(accuracy)

    def print_total(self, truth_count=None):

        if truth_count is None:
            print_stats(self.stats['tp'],
                        self.stats['tn'],
                        self.stats['fp'],
                        self.stats['fn'],
                        label='Total',
                        print_prf=False)

        else:
            print_stats(self.stats['tp'],
                        self.stats['tn'],
                        self.stats['fp'],
                        truth_count - self.stats['tp'],
                        label='Overall Total',
                        print_prf=False)

        self.print_cv_total()

    def print_cv_total(self):
        print('\nCV macro:')
        l = len(self.cv_stats['prec'])
        prec = sum(self.cv_stats['prec']) / l
        recall = sum(self.cv_stats['recall']) / l
        fscore = sum(self.cv_stats['f1']) / l
        accuracy = sum(self.cv_stats['accuracy']) / l
        print_scores(recall, prec, fscore=fscore, accuracy=accuracy)

    def print_distribution(self, ambig, function=lambda x: x):
        '''Display correct word distribution by ambiguity'''
        print('Answer distribution by ambiguity of words:')
        d = defaultdict(int)
        t = defaultdict(int)

        correct_words = function(self.tp_labels)

        for l, v in ambig.ambig.items():
            d[v] += 1
            if l in correct_words:
                t[v] += 1

        for v in sorted(d):
            print('{}: {}/{}'.format(v, t[v], d[v]))

    def count_by(self, function=lambda x: len(x[1]), label=''):
        print('Count by {}:'.format(label))

        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        keys = set()
        for d, ls in zip((tp, fp, fn), (
                self.tp_labels,
                self.fp_labels,
                self.fn_labels)):
            for x in ls:
                key = function(x)
                d[key] += 1
                keys.add(key)

        for key in sorted(keys):
            t = tp[key] + fn[key]
            r = tp[key] / t
            p = tp[key] + fp[key]
            p = 1 if p == 0 else tp[key] / p
            print('{}:\t{}+{}/{}\tr:{} p:{}'.format(
                key,
                tp[key],
                fp[key],
                t,
                r,
                p
            ))
