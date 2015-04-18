import numpy as np

from collections import defaultdict

from sklearn import metrics


def f1(recall, prec):
    return 2 * recall * prec / (recall + prec)


def print_scores(recall, prec, label=None):
    if label is not None:
        print(label + '\t', end='')

    print('prec: {}\trecall: {}\t f1: {}'.format(
          prec, recall, f1(recall, prec)))

_HEADERS = [
    'causality',
    'coordination',
    'transition',
    'explanation',
    'micro-AVG',
    'macro-AVG',
]

_ALL_HEADERS = [
    'causality',
    'coordination',
    'transition',
    'explanation',
    'non-discourse',
    'micro-AVG',
    'macro-AVG',
    'micro-AVG*',
    'macro-AVG*',
]


def print_sense_scores(Y, Yp, label):
    print()
    print(label, ':')

    print('length', len(Y))
    print('Relation\tPrec\tRecall\tF1\tcases')

    scores = list(metrics.precision_recall_fscore_support(Y, Yp)[:3])
    scores.append([0] * len(scores[0]))
    scores = [list(ss) for ss in scores]
    scores = list(list(ss) for ss in zip(*scores))

    for y in Y:
        scores[y][-1] += 1

    scores.extend([
        [
            metrics.precision_score(Y, Yp, average='micro'),
            metrics.recall_score(Y, Yp, average='micro'),
            metrics.f1_score(Y, Yp, average='micro'),
        ],
        [
            metrics.precision_score(Y, Yp, average='macro'),
            metrics.recall_score(Y, Yp, average='macro'),
            metrics.f1_score(Y, Yp, average='macro'),
        ]
    ])

    if len(scores) == len(_HEADERS):
        headers = _HEADERS
    else:
        headers = _ALL_HEADERS

        # calculate average scores for true connectives only
        rel_scores = scores[:4]

        total_pred = total_positive = positive = 0
        for y, yp in zip(Y, Yp):
            if yp != 4:
                total_pred += 1
            if y != 4:
                total_positive += 1
                if y == yp:
                    positive += 1
        prec = positive / total_pred
        rec = positive / total_positive

        scores.extend([
                      [prec, rec, f1(rec, prec)],
                      np.mean(rel_scores, axis=0)[:3]
                      ])

    for header, score in zip(headers, scores):
        score_line = '\t'.join('{:.02}'.format(s) for s in score[:3])
        if len(score) > 3:
            score_line += '\t{}'.format(score[3])
        print('{}\t{}'.format(header, score_line))


class ProgressCounter(object):

    def __init__(self):
        self.n = 0

    def step(self):
        self.n += 1
        if self.n % 200 == 0:
            print('{} processed'.format(self.n))


def print_stats(tp, tn, fp, fn, label=None):
    if label is not None:
        print('\n' + label)
    print('negative: {}/{}, positive: {}/{}, accuracy: {}'.format(
        tn, tn + fp, tp, tp + fn, (tp + tn) / (tp + tn + fp + fn)))

    print_scores(tp / (tp + fn), tp / (tp + fp))


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


class FoldStats(object):

    def __init__(self, threshold=0.5, show_fold=False):
        self.stats = defaultdict(int)
        self.tp_labels = set()
        self.fp_labels = set()
        self.fn_labels = set()
        self.threshold = threshold
        self.show_fold = show_fold

    def compute_fold(self, labels, Yp, Y):
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

        if self.show_fold:
            print_stats(tp, tn, fp, fn)

        self.stats['tp'] += tp
        self.stats['tn'] += tn
        self.stats['fp'] += fp
        self.stats['fn'] += fn

    def print_total(self, truth_count=None):
        print_stats(self.stats['tp'], self.stats[
                    'tn'], self.stats['fp'], self.stats['fn'], label='Total')

        if truth_count is not None:
            print_stats(self.stats['tp'],
                        self.stats['tn'],
                        self.stats['fp'],
                        truth_count - self.stats['tp'],
                        label='Overall Total')

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
