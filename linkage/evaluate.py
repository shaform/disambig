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


def print_scores(recall, prec, *, fscore=None, label=None):
    if label is not None:
        print(label + '\t', end='')

    if fscore is None:
        fscore = f1(recall, prec)

    print('prec: {:.04}\trecall: {:.04}\tf1: {:.04}'.format(
          prec, recall, fscore))

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
    ]
]


def compute_cv_sense_scores(Ys, Yps):
    total_scores = []
    for Y, Yp in zip(Ys, Yps):
        # get list of p, r, f
        scores = list(metrics.precision_recall_fscore_support(Y, Yp)[:3])
        # convert to list
        scores = [list(ss) for ss in scores]
        # convert to list of (p, r, f)
        scores = list(list(ss) for ss in zip(*scores))
        total_scores.append(scores)

    total_scores = [list(ss) for ss in list(np.mean(total_scores, axis=0))]

    # add one column for num
    for scores in total_scores:
        scores.append(0)

    return total_scores


def print_sense_scores(Ys, Yps, label):
    print()
    print(label, ':')

    scores = compute_cv_sense_scores(Ys, Yps)
    Y = list(np.concatenate(Ys, axis=0))
    Yp = list(np.concatenate(Yps, axis=0))

    print('length', len(Y))
    print('Relation\tPrec\tRecall\tF1\tcases')

    # count num
    for y in Y:
        scores[y][-1] += 1

    scores.extend([
        [
            metrics.precision_score(Y, Yp, average='micro'),
            metrics.recall_score(Y, Yp, average='micro'),
            metrics.f1_score(Y, Yp, average='micro'),
        ],
        np.mean(scores, axis=0)[:3]
    ])

    headers = None
    for headers_ in _HEADERS:
        if len(scores) == len(headers_):
            headers = headers_

    assert(headers is not None)
    if len(scores) % 2:
        headers = list(headers)
        headers.extend(['micro-AVG*', 'macro-AVG*'])

        # calculate average scores for true connectives only
        rel_scores = scores[:-3]

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
        score_line = '\t'.join('{:.04}'.format(s) for s in score[:3])
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
    print('total predicted: {}'.format(tp + fp))

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

    def count_fold(self, labels):
        total = 0
        labels = set(labels)
        for label, _ in self.total_ambig:
            if label in labels:
                total += 1
        return total


class ArgStats(object):
    pass


class FoldStats(object):

    def __init__(self, threshold=0.5, show_fold=False):
        self.stats = defaultdict(int)
        self.cv_stats = defaultdict(list)
        self.tp_labels = set()
        self.fp_labels = set()
        self.fn_labels = set()
        self.threshold = threshold
        self.show_fold = show_fold

    def compute_fold(self, labels, Yp, Y, truth_count=None):
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

        if truth_count is not None:
            fn = truth_count - tp

        recall, prec = tp / (tp + fn), tp / (tp + fp)
        self.cv_stats['prec'].append(prec)
        self.cv_stats['recall'].append(recall)
        self.cv_stats['f1'].append(f1(recall, prec))

    def print_total(self, truth_count=None):

        if truth_count is None:
            print_stats(self.stats['tp'],
                        self.stats['tn'],
                        self.stats['fp'],
                        self.stats['fn'],
                        label='Total')

        else:
            print_stats(self.stats['tp'],
                        self.stats['tn'],
                        self.stats['fp'],
                        truth_count - self.stats['tp'],
                        label='Overall Total')

        self.print_cv_total()

    def print_cv_total(self):
        print('\nCV macro:')
        l = len(self.cv_stats['prec'])
        prec = sum(self.cv_stats['prec']) / l
        recall = sum(self.cv_stats['recall']) / l
        fscore = sum(self.cv_stats['f1']) / l
        print_scores(recall, prec, fscore=fscore)

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
