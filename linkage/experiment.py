"""Main program for linkage experiments"""
import argparse

import corpus
import evaluate
import features
import linkage

from collections import defaultdict
from multiprocessing import Pool

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--connective', required=True,
                        help='connective token file')
    parser.add_argument('--word_ambig', required=True,
                        help='word ambiguity file')
    parser.add_argument('--word_probs', required=True,
                        help='word probability file')
    parser.add_argument('--word_count', required=True,
                        help='word count file')
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--linkage_features', required=True,
                        help='linkage features file')
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')
    parser.add_argument('--corpus_pos', required=True,
                        help='pos-tagged raw corpus file')
    parser.add_argument('--corpus_parse', required=True,
                        help='syntax-parsed raw corpus file')
    parser.add_argument('--linkage_probs', required=True,
                        help='linkage probability file')
    parser.add_argument('--linkage_class', required=True,
                        help='linkage class file')
    parser.add_argument('--perfect', action='store_true',
                        help='whether to do perfect experiment')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--arg_output')
    parser.add_argument('--greedy', action='store_true')
    parser.add_argument('--rank', nargs='*', default=[],
                        choices=('length', 'score'))
    parser.add_argument('--predict_sense', action='store_true')
    parser.add_argument('--predict_wstats', action='store_true')

    return parser.parse_args()


def count_fold(cdict, labels):
    total = 0
    for l in labels:
        total += cdict[l]
    return total


def count_linkage(path):
    total_count = 0
    count_by_label = defaultdict(int)
    with open(path, 'r') as f:
        for l in f:
            label = l.split('\t', 1)[0]
            total_count += 1
            count_by_label[label] += 1

    return total_count, count_by_label


def load_linkage_probs(path):
    linkage_probs = {}
    with open(path, 'r') as f:
        for l in f:
            label, tlabel, truth, prob = l.split('\t')
            words = tuple(tlabel.split('-'))
            linkage_probs[(label, words)] = float(prob)
    return linkage_probs


def load_word_probs(path):
    word_probs = {}
    truth_probs = {}
    with open(path, 'r') as f:
        for l in f:
            label, indices, truth, prob = l.split('\t')
            word_probs[(label, indices)] = float(prob)
            truth_probs[(label, indices)] = int(truth)
    return word_probs, truth_probs


def compute_ranking_probs(linkage_probs, key=None):
    if key is None:
        key = lambda x: linkage_probs[x]
    ranking_probs = {}
    for label in linkage_probs:
        words = label[1]
        token_lst = linkage.list_of_token_indices(words)
        ranking_probs[label] = (
            # len(words),
            key(label))

    return ranking_probs


def detect_wrong(indices, visited, crossed):
    '''
    if len(indices) > 1:
        pure_indices = tuple(next(linkage.token_indices(token))
                             for token in indices)
        for l, r in zip(pure_indices, pure_indices[1:]):
            for prev_indices in crossed:
                if any((l < pl < r < pr or
                        pl < l < pr < r)
                        for pl, pr in zip(prev_indices,
                                          prev_indices[1:])):
                    return True

        crossed.add(pure_indices)
    '''

    all_indices = set()

    for token in indices:
        token_indices = tuple(linkage.token_indices(token))

        for idx in token_indices:
            if idx in visited:
                return True
            else:
                all_indices.add(idx)

    # correct for these indices
    visited.update(all_indices)

    return False


def collect_sense_data(fhelper, data_set, feature_tbl, truth):
    X = []
    Y = []
    for (label, indices), x, y in zip(*fhelper.features(
            data_set, feature_tbl)):
        if indices in truth[label]:
            X.append(x)
            Y.append(truth[label][indices])
    return X, Y


def train_sense_lr(args):
    X, Y = args
    lr = LogisticRegression()
    lr = LogisticRegressionCV()
    lr.fit(X, Y)
    print('.', end='', flush=True)
    return lr


def train_sense_lr_simple(args):
    X, Y = args
    lr = LogisticRegression()
    lr = GaussianNB()
    lr.fit(X, Y)
    print('.', end='', flush=True)
    return lr

NON_DIS = 4
NON_DIS_2 = 17


def predict_sense(labels, Yp, Y, lr, feature_set, truth, non_dis=NON_DIS):
    sX = []
    sY = []
    slabels = []

    for label, yp, y in zip(labels, Yp, Y):
        # only predict extracted connectives
        if yp == 1:
            sX.append(feature_set[label])
            if y == 0:
                # non-discourse
                sY.append(non_dis)
            else:
                sY.append(truth[label[0]][label[1]])

            slabels.append(label)

    sYp = lr.predict(sX)
    return slabels, list(sYp), sY


def append_sense_items(slabels, sYp, sY, feature_tbl, truth, plabels,
                       non_dis=NON_DIS):
    s = set(slabels)

    wsYp = []
    wsY = []

    wlabels = {}
    pwlabels = defaultdict(lambda: non_dis)
    for label, types in truth.items():
        if label in plabels:
            for l, t in types.items():
                for windex in l:
                    wlabels[(label, windex)] = t

    for (label, indices), yp, y in zip(slabels, sYp, sY):
        for windex in indices:
            wlabel = (label, windex)
            assert(wlabel not in pwlabels)
            if yp != non_dis:
                pwlabels[wlabel] = yp

    for wlabel, y in wlabels.items():
        wsY.append(y)
        wsYp.append(pwlabels[wlabel])

    for wlabel, yp in pwlabels.items():
        if wlabel not in wlabels:
            wsY.append(non_dis)
            wsYp.append(yp)

    for label, all_features in feature_tbl.items():
        if label in plabels:
            for l, y, x in all_features:
                if (label, l) not in s:
                    s.add((label, l))

                    if y == 0:
                        # non-discourse
                        ctype = non_dis
                    else:
                        ctype = truth[label][l]

                    sY.append(ctype)
                    sYp.append(non_dis)

    for label, types in truth.items():
        if label in plabels:
            for l, t in types.items():
                if (label, l) not in s:
                    s.add((label, l))
                    sY.append(t)
                    sYp.append(non_dis)

    return wsYp, wsY


def get_feature_set(feature_tbl):
    d = {}
    for label, all_features in feature_tbl.items():
        for l, _, x in all_features:
            d[(label, l)] = x
    return d


def train_sense_predictors(num_of_folds, fhelper, feature_tbl, truth):
    print('training predictors...', end='', flush=True)

    predictors = []
    predictors2 = []
    for i in range(num_of_folds):

        X, Y = collect_sense_data(fhelper,
                                  fhelper.train_set(i),
                                  feature_tbl,
                                  truth.linkage_type)
        predictors.append((X, Y))

        X2, Y2 = collect_sense_data(fhelper,
                                    fhelper.train_set(i),
                                    feature_tbl,
                                    truth.linkage_type2)
        predictors2.append((X2, Y2))

    with Pool(num_of_folds * 2) as p:
        predictors = p.map_async(train_sense_lr, predictors)
        predictors2 = p.map_async(train_sense_lr_simple, predictors2)
        predictors = predictors.get()
        predictors2 = predictors2.get()

    print('done')
    return predictors, predictors2


class Ranker(object):

    def __init__(self, markers, *, probs=None, label, method, truth, rank):
        self.markers = markers
        self.probs = probs
        self.label = label
        self.method = method
        self.truth = truth

        if method == 'greedy':
            self.min_ambig = defaultdict(int)
            self.ambig_comp = defaultdict(int)
            self.ambig_dict = defaultdict(set)
            for m in markers:
                for c in m:
                    self.ambig_comp[c] += 1
                    self.ambig_dict[c].add(m)
            for m in markers:
                self.update_ambig(m)

            self.min_ambig = dict(self.min_ambig)
            self.ambig_comp = dict(self.ambig_comp)
            self.ambig_dict = dict(self.ambig_dict)

            if rank is None or len(rank) == 0:
                self.k = lambda x: (
                    self.min_ambig[x],
                    x)
            elif rank == ['length', 'score']:
                self.k = lambda x: (
                    self.min_ambig[x],
                    -len(x),
                    -self.probs[(label, x)],
                    x)
            elif rank == ['score']:
                self.k = lambda x: (
                    self.min_ambig[x],
                    -self.probs[(label, x)],
                    x)
            elif rank == ['length']:
                self.k = lambda x: (
                    self.min_ambig[x],
                    -len(x),
                    x)
            else:
                assert(False)

        elif method == 'length-prob':
            assert(probs is not None)
            if rank is None or len(rank) == 0:
                self.k = lambda x: (
                    x,)
            elif rank == ['length', 'score']:
                self.k = lambda x: (
                    -len(x),
                    -self.probs[(label, x)],
                    x)
            elif rank == ['score']:
                self.k = lambda x: (
                    -self.probs[(label, x)],
                    x)
            elif rank == ['length']:
                self.k = lambda x: (
                    -len(x),
                    x)
            else:
                assert(False)
        else:
            assert(False)

        self.sort()

    def update_ambig(self, m):
        bs = [self.ambig_comp[c] for c in m]
        mini = min(bs)
        maxi = max(bs)
        avg = sum(bs) / len(m)

        outer = set()
        for c in m:
            outer |= self.ambig_dict[c]
        outer.remove(m)

        stats = defaultdict(int)
        for n in outer:
            for c in n:
                stats[c] += 1

        allowed = 1 if mini > 1 else 0
        '''
        if allowed == 1:
            for c, v in stats.items():
                if self.ambig_comp[c] <= v:
                    assert(self.ambig_comp[c] == v)
                    allowed = 2
                    break
        '''

        self.min_ambig[m] = (allowed, 0)

    def sort(self):
        label = self.label

        if self.method == 'length-prob':
            self.markers.sort(
                key=self.k,
                reverse=True
            )
        elif self.method == 'greedy':
            self.markers.sort(
                key=self.k,
                reverse=True
            )

    def generate(self):
        while len(self.markers) > 0:
            m = self.markers.pop()
            yield m

            if self.method == 'greedy':
                new_markers = set(self.markers)

                # collect overlapped markers
                modified = set()
                for c in m:
                    self.ambig_comp[c] -= 1
                    if self.ambig_comp[c] == 0:
                        del self.ambig_dict[c]
                    else:
                        self.ambig_dict[c].remove(m)
                        modified |= self.ambig_dict[c]
                for m in modified:
                    new_markers.remove(m)
                    del self.min_ambig[m]
                    for c in m:
                        self.ambig_dict[c].remove(m)
                        self.ambig_comp[c] -= 1
                for m in new_markers:
                    self.update_ambig(m)
                self.markers = list(new_markers)
                self.sort()


def cross_validation(corpus_file, fhelper, feature_tbl, truth, detector,
                     linkage_counts, lcdict, linkage_probs, word_ambig,
                     cut, *, words=None, perfect=False, arg_output=None,
                     predict_sstats=True,
                     predict_pstats=False,
                     predict_wstats=False,
                     count_path,
                     rank=None,
                     greedy=False):
    word_count = evaluate.WordCount(count_path)
    stats = evaluate.FoldStats(show_fold=True, label='linkage stats')
    pstats = evaluate.FoldStats(show_fold=False)
    wstats = evaluate.FoldStats(show_fold=True, label='word stats')
    rejected_ov = defaultdict(int)
    rejected_s = defaultdict(int)
    pType = {}

    if not perfect and predict_sstats:
        num_of_folds = len(fhelper.folds())
        predictors, predictors2 = train_sense_predictors(num_of_folds,
                                                         fhelper,
                                                         feature_tbl,
                                                         truth)

        # compute word sense statistics
        all_wsYp = []
        all_wsY = []
        # compute sense statistics
        all_slabels = []
        all_sYp = []
        all_sY = []
        # compute level-2 sense statistics
        all_ssYp = []
        all_ssY = []
        feature_set = get_feature_set(feature_tbl)

        if arg_output is not None:
            arg_output = open(arg_output, 'w')

    total_detected = 0
    print('\npredict for fold...', end='', flush=True)
    for i in fhelper.folds():
        print(i, end='', flush=True)

        # compute linkage statistics
        labels = []
        Yp = []
        Y = []

        # compute paragraph statistics
        plabels = list(fhelper.test_set(i))
        pYp = []
        pY = []

        # compute word statistics
        wlabels = []
        wYp = []
        wY = []

        cnnct_dict = {}

        has_words = set()
        for label in plabels:
            tokens = corpus_file.corpus[label]

            if perfect:
                truth_connectives = truth[label]
                all_tokens = detector.perfect_tokens(tokens,
                                                     truth=truth_connectives)

            else:
                all_tokens = detector.all_tokens(tokens,
                                                 continuous=True,
                                                 cross=False)

            markers = []
            ambig_count = defaultdict(int)
            cand_words = set()
            for cnnct, indices in all_tokens:
                markers.append(indices)
                if indices in truth[label]:
                    total_detected += 1
                cnnct_dict[(label, indices)] = cnnct

                # add words
                for windex in indices:
                    cand_words.add(windex)

                # count ambig
                for index_list in linkage.list_of_token_indices(indices):
                    for index in index_list:
                        ambig_count[index] += 1

            for windex in cand_words:
                wlabels.append((label, windex))
                if (label, windex) in words:
                    wY.append(1)
                else:
                    wY.append(0)
            # if features.num_of_sentences(tokens) == 1 and any([c > 1 for c in ambig_count.values()]):
            #    print(label, tokens)
            is_ambig_sent = any([c > 1 for c in ambig_count.values()])
            if is_ambig_sent:
                pType[label] = (features.num_of_sentences(tokens), 'ambig')
            else:
                pType[label] = (features.num_of_sentences(tokens), 'unique')

            if greedy:
                rmethod = 'greedy'
            else:
                rmethod = 'length-prob'
            ranker = Ranker(markers,
                            probs=linkage_probs, method=rmethod,
                            label=label,
                            truth=truth[label],
                            rank=rank)

            visited = set()
            crossed = set()
            correct = 0
            for indices in ranker.generate():
                if indices in truth[label]:
                    Y.append(1)
                else:
                    Y.append(0)
                labels.append((label, indices))

                if cut((label, indices), linkage_probs):
                    Yp.append(0)
                    rejected_s[len(indices)] += Y[-1] == 1
                    continue
                if detect_wrong(indices, visited, crossed):
                    Yp.append(0)
                    rejected_ov[len(indices)] += Y[-1] == 1
                    continue

                Yp.append(1)

                for windex in indices:
                    has_words.add((label, windex))

                if Yp[-1] == Y[-1] == 1:
                    correct += 1

                ilen = len(indices)

            if predict_pstats:
                if features.num_of_sentences(tokens) == 1 and is_ambig_sent:
                    # if is_ambig_sent:
                    # if True:
                    pY.append(1)
                    if correct == len(truth[label]):
                        pYp.append(1)
                    else:
                        pYp.append(0)

        stats.compute_fold(labels, Yp, Y,
                           truth_count=count_fold(lcdict, plabels))

        if predict_pstats:
            pstats.compute_fold(plabels, pYp, pY)
        for w in wlabels:
            if w in has_words:
                wYp.append(1)
            else:
                wYp.append(0)

        total_count = word_count.count_fold(fhelper.test_set(i))
        wstats.compute_fold(wlabels, wYp, wY,
                            truth_count=word_ambig.count_fold(plabels),
                            total_count=total_count)

        if not perfect and predict_sstats:
            lr = predictors[i]
            slabels, sYp, sY = predict_sense(labels, Yp, Y, lr,
                                             feature_set,
                                             truth.linkage_type)
            wsYp, wsY = append_sense_items(slabels, sYp, sY, feature_tbl,
                                           truth.linkage_type, plabels)

            all_slabels.append(slabels)
            all_sYp.append(sYp)
            all_sY.append(sY)
            all_wsYp.append(wsYp)
            all_wsY.append(wsY)

            lr = predictors2[i]
            sslabels, ssYp, ssY = predict_sense(labels, Yp, Y, lr,
                                                feature_set,
                                                truth.linkage_type2,
                                                non_dis=NON_DIS_2)
            assert(slabels == sslabels)
            append_sense_items(sslabels, ssYp, ssY, feature_tbl,
                               truth.linkage_type2, plabels,
                               non_dis=NON_DIS_2)

            if arg_output is not None:
                for (l, cl), rt, st in zip(slabels, sYp, ssYp):
                    cnnct = cnnct_dict[(l, cl)]
                    arg_output.write('{}\t{}\t{}\t{}\t{}\t\t\n'.format(
                        l,
                        '-'.join(cnnct),
                        '-'.join(cl),
                        rt,
                        st
                    ))

            all_ssYp.append(ssYp)
            all_ssY.append(ssY)

    print('\n== done ==')

    print('\nLinkage stats:')
    stats.print_total(truth_count=linkage_counts)
    print('total detected = {}'.format(total_detected))
    stats.print_distribution(
        word_ambig,
        function=lambda x: {(l, w) for (l, ws) in x for w in ws})
    stats.count_by(label='length')
    print('rejected overlapped:', rejected_ov, 'rejected scores:', rejected_s)

    if predict_pstats:
        print('\nParagraph stats:')
        pstats.print_total()
        pstats.count_by(function=pType.get, label='Sentence Length')
        pstats.count_by(function=lambda x: pType[x][1], label='Ambiguity')

    if not perfect and predict_wstats:
        print('\nWord stats:')
        wstats.print_total(truth_count=len(words))

    if not perfect and predict_sstats:
        print('\n=== 1-level sense stats ===')
        evaluate.print_sense_scores(
            all_sY, all_sYp, 'Overall', non_dis=NON_DIS)

        print('\n=== 1-level sense stats for words ===')
        evaluate.print_sense_scores(
            all_wsY, all_wsYp, 'Overall', non_dis=NON_DIS)

        print('\n=== 2-level sense stats ===')
        evaluate.print_sense_scores2(
            all_ssY, all_ssYp, 'Overall', non_dis=NON_DIS_2)

        if arg_output is not None:
            arg_output.close()


def main():
    args = process_commands()

    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    fhelper = corpus.FoldsHelper(args.folds)
    truth = linkage.LinkageFile(args.linkage)
    words = truth.all_words()
    detector = linkage.LinkageDetector(args.connective)
    feature_tbl = features.load_features_table(
        args.linkage_features, lambda x: tuple(x.split('-')))

    linkage_counts, lcdict = count_linkage(args.linkage)
    linkage_probs = load_linkage_probs(args.linkage_probs)
    linkage_class = load_linkage_probs(args.linkage_class)

    word_ambig = evaluate.WordAmbig(args.word_ambig)

    '''
    print('score model')
    cross_validation(
        corpus_file,
        fhelper,
        truth,
        detector,
        linkage_counts,
        linkage_probs,
        word_ambig,
        cut=lambda x, y: linkage_probs[x] < 0.7)
    '''

    ranking_probs = compute_ranking_probs(linkage_probs)

    if args.perfect:
        cut = lambda x, _: any((x[0], w) not in words for w in x[1])
    else:
        # cut = lambda x, _: linkage_probs[x] < args.threshold
        cut = lambda x, _: linkage_class[x] < args.threshold

    print('===== ranking model =====')
    cross_validation(
        corpus_file,
        fhelper,
        feature_tbl,
        truth,
        detector,
        linkage_counts,
        lcdict,
        ranking_probs,
        word_ambig,
        cut=cut,
        words=words,
        perfect=args.perfect,
        count_path=args.word_count,
        arg_output=args.arg_output,
        greedy=args.greedy,
        rank=args.rank,
        predict_sstats=args.predict_sense,
        predict_wstats=args.predict_wstats,
    )

    if not args.perfect:
        word_probs, word_truth = load_word_probs(args.word_probs)
        # cut by word probs
        cut = lambda x, _: any(
            word_probs[(x[0], w)] < args.threshold for w in x[1])  # or linkage_class[x] < args.threshold
        print('\n===== pipeline model =====')
        cross_validation(
            corpus_file,
            fhelper,
            feature_tbl,
            truth,
            detector,
            linkage_counts,
            lcdict,
            ranking_probs,
            word_ambig,
            cut=cut,
            words=words,
            count_path=args.word_count,
            perfect=args.perfect,
            greedy=args.greedy,
            rank=args.rank,
            predict_sstats=args.predict_sense,
            predict_wstats=args.predict_wstats,
        )

if __name__ == '__main__':
    main()
