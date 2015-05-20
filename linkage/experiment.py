"""Main program for linkage experiments"""
import argparse

import corpus
import evaluate
import features
import linkage

from collections import defaultdict

from sklearn.linear_model import LogisticRegression


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', required=True,
                        help='cross validation folds distribution file')
    parser.add_argument('--tag', required=True,
                        help='connective token file')
    parser.add_argument('--word_ambig', required=True,
                        help='word ambiguity file')
    parser.add_argument('--word_probs', required=True,
                        help='word probability file')
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
    parser.add_argument('--perfect', action='store_true',
                        help='whether to do perfect experiment')
    parser.add_argument('--check_accuracy', action='store_true',
                        help='use svm to check classification accuracy')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--arg_output')

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
            len(words),
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
    visited |= all_indices

    return False


def train_sense_lr(lr, fhelper, data_set, feature_tbl, truth):
    X = []
    Y = []
    for (label, indices), x, y in zip(*fhelper.features(
            data_set, feature_tbl)):
        if indices in truth[label]:
            X.append(x)
            Y.append(truth[label][indices])
    lr.fit(X, Y)

NON_DIS = 4


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
    ps = set(plabels)

    for label, all_features in feature_tbl.items():
        if label in plabels:
            for l, y, x in all_features:
                if (label, l) not in s:
                    s.add((label, l))
                    if y == 0:
                        # non-discourse
                        sY.append(non_dis)
                    else:
                        sY.append(truth[label][l])
                    sYp.append(non_dis)

    for label, types in truth.items():
        if label in plabels:
            for l, t in types.items():
                if (label, l) not in s:
                    s.add((label, l))
                    sY.append(t)
                    sYp.append(non_dis)


def get_feature_set(feature_tbl):
    d = {}
    for label, all_features in feature_tbl.items():
        for l, _, x in all_features:
            d[(label, l)] = x
    return d


def cross_validation(corpus_file, fhelper, feature_tbl, truth, detector,
                     linkage_counts, lcdict, linkage_probs, word_ambig,
                     cut, *, words=None, perfect=False, arg_output=None):
    stats = evaluate.FoldStats(show_fold=False)
    # pstats = evaluate.FoldStats(show_fold=True)
    wstats = evaluate.FoldStats(show_fold=False)
    rejected_ov = defaultdict(int)
    rejected_s = defaultdict(int)
    # pType = {}

    lr = LogisticRegression()

    # compute sense statistics
    all_slabels = []
    all_sYp = []
    all_sY = []
    # structure type statistics
    all_ssYp = []
    all_ssY = []
    feature_set = get_feature_set(feature_tbl)

    if arg_output is not None:
        arg_output = open(arg_output, 'w')

    for i in fhelper.folds():
        print('\npredict for fold', i)

        # compute linkage statistics
        labels = []
        Yp = []
        Y = []

        # compute paragraph statistics
        plabels = list(fhelper.test_set(i))
        # pYp = []
        # pY = [1] * len(plabels)

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
            else:
                truth_connectives = None

            markers = []
            ambig_count = defaultdict(int)
            cand_words = set()
            for cnnct, indices in detector.all_tokens(tokens,
                                                      continuous=True,
                                                      cross=False,
                                                      truth=truth_connectives):
                markers.append(indices)
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

            # if any([c > 1 for c in ambig_count.values()]):
            #     pType[label] = (features.num_of_sentences(tokens), 'ambig')
            # else:
            #     pType[label] = (features.num_of_sentences(tokens), 'unique')

            markers.sort(key=lambda x: linkage_probs[(label, x)], reverse=True)

            visited = set()
            crossed = set()
            correct = 0
            for indices in markers:
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

            # if correct == len(truth[label]):
            #     pYp.append(1)
            # else:
            #     pYp.append(0)

        print('\nLinkage stats:')
        stats.compute_fold(labels, Yp, Y,
                           truth_count=count_fold(lcdict, plabels))

        # print('\nParagraph stats:')
        # pstats.compute_fold(plabels, pYp, pY)
        print('\nWord stats:')
        for w in wlabels:
            if w in has_words:
                wYp.append(1)
            else:
                wYp.append(0)
        wstats.compute_fold(wlabels, wYp, wY,
                            truth_count=word_ambig.count_fold(plabels))

        print('compute sense statistics...', end='', flush=True)
        train_sense_lr(lr, fhelper, fhelper.train_set(i), feature_tbl,
                       truth.linkage_type)
        slabels, sYp, sY = predict_sense(labels, Yp, Y, lr, feature_set,
                                         truth.linkage_type)
        append_sense_items(slabels, sYp, sY, feature_tbl,
                           truth.linkage_type, plabels)
        print('done!')

        all_slabels.append(slabels)
        all_sYp.append(sYp)
        all_sY.append(sY)

        print('compute structure statistics...', end='', flush=True)
        train_sense_lr(lr, fhelper, fhelper.train_set(i), feature_tbl,
                       truth.structure_type)
        sslabels, ssYp, ssY = predict_sense(labels, Yp, Y, lr, feature_set,
                                            truth.structure_type,
                                            non_dis=2)
        assert(slabels == sslabels)
        append_sense_items(sslabels, ssYp, ssY, feature_tbl,
                           truth.structure_type, plabels,
                           non_dis=2)

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

        print('done!')

        all_ssYp.append(ssYp)
        all_ssY.append(ssY)

    print('== done ==')

    print('\nLinkage stats:')
    stats.print_total(truth_count=linkage_counts)
    stats.print_distribution(
        word_ambig,
        function=lambda x: {(l, w) for (l, ws) in x for w in ws})
    stats.count_by(label='length')
    print('rejected overlapped:', rejected_ov, 'rejected scores:', rejected_s)

    # print('\nParagraph stats:')
    # pstats.print_total()
    # pstats.count_by(function=pType.get, label='Sentence Length')
    # pstats.count_by(function=lambda x: pType[x][1], label='Ambiguity')

    print('\nWord stats:')
    wstats.print_total(truth_count=len(words))

    print('Sense stats:')
    evaluate.print_sense_scores(all_sY, all_sYp, 'Overall')

    print('Structure stats:')
    evaluate.print_sense_scores(all_ssY, all_ssYp, 'Overall')

    if arg_output is not None:
        arg_output.close()


def main():
    args = process_commands()

    corpus_file = corpus.CorpusFile(
        args.corpus, args.corpus_pos, args.corpus_parse)
    fhelper = corpus.FoldsHelper(args.folds)
    truth = linkage.LinkageFile(args.linkage)
    words = truth.all_words()
    detector = linkage.LinkageDetector(args.tag)
    feature_tbl = features.load_features_table(
        args.linkage_features, lambda x: tuple(x.split('-')))

    linkage_counts, lcdict = count_linkage(args.linkage)
    linkage_probs = load_linkage_probs(args.linkage_probs)

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
        cut = lambda x, _: linkage_probs[x] < args.threshold

    print('ranking model')
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
        arg_output=args.arg_output)

    '''
    baseline_probs = compute_ranking_probs(linkage_probs, key=lambda x: 1)
    word_probs, word_truth = load_word_probs(args.word_probs)

    print('baseline model')
    cross_validation(
        corpus_file,
        fhelper,
        truth,
        detector,
        linkage_counts,
        baseline_probs,
        word_ambig,
        cut=lambda x, y: any(word_probs[(x[0], w)] < 0.5 for w in x[1]))
    '''

if __name__ == '__main__':
    main()
