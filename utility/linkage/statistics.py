"""Generate statistics for papers"""
import argparse

import corpus
import evaluate
import linkage

from collections import defaultdict


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--connective', required=True,
                        help='connective file')
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')

    return parser.parse_args()


def print_total_recall_cand(total, recall, cand, label=''):
    print('total/recall/cand {}: {}/{}/{}\n'.format(
        label,
        total,
        recall,
        cand
    ))


def count_by_num(counts, bound=5):
    d = defaultdict(int)
    for v in counts.values():
        if v >= bound:
            d[bound] += 1
        else:
            d[v] += 1
    return d


def print_distribution(d):
    for k, v in sorted(d.items()):
        print('{}:\t{}'.format(k, v))
    print()


def stat_all_detect(detector, corpus_file, truth):

    counter = evaluate.ProgressCounter()
    total_connective_count = 0
    recall_conective_count = 0
    cand_connective_count = 0
    recall_component_count = 0
    cand_component_count = 0
    t_recall_component_count = 0
    t_cand_component_count = 0

    length_count = defaultdict(int)
    disambig_count = defaultdict(int)
    cand_disambig_count = defaultdict(int)

    words = truth.all_words()

    # count stats

    for l, tokens in corpus_file.corpus.items():
        counter.step()

        components = set()
        connectives = set()
        for _, poss in detector.detect_all(tokens):
            if poss in truth[l]:
                recall_conective_count += 1
            connectives.add(poss)
            for pos in poss:
                components.add(pos)
                cand_disambig_count[(l, pos)] += 1

            if all((l, pos) in words for pos in poss):
                for pos in poss:
                    disambig_count[(l, pos)] += 1

            cand_connective_count += 1

        cand_component_count += len(components)
        for x in components:
            if (l, x) in words:
                recall_component_count += 1

        total_connective_count += len(truth[l])
        for c in truth[l]:
            length_count[len(c)] += 1
            if c not in connectives:
                for x in c:
                    disambig_count[(l, x)] = 1

        components = set()
        # collect stats by purely components
        for _, pos in detector.detect_all_components(tokens):
            components.add(pos)

        t_cand_component_count += len(components)
        for x in components:
            if (l, x) in words:
                t_recall_component_count += 1

    # print stats

    print()

    print_total_recall_cand(
        total_connective_count,
        recall_conective_count,
        cand_connective_count,
        'connectives'
    )

    print_total_recall_cand(
        len(truth.all_words()),
        recall_component_count,
        cand_component_count,
        'components'
    )

    print_total_recall_cand(
        len(truth.all_words()),
        t_recall_component_count,
        t_cand_component_count,
        'token components'
    )

    print('length')
    print_distribution(length_count)

    # linking ambiguity among truth
    print('disambig truth')
    print_distribution(count_by_num(disambig_count))

    # linking ambiguity among all candidates
    print('disambig cand')
    print_distribution(count_by_num(cand_disambig_count))


def main():
    args = process_commands()

    detector = linkage.LinkageDetector(args.connective)
    corpus_file = corpus.CorpusFile(args.corpus)
    truth = linkage.LinkageFile(args.linkage)

    stat_all_detect(detector, corpus_file, truth)

if __name__ == '__main__':
    main()
