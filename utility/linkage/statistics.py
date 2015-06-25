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
    parser.add_argument('--output_count', required=True,
                        help='output word count file')

    return parser.parse_args()


def print_total_correct_cand(total, correct, cand, label=''):
    print('total real/correct/cand {}: {}/{}/{}'.format(
        label,
        total,
        correct,
        cand
    ))
    recall = correct / total
    prec = correct / cand

    print('recall: {:.4f}, prec: {:.4f}'.format(recall, prec))
    print()


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


class Stats(object):

    def __init__(self, truth):
        self.truth = truth
        self.words = truth.all_words()
        self.visited = set()

        self.total_component_count = len(self.words)
        self.total_connective_count = 0
        for s in truth.linkage.values():
            self.total_connective_count += len(s)

        self.cand_component_count_by_label = defaultdict(int)
        self.cand_component_count = 0
        self.correct_component_count = 0
        self.cand_connective_count = 0
        self.correct_connective_count = 0

        self.disambig_count = defaultdict(int)
        self.cand_disambig_count = defaultdict(int)

    def collect_connective(self, l, indices_lst):
        self.cand_connective_count += 1
        if indices_lst in self.truth[l]:
            self.correct_connective_count += 1

        for indices in indices_lst:
            comp = (l, indices)
            self.cand_disambig_count[comp] += 1
            if comp not in self.visited:
                self.visited.add(comp)
                self.cand_component_count += 1
                self.cand_component_count_by_label[l] += 1
                if comp in self.words:
                    self.correct_component_count += 1

        if all((l, indices) in self.words for indices in indices_lst):
            for indices in indices_lst:
                self.disambig_count[(l, indices)] += 1

    def collect_component(self, l, indices):
        self.cand_component_count += 1
        if (l, indices) in self.words:
            self.correct_component_count += 1

    def print_connective(self):
        print_total_correct_cand(
            self.total_connective_count,
            self.correct_connective_count,
            self.cand_connective_count,
            'connectives'
        )

    def print_component(self):
        print_total_correct_cand(
            self.total_component_count,
            self.correct_component_count,
            self.cand_component_count,
            'components'
        )

    def print_ambiguity(self):
        print('disambig truth')
        print_distribution(count_by_num(self.disambig_count))

        # linking ambiguity among all candidates
        print('disambig cand')
        print_distribution(count_by_num(self.cand_disambig_count))


def stat_all_detect(detector, corpus_file, truth, count_path):

    counter = evaluate.ProgressCounter()
    mstats = Stats(truth)
    cstats = Stats(truth)
    tstats = Stats(truth)

    length_count = defaultdict(int)
    for s in truth.linkage.values():
        for x in s:
            length_count[len(x)] += 1

    for l, tokens in corpus_file.corpus.items():
        counter.step()

        # count string matching stats
        for _, indices_lst in detector.detect_all(tokens):
            mstats.collect_connective(l, indices_lst)

        # collect stats by purely components
        for _, indices in detector.detect_all_components(tokens):
            cstats.collect_component(l, indices)

        # collect stats by segmentation
        for _, indices_lst in detector.detect_by_tokens(tokens,
                                                        continuous=True,
                                                        cross=False):
            tstats.collect_connective(l, indices_lst)

    # print stats

    print()

    print('## corpus ##')
    print('length')
    print_distribution(length_count)

    print('## string matching by components ##')
    cstats.print_component()

    print('## string matching by connectives ##')
    mstats.print_connective()
    mstats.print_component()
    mstats.print_ambiguity()

    with open(count_path, 'w') as f:
        print('\noutput file to {}'.format(count_path))
        total = 0
        for l, v in mstats.cand_component_count_by_label.items():
            total += v
            f.write('{}\t{}\n'.format(l, v))
        print('totally {} word counts written'.format(total))

    print('## segmentation##')
    tstats.print_connective()
    tstats.print_component()
    tstats.print_ambiguity()


def main():
    args = process_commands()

    detector = linkage.LinkageDetector(args.connective)
    corpus_file = corpus.CorpusFile(args.corpus)
    truth = linkage.LinkageFile(args.linkage)

    stat_all_detect(detector, corpus_file, truth, args.output_count)

if __name__ == '__main__':
    main()
