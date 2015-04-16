"""Generate statistics for papers"""
import argparse

import corpus
import evaluate
import linkage


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True,
                        help='connective file')
    parser.add_argument('--linkage', required=True,
                        help='linkage ground truth file')
    parser.add_argument('--corpus', required=True,
                        help='raw corpus file')

    return parser.parse_args()


def stat_all_detect(detector, corpus_file, truth):

    counter = evaluate.ProgressCounter()
    total_connective_count = 0
    recall_conective_count = 0
    cand_connective_count = 0
    recall_component_count = 0
    cand_component_count = 0

    words = truth.all_words()

    for l, tokens in corpus_file.corpus.items():
        counter.step()
        total_connective_count += len(truth[l])

        components = set()
        for _, poss in detector.detect_all(tokens):
            if poss in truth[l]:
                recall_conective_count += 1
            for pos in poss:
                components.add(pos)
            cand_connective_count += 1
        cand_component_count += len(components)
        for x in components:
            if (l, x) in words:
                recall_component_count += 1

    print('total/recall/cand connectives: {}/{}/{}'.format(
        total_connective_count,
        recall_conective_count,
        cand_connective_count
    ))

    print('total/recall/cand components: {}/{}/{}'.format(
        len(truth.all_words()),
        recall_component_count,
        cand_component_count
    ))


def main():
    args = process_commands()

    detector = linkage.LinkageDetector(args.tag)
    corpus_file = corpus.CorpusFile(args.corpus)
    truth = linkage.LinkageFile(args.linkage)

    stat_all_detect(detector, corpus_file, truth)

if __name__ == '__main__':
    main()
