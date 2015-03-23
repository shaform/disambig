"""Count the number of all possible linkages."""
import argparse
import os
import re
import sys
import linkage

from collections import defaultdict


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('tag_path')
    parser.add_argument('answer_path')

    return parser.parse_args()


def all_indices(s, target, offset=0):
    listindex = []
    i = s.find(target, offset)
    while i >= 0:
        listindex.append(i + len(target))
        i = s.find(target, i + 1)
    return listindex


def count_a_tag(idx, tag, offset, text, truth, label):
    if idx >= len(tag):
        return 1
    else:
        total = 0
        for i in all_indices(text, tag[idx], offset=offset):
            total += count_a_tag(idx + 1, tag, i, text, truth, label)
        if idx == 0:
            return 0, total
        else:
            return total


def count(tags, text, truth, label, count_by=count_a_tag):
    total = 0
    total_correct = 0
    for tag in tags:
        tc, t = count_by(0, tag, 0, text, truth, label)
        total_correct += tc
        total += t
    if count_by is count_a_tag:
        total_correct = len(truth[label])
    return total_correct, total


if __name__ == '__main__':
    args = process_commands()

    with open(args.tag_path, 'r') as f:
        tags = [l.rstrip().split('\t') for l in f]

    detector = linkage.LinkageDetector(args.tag_path)

    truth = defaultdict(set)
    with open(args.answer_path, 'r') as f:
        for l in f:
            label, _, pos = l.rstrip().split('\t')
            truth[label].add(pos)

    total = 0
    total_correct = 0
    total_by_token = 0
    total_by_token_correct = 0
    total_by_cont_token = 0
    total_by_cont_token_correct = 0
    total_by_label = defaultdict(int)
    total_by_label_correct = defaultdict(int)
    with open(args.input_path, 'r') as f:
        for i, l in enumerate(f):
            label, tokens = l.rstrip().split('\t', 1)
            tokens = tokens.split(' ')
            text = ''.join(tokens)
            short_label = label.split('-')[1]

            tc, t = count(tags, text, truth, label)
            total += t
            total_correct += tc

            for _, x in detector.detect_by_tokens(tokens,
                                                  continuous=False,
                                                  cross=False):
                x = '-'.join(x)
                if x in truth[label]:
                    total_by_token_correct += 1
                total_by_token += 1

            test_set = set()

            def key_comp(x):
                ps = [[int(z) for z in y] for y in x[1]]
                total = 0
                if len(ps) > 1:
                    for i in range(1, len(ps)):
                        total += ps[i][0] - ps[i - 1][-1]
                    total /= (len(ps) - 1)
                # calculate symbol dist
                total_dist = 0
                for p in ps:
                    lhs = -1
                    rhs = len(tokens)

                    for i in range(p[0] - 1, -1, -1):
                        if re.search(r'\W', tokens[i]) is not None:
                            lhs = i
                            break
                    for i in range(p[-1] + 1, len(tokens)):
                        if re.search(r'\W', tokens[i]) is not None:
                            rhs = i
                            break

                    total_dist = min(p[0] - lhs, rhs - p[-1])
                total_dist /= len(ps)
                test_set.add((-len(x[1]), total, total_dist))
                return -len(x[1]), total, total_dist

            markers = []
            for _, x in detector.detect_by_tokens(tokens,
                                                  continuous=True, cross=True):
                x = '-'.join(x)
                markers.append((x,
                                [[u.split('[')[0] for u in t.split(',')]
                                    for t in x.split('-')]))
            markers.sort(key=key_comp)

            visited = set()
            for x in markers:
                for t in x[1]:
                    for p in t:
                        if p in visited:
                            break
                        else:
                            visited.add(p)
                    else:
                        continue
                    break
                else:
                    if x[0] in truth[label]:
                        total_by_label_correct[short_label] += 1
                        total_by_cont_token_correct += 1
                    total_by_cont_token += 1
                    total_by_label[short_label] += 1

            print('{} handled, {}'.format(i + 1, label))

print('total: {}/{}, total by token: {}/{}, cont: {}/{}'.format(
    total_correct, total,
    total_by_token_correct, total_by_token,
    total_by_cont_token_correct, total_by_cont_token))

for l in sorted(total_by_label):
    print('{}\t{}\t{}\t{}'.format(l,
                                  total_by_label[l],
          total_by_label_correct[l],
          total_by_label_correct[l] / total_by_label[l]))
