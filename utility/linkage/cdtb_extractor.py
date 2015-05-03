"""Extract all discourse connectives positions from CDTB.
Outputs a text file with all CDTB passages and a file with connetives.
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as etree

from collections import defaultdict
from glob import glob

Explicit = '\u663e\u5f0f\u5173\u7cfb'
Rsym = re.compile(r'\W')


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--connective', required=True)
    parser.add_argument('--arg', required=True,
                        help='output for arguments')

    return parser.parse_args()


def split_dot(text):
    return text.split('\u2026')


def get_offsets(text):
    '''get [) offsets'''
    x, y = split_dot(text)
    return int(x) - 1, int(y)


def print_tree(tree):
    print(etree.tostring(tree, encoding='unicode'))


def extract(article, l, r):
    text = article[l:r]
    if text == '':
        return '[None]'
    else:
        return text


def print_distribution(d):
    for k, v in sorted(d.items()):
        print('{}:\t{}'.format(k, v))
    print('total:\t{}'.format(sum(d.values())))
    print()

if __name__ == '__main__':
    args = process_commands()
    print('\nstart extract')

    # stats for relations
    normal = abnormal = 0

    # stats for arguments
    argument_stats = defaultdict(int)
    connective_stats = defaultdict(int)
    mapping_stats = defaultdict(int)
    before_stats = defaultdict(int)
    end_stats = defaultdict(int)
    in_range = out_range = 0

    with open(args.output, 'w') as f, open(args.connective, 'w') as connective_output:
        for path in glob(os.path.join(args.input, '*')):
            fname = os.path.basename(path).rsplit('.', 1)[0]

            with open(path, 'r', encoding='utf-8') as pf:
                paragraph_list = etree.parse(pf).getroot().findall('P')

            for p in paragraph_list:
                relation_list = p.findall('R')
                detected_num_of_paragraph = 0
                article = ''

                # filter duplicate annotations
                annotated = set()

                for r in relation_list:
                    if r.get('ParentId') == '-1':
                        sen_num = len(r.get('SentencePosition').split('|'))
                        sents = r.get('Sentence').split('|')

                        if len(sents) != sen_num:
                            print('-- strange --')
                            print_tree(r)
                        article = ''.join(sents)
                        f.write('cdtb-{}-{}\t{}\n'.format(
                            fname, p.get('ID'), article))
                        detected_num_of_paragraph += 1

                    is_explicit = r.get('ConnectiveType') == Explicit

                    # -- extract connective -- #

                    if is_explicit:
                        indices = tuple(
                            r.get('ConnectivePosition').split('&&'))

                        # check duplication
                        if indices in annotated:
                            print('# duplicate detected')
                            continue
                        else:
                            annotated.add(indices)

                        indices = list(map(get_offsets, indices))
                        conncts = split_dot(r.get('Connective'))

                        # check index correctness
                        for index, connct in zip(indices, conncts):
                            x, y = index
                            if article[x:y] != connct:
                                print('# wrong position')
                                print(connct, article[x:y])

                        rtype = r.get('RelationType')

                        connective_output.write('cdtb-{}-{}\t{}\t{}\t{}\n'.format(
                            fname,
                            p.get('ID'),
                            '-'.join(conncts),
                            '-'.join('{}:{}'.format(x, y) for x, y in indices),
                            rtype
                        ))

                        # generate role stats
                        roleloc = r.get('RoleLocation')
                        if roleloc == 'normal':
                            normal += 1
                        elif roleloc == 'abnormal':
                            abnormal += 1

                    # -- extract arguments -- #

                    sent_indices = [get_offsets(i)
                                    for i in r.get('SentencePosition').split('|')]
                    sents = r.get('Sentence').split('|')

                    # check sentence correctness
                    incorrect = False
                    if len(sent_indices) != len(sents):
                        print('# sentence num not matched')
                        print('{} P{} R{}'.format(
                            fname,
                            p.get('ID'),
                            r.get('ID')
                        ))
                        incorrect = True

                    for index, sent in zip(sent_indices, sents):
                        x, y = index
                        if article[x:y] != sent:
                            print('# wrong sentence position')
                            print('{} P{} R{}'.format(
                                fname,
                                p.get('ID'),
                                r.get('ID')
                            ))
                            print('correct:', sent,
                                  'extracted:', article[x:y])
                            incorrect = True

                    # generate sent stats
                    if is_explicit:
                        argument_stats[len(sents)] += 1
                        connective_stats[len(conncts)] += 1
                        mapping_stats['{}-{}'.format(
                            len(conncts),
                            len(sents))] += 1
                        if sent_indices[0][0] <= indices[0][0] and indices[-1][-1] <= sent_indices[-1][-1]:
                            in_range += 1
                        else:
                            out_range += 1

                    if not incorrect:
                        for x, y in sent_indices:
                            before = extract(article, x - 1, x)
                            start = extract(article, x, x + 1)
                            end = extract(article, y - 1, y)
                            after = extract(article, y, y + 1)
                            before_stats[before] += 1
                            end_stats[end] += 1

                            if Rsym.match(before) is None or (after != '[None]' and Rsym.match(end) is None):
                                print('##not in', before, start)
                                print('##not in', end, after)
                                print(extract(article, x, y))
                                print('{} P{} R{}'.format(
                                    fname,
                                    p.get('ID'),
                                    r.get('ID')
                                ))

                if detected_num_of_paragraph > 1:
                    print('strange {} {} {}'.format(
                        detected_num_of_paragraph, fname, p.get('ID')))
                    print_tree(r)

    print('normal: {}, abnormal: {}'.format(normal, abnormal))
    print('\nargument stats:')
    print_distribution(argument_stats)
    print('\nconnective stats:')
    print_distribution(connective_stats)
    print('\nmapping stats:')
    print_distribution(mapping_stats)
    # print('\nbefore stats:')
    # print_distribution(before_stats)
    # print('\nend stats:')
    # print_distribution(end_stats)

    print('in range: {}, out range: {}'.format(in_range, out_range))
