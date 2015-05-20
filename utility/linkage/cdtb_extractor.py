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

from statistics import print_distribution


class SkipException(Exception):
    pass

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


def get_text_span(r):
    indices = get_sent_indices(r)
    return indices[0][0], indices[-1][-1]


def get_sent_indices(r):
    return [get_offsets(i) for i in r.get('SentencePosition').split('|')]


def get_sents(r):
    return r.get('Sentence').split('|')


def check_indices(article, texts, indices):
    '''check indices correctness'''

    correct = True
    if len(indices) != len(texts):
        print('# not matched')
        correct = False

    for index, text in zip(indices, texts):
        x, y = index
        if article[x:y] != text:
            print('correct:', text,
                  'extracted:', article[x:y])
            correct = False

    return correct


def handle_cnncts(r, annotated):
    indices = tuple(
        r.get('ConnectivePosition').split('&&'))

    # check duplication
    if indices in annotated:
        print('# duplicate detected')
        raise SkipException()
    else:
        annotated.add(indices)

    cnncts = split_dot(r.get('Connective'))
    indices = list(map(get_offsets, indices))
    rtype = r.get('RelationType')
    stype = r.get('StructureType')
    return cnncts, indices, rtype, stype


def get_hierarchy_spans(rlist, rmap, r):
    spans = []
    while True:
        spans.append(get_text_span(r))
        p = r.get('ParentId')
        if p == '-1':
            break
        else:
            r = rlist[rmap[p]]
    return spans


def extract_linkages(dir_path, pout, cout, aout):
    print('\nstart extract')

    # stats for relations
    role_stats = defaultdict(int)

    # stats for arguments
    argument_stats = defaultdict(int)
    connective_stats = defaultdict(int)
    mapping_stats = defaultdict(int)
    mapping_examples = defaultdict(int)
    before_stats = defaultdict(int)
    end_stats = defaultdict(int)
    range_stats = defaultdict(int)

    for path in glob(os.path.join(dir_path, '*')):
        fname = os.path.basename(path).rsplit('.', 1)[0]

        with open(path, 'r', encoding='utf-8') as pf:
            paragraph_list = etree.parse(pf).getroot().findall('P')

        for p in paragraph_list:
            relation_list = p.findall('R')
            detected_num_of_paragraph = 0
            article = ''

            # filter duplicate annotations
            annotated = set()

            rel_map = {}
            for i, r in enumerate(relation_list):
                rel_map[r.get('ID')] = i

            for r in relation_list:
                try:
                    # -- checks -- #
                    def print_current():
                        print('## -- current -- ##')
                        print('{} P{} R{}'.format(
                            fname,
                            p.get('ID'),
                            r.get('ID')
                        ))
                        print_tree(r)

                    sent_indices = get_sent_indices(r)
                    sents = get_sents(r)

                    if r.get('ParentId') == '-1':
                        article = ''.join(sents)
                        # ensure no strange whitespace is present
                        assert(article == article.strip())
                        pout.write('cdtb-{}-{}\t{}\n'.format(
                            fname, p.get('ID'), article))
                        detected_num_of_paragraph += 1

                    spans = get_hierarchy_spans(relation_list, rel_map, r)
                    assert(spans[-1][0] == 0 and spans[-1][-1] == len(article))

                    correct_sent = check_indices(article, sents, sent_indices)
                    if not correct_sent:
                        print('sentence not correct')
                        print_current()

                    is_explicit = r.get('ConnectiveType') == Explicit

                    if is_explicit:
                        # -- extract connective -- #
                        cnncts, indices, rtype, stype = handle_cnncts(
                            r, annotated)
                        check_indices(article, cnncts, indices)

                        cnnct_offsets = '-'.join('{}:{}'.format(x, y)
                                                 for x, y in indices)
                        cout.write('cdtb-{}-{}\t{}\t{}\t{}\t{}\n'.format(
                            fname,
                            p.get('ID'),
                            '-'.join(cnncts),
                            cnnct_offsets,
                            rtype,
                            stype
                        ))

                        # -- extract arguments -- #
                        aout.write('cdtb-{}-{}\t{}\t{}\t{}\t{}\n'.format(
                            fname,
                            p.get('ID'),
                            cnnct_offsets,
                            '|'.join(sents),
                            '|'.join('{}:{}'.format(x, y)
                                     for x, y in sent_indices),
                            '|'.join('{}:{}'.format(x, y)
                                     for x, y in spans)
                        ))

                        # -- generate stats -- #
                        role_stats[r.get('RoleLocation')] += 1
                        argument_stats[len(sents)] += 1
                        connective_stats[len(cnncts)] += 1
                        if (len(cnncts) > 1 and len(cnncts) < len(sents)
                                or len(cnncts) == 1 and len(sents) > 2):
                            mapping_examples[r.get('StructureType')] += 1

                        mapping_stats['{}-{}'.format(
                            len(cnncts),
                            len(sents))] += 1
                        if (sent_indices[0][0] <= indices[0][0]
                                and indices[-1][-1] <= sent_indices[-1][-1]):
                            range_stats['in_range'] += 1
                        else:
                            range_stats['out_range'] += 1

                    if correct_sent:
                        for x, y in sent_indices:
                            before = extract(article, x - 1, x)
                            start = extract(article, x, x + 1)
                            end = extract(article, y - 1, y)
                            after = extract(article, y, y + 1)
                            before_stats[before] += 1
                            end_stats[end] += 1

                            if (Rsym.match(before) is None
                                    or after != '[None]' and Rsym.match(end) is None):
                                print('##not in', before, start)
                                print('##not in', end, after)
                                print(extract(article, x, y))
                                print_current()
                except SkipException:
                    continue

            if detected_num_of_paragraph > 1:
                print('strange {} {} {}'.format(
                    detected_num_of_paragraph, fname, p.get('ID')))
                print_tree(r)

    print()
    print('role stats:')
    print_distribution(role_stats)
    print('argument stats:')
    print_distribution(argument_stats)
    print('connective stats:')
    print_distribution(connective_stats)
    print('mapping stats:')
    print_distribution(mapping_stats)
    print('relation types that exceed connective length:')
    print_distribution(mapping_examples)
    # print('\nbefore stats:')
    # print_distribution(before_stats)
    # print('\nend stats:')
    # print_distribution(end_stats)

    print_distribution(range_stats)


if __name__ == '__main__':
    args = process_commands()
    with open(args.output, 'w') as pout:
        with open(args.connective, 'w') as cout:
            with open(args.arg, 'w') as aout:
                extract_linkages(args.input, pout, cout, aout)
