"""Extract all discourse connectives positions from CDTB.
Outputs a text file with all CDTB passages and a file with connetives.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as etree

from glob import glob


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--arg', required=True,
                        help='output for arguments')

    return parser.parse_args()


if __name__ == '__main__':
    args = process_commands()
    normal = abnormal = 0
    with open(args.output, 'w') as f, open(args.tag, 'w') as tf:
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
                            print(etree.tostring(r, encoding='unicode'))
                        article = ''.join(sents)
                        f.write('cdtb-{}-{}\t{}\n'.format(
                            fname, p.get('ID'), article))
                        detected_num_of_paragraph += 1

                    # explicit relations
                    if r.get('ConnectiveType') == '\u663e\u5f0f\u5173\u7cfb':
                        # split by '...'
                        conncts = r.get('Connective').split('\u2026')
                        indices = tuple(
                            r.get('ConnectivePosition').split('&&'))

                        if indices in annotated:
                            print('# duplicate detected')
                            continue
                        else:
                            annotated.add(indices)

                        rtype = r.get('RelationType')

                        roleloc = r.get('RoleLocation')
                        if roleloc == 'normal':
                            normal += 1
                        elif roleloc == 'abnormal':
                            abnormal += 1

                        def g(p):
                            '''get [) offsets for connective'''
                            # split by '...'
                            x, y = p.split('\u2026')
                            return int(x) - 1, int(y)

                        indices = list(map(g, indices))

                        # check correctness
                        for index, connct in zip(indices, conncts):
                            x, y = index
                            if article[x:y] != connct:
                                print('# wrong position')
                                print(connct, article[x:y])

                        tf.write('cdtb-{}-{}\t{}\t{}\t{}\n'.format(
                            fname,
                            p.get('ID'),
                            '-'.join(conncts),
                            '-'.join('{}:{}'.format(x, y) for x, y in indices),
                            rtype
                        ))

                if detected_num_of_paragraph > 1:
                    print('strange {} {} {}'.format(
                        detected_num_of_paragraph, fname, p.get('ID')))
                    print(etree.tostring(r, encoding='unicode'))

    print('normal: {}, abnormal: {}'.format(normal, abnormal))
