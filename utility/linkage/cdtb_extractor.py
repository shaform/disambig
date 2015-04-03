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

    return parser.parse_args()


if __name__ == '__main__':
    args = process_commands()
    with open(args.output, 'w') as f, open(args.tag, 'w') as tf:
        for path in glob(os.path.join(args.input, '*')):
            fname = os.path.basename(path).rsplit('.', 1)[0]

            with open(path, 'r', encoding='utf-8') as pf:
                p_list = etree.parse(pf).getroot().findall('P')
            for p in p_list:
                r_list = p.findall('R')
                detected = 0
                article = ''
                for r in r_list:
                    if r.get('ParentId') == '-1':
                        sen_num = len(r.get('SentencePosition').split('|'))
                        sents = r.get('Sentence').split('|')

                        if len(sents) != sen_num:
                            print('-- strange --')
                            print(etree.tostring(r, encoding='unicode'))
                        article = ''.join(sents)
                        f.write('cdtb-{}-{}\t{}\n'.format(
                            fname, p.get('ID'), article))
                        detected += 1
                    if r.get('ConnectiveType') == '显式关系':
                        conncts = r.get('Connective').split('\u2026')
                        pos_s = r.get('ConnectivePosition').split('&&')
                        tp = r.get('RelationType')

                        def g(p):
                            x, y = p.split('\u2026')
                            return int(x) - 1, int(y)

                        pos_s = list(map(g, pos_s))
                        for pos, connct in zip(pos_s, conncts):
                            x, y = pos
                            if article[x:y] != connct:
                                print(connct, article[x:y])
                        tf.write('cdtb-{}-{}\t{}\t{}\t{}\n'.format(fname, p.get('ID'), '-'.join(
                            conncts), '-'.join('{}:{}'.format(x, y) for x, y in pos_s),
                            tp))

                if detected > 1:
                    print('strange {} {} {}'.format(
                        detected, fname, p.get('ID')))
                    print(etree.tostring(r, encoding='unicode'))
