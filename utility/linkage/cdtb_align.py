"""Align CDTB connective positions with segmented text CDTB."""
import argparse
import os
import sys


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='segmented text')
    parser.add_argument('--tag', required=True,
                        help='linkage text')
    parser.add_argument('--output', required=True)

    return parser.parse_args()

_TP = {
    # causality
    '因果关系': 0,
    '推断关系': 0,
    '假设关系': 0,
    '目的关系': 0,
    '条件关系': 0,
    '背景关系': 0,

    # coordination
    '并列关系': 1,
    '顺承关系': 1,
    '递进关系': 1,
    '选择关系': 1,
    '对比关系': 1,

    # transition
    '转折关系': 2,
    '让步关系': 2,

    # explanation
    '解说关系': 3,
    '总分关系': 3,
    '例证关系': 3,
    '评价关系': 3,
}


if __name__ == '__main__':
    args = process_commands()
    d = {}
    with open(args.input, 'r') as f:
        for l in f:
            tag, r = l.rstrip('\n').split('\t', 1)
            d[tag] = r.split(' ')

    total = 0
    not_fit = 0
    cross = 0
    with open(args.tag, 'r') as f, open(args.output, 'w') as of:
        for l in f:
            tag, conncts, pos_s, tp = l.rstrip('\n').split('\t')
            conncts = conncts.split('-')
            pos_s = [x.split(':') for x in pos_s.split('-')]
            pos_s = [(int(x), int(y)) for x, y in pos_s]

            detected_pos = []
            for connct, pos in zip(conncts, pos_s):
                total += 1

                start = 0
                text = ''

                extracted = []
                for i, x in enumerate(d[tag]):
                    end = start + len(x)
                    if start >= pos[0] and start < pos[1] or end > pos[0] and end <= pos[1]:
                        text += x
                        extracted.append((i, len(x)))
                    start = end

                if connct in text:
                    if connct != text:
                        not_fit += 1
                    elif len(extracted) > 1:
                        cross += 1

                    offset = text.find(connct)
                    offset_end = offset + len(connct)

                    offsets = []
                    for i, length in extracted:
                        if offset < length:
                            offsets.append('{}[{}:{}]'.format(i,
                                                              max(0, offset),
                                                              min(length,
                                                                  offset_end)
                                                              ))
                            if offset_end <= length:
                                break
                        offset -= length
                        offset_end -= length
                    detected_pos.append(','.join(offsets))

                else:
                    print('wrong')
                    print(d[tag])
                    print(tag, text, connct)
                    print(pos)
                    break
            else:
                # successful
                of.write('{}\t{}\t{}\t{}\n'.format(tag, '-'.join(conncts),
                                                   '-'.join(detected_pos),
                         _TP[tp]))

    print('total', total)
    print('not fit', not_fit)
    print('cross', cross)
