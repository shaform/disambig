"""Extract ClueWeb for training"""
import argparse
import os

import subprocess
from subprocess import Popen, PIPE


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', required=True)
    parser.add_argument('--tmp', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    return parser.parse_args()


def main():
    args = process_commands()
    with open(args.output, 'w') as out:
        for root, dirs, files in os.walk(args.input):
            files = list(files)
            for n, fname in enumerate(files):
                print('process file {}: {}/{}'.format(root, n, len(files)))
                if fname.endswith('pos.ser.gz'):
                    source = os.path.join(root, fname)
                    tmp = args.tmp
                    cmdline = 'java edu.ntu.nlp.discourseRelation.utils.DR_ConvertSerDR2TextFile {} {}'
                    cmdline = cmdline.format(source, tmp)
                    Popen(cmdline, shell=True, cwd=args.tool).wait()

                    print('==== unzipped :{}/{} ===='.format(root, fname))
                    print('==== start converting ====')

                    cmdline = ['grep', r'^[0-9]\+:', tmp]
                    text = subprocess.check_output(cmdline,
                                                   universal_newlines=True)

                    for l in text.split('\n'):
                        i = l.find(':')
                        if i == -1:
                            continue
                        l = l[i + 1:].strip()

                        if l.count(' ') + 1 != l.count('/'):
                            continue
                        out.write(l + '\n')


if __name__ == '__main__':
    main()
