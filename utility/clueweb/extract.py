"""Extract ClueWeb for training"""
import argparse
import multiprocessing as mp
import os
import subprocess

from subprocess import Popen, PIPE


def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', required=True)
    parser.add_argument('--tmp', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lang',
                        choices=('chinese', 'english'), default='chinese')
    parser.add_argument('--threads', type=int, default=6)

    return parser.parse_args()


def unzip_chinese(src, dst, tool):
    cmdline = 'java edu.ntu.nlp.discourseRelation.utils.DR_ConvertSerDR2TextFile {} {}'
    cmdline = cmdline.format(src, dst)
    Popen(cmdline, shell=True, cwd=tool).wait()


def unzip_english(src, dst, tool):
    cmdline = 'java edu.ntu.nlp.discourseRelation.utils.ENG_ConvertSerDR2TextFile {} {}'
    cmdline = cmdline.format(src, dst)
    Popen(cmdline, shell=True, cwd=tool).wait()


def convert_chinese(src, dst):
    cmdline = ['grep', r'^[0-9]\+:', src]
    try:
        text = subprocess.check_output(cmdline,
                                       universal_newlines=True)

        with open(dst, 'w') as out:
            for l in text.split('\n'):
                i = l.find(':')
                if i == -1:
                    continue
                l = l[i + 1:].strip()

                if l.count(' ') + 1 != l.count('/'):
                    continue
                out.write(l + '\n')

    except subprocess.CalledProcessError:
        return


def convert_english(src, dst, tool):
    cmdline = r'''tr -d '\r' < {} | grep -v "^$" > {}'''
    cmdline = cmdline.format(src, dst)
    Popen(cmdline, shell=True, cwd=tool).wait()


def process_directory(src, files, dst, tmp, tool, lang):
    print('==== process directory {} ===='.format(src))

    count = 0
    basename = os.path.basename(src)
    total = len(files)

    for n, fname in enumerate(files):
        if not fname.endswith('pos.ser.gz'):
            continue

        txtfile = os.path.join(dst, fname + '.txt')
        if os.path.exists(txtfile):
            continue

        print('==== process file {}: {}/{} ===='.format(
            basename, n, total))

        gzfile = os.path.join(src, fname)
        if lang == 'chinese':
            unzip_chinese(gzfile, tmp, tool)
        elif lang == 'english':
            unzip_english(gzfile, tmp, tool)
        print('==== unzipped :{}/{} ===='.format(src, fname))

        print('==== start converting ====')
        if lang == 'chinese':
            convert_chinese(tmp, txtfile)
        elif lang == 'english':
            convert_english(tmp, txtfile, tool)


def main():
    args = process_commands()

    processes = []
    for root, dirs, files in os.walk(args.input):
        files = list(files)
        if len(files) == 0:
            continue

        dst = os.path.join(args.output, os.path.relpath(root, args.input))
        mkdir_p(dst)
        tmp = '{}.{}.{}'.format(args.tmp, args.lang, len(processes))

        p = mp.Process(target=process_directory,
                       args=(root, files, dst, tmp, args.tool, args.lang))
        processes.append(p)

    k = args.threads
    for i in range(0, len(processes), k):
        for p in processes[i:i + k]:
            p.start()
        for j, p in enumerate(processes[i:i + k]):
            p.join()
            print('=== {}/{} processed'.format(i + j + 1, len(processes)))

if __name__ == '__main__':
    main()
