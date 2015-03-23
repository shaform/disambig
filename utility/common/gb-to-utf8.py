"""Convert entire directory from gb to utf8"""
import argparse
import errno
import os

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
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = process_commands()
    for root, dirs, files in os.walk(args.input):
        mkdir_p(os.path.join(args.output, os.path.relpath(root, args.input)))
        for fname in files:
            source = os.path.join(root, fname)
            dest = os.path.join(
                args.output, os.path.relpath(source, args.input))
            print(root, fname)
            with open(source, 'r') as sf, open(dest, 'w') as df:
                cmdline = 'iconv -f gb18030 -t utf8'
                Popen(cmdline, shell=True, stdout=df, stdin=sf)
