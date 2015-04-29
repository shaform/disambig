import argparse


def process_commands():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='filtered corpus file')
    parser.add_argument('--output', required=True,
                        help='output corpus file')

    return parser.parse_args()


def main():
    args = process_commands()

    with open(args.input) as f, open(args.output, 'w') as out:
        for l in f:
            head, tokens = l.strip().split(' ', 1)
            _, cnnct, sense, num = head.split('-')

            out.write('{}\t{}\t{}\t{}\n'.format(num, cnnct, sense, tokens))

if __name__ == '__main__':
    main()
