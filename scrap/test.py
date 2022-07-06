import sys
import argparse

parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
parser.add_argument('--root', type=str,
                        help='location of the results')

args = parser.parse_args()

print(args.root)