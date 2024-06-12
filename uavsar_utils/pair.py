#!/usr/bin/env python
usage = """
Usage: pair.py [options]

Read in a description of SLC images and write out a file listing the
interferometric pairs.

Options:
    -i <filename>
        Input JSON file describing SLCs (output of group_segments.py).
        If not supplied, read from stdin.

    -o <filename>
        Output file describing pairs.  Defaults to stdout.

    -n <lags>
        Maximum number of pairs to generate for each image, nearest in time.
        Default lags=1 for adjacent pairs.
"""

from getopt import getopt
import json
import os
import sys


def generate_pairs(ids, lags=1):
    """Given a sorted list of SLCs return a list of pairs.  Up to N=lags pairs
    will be generated for each image, nearest in time.
    """
    n = len(ids)
    pairs = []
    for i in range(n-1):
        for j in range(i+1, min(i+1+lags, n)):
            pairs.append([ids[i], ids[j]])
    return pairs


def write_pairs(pairs, fd):
    for a, b in pairs:
        # Make sure we'll be able to parse this later.
        cat = a + b
        assert '_' not in cat, "Cannot have underscores in image IDs."
        assert cat == cat.strip(), "Cannot have spaces in image IDs."
        fd.write('%s_%s\n' % (a,b))


def read_pairs(fd):
    pairs = []
    for line in fd:
        # Parser validated by asserts in write_pairs().
        a, b = line.strip().split('_')
        pairs.append([a,b])
    return pairs


def main(argv):
    try:
        options = "i:o:n:h"
        opts, args = getopt(argv[1:], options)
    except:
        print (usage) 
        sys.exit(1)

    # Set up defaults.
    lags = 1
    images_json = sys.stdin
    pairs_file = sys.stdout

    # Process options.
    for o, a in opts:
        if o == '-i':
            images_json = open(a, 'r')
        elif o == '-o':
            pairs_file = open(a, 'w')
        elif o == '-n':
            lags = int(a)
        elif o == '-h':
            print (usage)
            sys.exit(0)
        else:
            assert False, 'Unhandled option.'

    # Read description of SLC images, e.g., from group_segments.py.
    images = json.load(images_json)
    # Assume image IDs are in format YYYYMMDDhh or a similar format that
    # sorts in time order.
    ids = sorted(images.keys())
    pairs = generate_pairs(ids, lags)
    write_pairs(pairs, pairs_file)


if __name__ == '__main__':
    main(sys.argv)
