#!/usr/bin/env python
usage = """
Usage: cat_and_interfere.py [options] images.json pairs.txt

Concatenate SLC segments and generate the requested interferograms with
amplitude and correlation layers.

Options:
    -d <directory>
        Output directory to store interferograms (default=./int).

    -r <range_looks>
        Number of looks to take in range (default=3).

    -a <azimuth_looks>
        Number of looks to take in azimuth (default=12).

The first input file is in JSON format and describes each SLC image.  The data
are structured in the form
{
    "img1": {
        "annotation": "foo.ann",
        "segments": {
            "1": "foo_s1.slc",
            "2": "foo_s2.slc"
        }
    },
    "img2": {...},
...}
which can be generated using the script group_segments.py.

The second input file is a text file listing the pairs to be generated, one
per line.  The pairs are described as two keys from images.json joined by an
underscore, e.g. img1_img2.  The file can be generated using the script pair.py.

The output files will be have .int, .amp, and .cor extensions added to the
strings in the pairs file.  If these files already exist they will not be
regenerated, allowing you to easily add members to a data stack as they become
available.
"""

from getopt import getopt
import json
import logging
import os
import sys
import numpy as np

from pair import read_pairs
from insar import ImageReader, Interferogram, correlation

log = logging.getLogger('cat_and_interfere')


def get_slc_samples(ann):
    """Figure out number of SLC samples (columns) given its annotation file.
    """
    n = None
    # Ad-hoc parser, sorry.
    with open(ann) as f:
        for line in f:
            if line.startswith('slc_1_1x1 Columns'):
                n = int(line.split('=')[1].split()[0])
    if n is None:
        raise IOError('Could not find SLC columns in annotation file.')
    return n


def interfere_groups(group0, group1, intf, amp, n, looks=(1,1)):
    """Given two groups of SLC segments with n columns, form the concatenated
    interferogram and amplitude files, optionally averaged down by
    (azimuth,range) looks.
    """
    log.info('Generating interferogram %s looked %dx%d', os.path.basename(intf),
             looks[1], looks[0])
    # List of segments (sorted as integers).
    segs = sorted(group0, key=int)
    igram = Interferogram(n, looks)
    with open(intf, 'wb') as fint, open(amp, 'wb') as famp:
        for i in segs:
            img0 = ImageReader(group0[i], n, dtype='complex64')
            img1 = ImageReader(group1[i], n, dtype='complex64')
            log.info('... %s * conj(%s)', img0.filename, img1.filename)
            for int_row, amp_row in igram.iterrows(img0, img1):
                int_row.tofile(fint)
                amp_row.tofile(famp)

def main(argv):
    # Parse command line arguments.
    try:
        options = "d:r:a:h"
        opts, args = getopt(argv[1:], options)
    except:
        print (usage)
        sys.exit(1)

    # Set up defaults.
    looks = [12, 3]
    intdir = 'int'

    # Process options.
    for o, a in opts:
        if o == '-d':
            intdir = a
        elif o == '-r':
            looks[1] = int(a)
        elif o == '-a':
            looks[0] = int(a)
        elif o == '-h':
            print (usage)
            sys.exit(0)
        else:
            assert False, 'Unhandled option.'

    if len(args) < 2:
        print (usage)
        sys.exit(1)

    images_json = args[0]
    pairs_txt = args[1]

    # Load input files.
    with open(images_json) as fp:
        images = json.load(fp)
    with open(pairs_txt) as fp:
        pairs = read_pairs(fp)
    # Number of samples should be the same for all SLCs in the stack.
    id = list(images.keys())[0] #### Modified for Python 3
    n = get_slc_samples(images[id]['annotation'])
    # Log processing parameters to a machine-readable file.
    info = {
        'images': os.path.abspath(images_json),
        'pairs': os.path.abspath(pairs_txt),
        'looks_range': looks[1],
        'looks_azimuth': looks[0],
        'samples_slc': n,
        'samples_int': n // looks[1],
    }
    if not os.path.exists(intdir):
        os.mkdir(intdir)
    with open(os.path.join(intdir, 'int.json'), 'w') as fp:
        json.dump(info, fp, indent=2, sort_keys=True)
    # Form all the interferograms.
    for a, b in pairs:
        id = a + '_' + b
        int_path = os.path.join(intdir, id) 
        if not os.path.exists(int_path):
            os.mkdir(int_path)
        #intf = os.path.join(intdir, id + '.int') #original
        intf = os.path.join(int_path, id + '.int') ## mintpy format
        amp = intf[:-4] + '.amp'
        cor = intf[:-4] + '.cor'
        if all(os.path.exists(f) for f in (intf, amp, cor)):
            log.info('Skipping %s because it already exists.', id)
            continue
        interfere_groups(images[a]['segments'], images[b]['segments'],
                         intf, amp, n, looks)
        correlation(intf, amp, cor)


if __name__ == '__main__':
    log_level = logging.DEBUG
    log.setLevel (log_level)
    sh = logging.StreamHandler()
    sh.setLevel (log_level)
    log.addHandler (sh)

    main(sys.argv)
