#!/usr/bin/env python3
############################################################
# Script to concatenate UAVSAR segmets for isce processing.#
# Author: Talib Oliver, 2021                               #
############################################################

import isce
from isceobj.Sensor import createSensor
import shelve
import argparse
import glob
from isceobj.Util import Poly1D
from isceobj.Planet.AstronomicalHandbook import Const
import os

def cmdLineParse():
    '''
    Command line parser.
    '''

    parser = argparse.ArgumentParser(description='Update shelve file for UAVSAR.')
    parser.add_argument('-i','--input', dest='workdir', type=str,
            required=True, help='Input UAVSAR directory')
    parser.add_argument('-d','--shelve', dest='shelve_file', type=str,
            default=None, help='Shelve file')
    parser.add_argument('-l', '--length', dest='length', type=int,
            required=True, help='Length of merged file')
    parser.add_argument('-s2', '--seg2', dest='seg2', type=str,
            required=True, help='path to segment2 shelve file')
    return parser.parse_args()

def get_corners(seg2):
    with shelve.open(seg2, flag='r') as mdb:
        frame = mdb['frame']
    lrc = frame.lowerRightCorner 
    llc = frame.lowerLeftCorner

    return lrc, llc

def update_shelve(fname, length, lrc, llc):
    '''
    Update shelve file with new length.
    '''
    #print(fname)
    shelveFile = shelve.open(fname, writeback = True)
    frame = shelveFile['frame']
    ## inputting total values we want to update 
    ## to the already existing list in shelf_file.
    frame.numberOfLines = length
    frame.lowerRightCorner = lrc
    frame.lowerLeftCorner = llc
    shelveFile.sync() 
    ## now, we close the file 'shelf_file'.
    shelveFile.close()


if __name__ == '__main__':
    '''
    Main driver.
    '''

    inps = cmdLineParse()
    work_dir = os.path.expanduser(inps.workdir) #go to work dir
    #os.chdir(work_dir) 

    lrc, llc = get_corners(inps.seg2) 

    update_shelve(inps.shelve_file, inps.length, lrc, llc)
