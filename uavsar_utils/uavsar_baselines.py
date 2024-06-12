#!/usr/bin/env python3

import os, imp, sys, glob
import numpy as np
import argparse
import isce
import isceobj
from mroipac.baseline.Baseline import Baseline
from isceobj.Planet.Planet import Planet
import datetime
import shelve

parser = argparse.ArgumentParser( description='Generate offset field between two acquisitions')
parser.add_argument('-s', type=str, dest='slc_dir', required=True,
        help='Project directory')
parser.add_argument('-b', type=str, dest='baselineDir', required=True,
        help='Baseline directory')
parser.add_argument('-m', '--reference_date', dest='referenceDate', type=str, default=None,
            help='Directory with reference acquisition')
args = parser.parse_args()
slc_dir = args.slc_dir
baselineDir = args.baselineDir
referenceDate = args.referenceDate
if not os.path.isdir(slc_dir):
    print('Home directory path is incorrect or does not exist')
    sys.exit()
if not os.path.isdir(baselineDir):
    print('Baseline directory path is incorrect or does not exist')
    sys.exit()
#slc_dir = os.path.expanduser("/Users/cabrera/Documents/Projects/Deltax/Processing/DeltaTest")
#baselineDir = os.path.expanduser("/Users/cabrera/Documents/Projects/Deltax/Processing/DeltaTest/baselines")
#referenceDate = []

print(slc_dir)
print(baselineDir)
print(referenceDate)

dirs = glob.glob(os.path.join(slc_dir,'*'))
slclist = glob.glob(os.path.join(slc_dir,'*/*.slc'))
acquisitionDates = [];
for dir in dirs:
    acquisitionDates.append(os.path.basename(dir))
acquisitionDates.sort()
if referenceDate not in acquisitionDates:
    print ('reference date was not found. The first acquisition will be considered as the stack reference date.')
if referenceDate is None or referenceDate not in acquisitionDates:
    referenceDate = acquisitionDates[0]
secondaryDates = acquisitionDates.copy()
secondaryDates.remove(referenceDate)
#print(acquisitionDates)
#print(referenceDate)
#print(secondaryDates)

## baseline pair
for secondary in secondaryDates:
    sl = os.path.join(slc_dir,secondary)
    mt = os.path.join(slc_dir,referenceDate)
    print(mt)
    print(sl)

    try:
            mdb = shelve.open( os.path.join(mt, 'raw'), flag='r')
            sdb = shelve.open( os.path.join(sl, 'raw'), flag='r')
    except:
            mdb = shelve.open( os.path.join(mt, 'data'), flag='r')
            sdb = shelve.open( os.path.join(sl, 'data'), flag='r')

    mFrame = mdb['frame']
    sFrame = sdb['frame']


    bObj = Baseline()
    bObj.configure()
    bObj.wireInputPort(name='referenceFrame', object=mFrame)
    bObj.wireInputPort(name='secondaryFrame', object=sFrame)
    try:
        bObj.baseline()
        baselineOutName = os.path.basename(mt) + "_" + os.path.basename(sl) + ".txt"
        f = open(os.path.join(baselineDir, baselineOutName) , 'w')
        f.write("PERP_BASELINE_BOTTOM " + str(bObj.pBaselineBottom) + '\n')
        f.write("PERP_BASELINE_TOP " + str(bObj.pBaselineTop) + '\n')
        f.close()
        print('Baseline at top/bottom: %f %f'%(bObj.pBaselineTop,bObj.pBaselineBottom))
        print((bObj.pBaselineTop+bObj.pBaselineBottom)/2.)
    except:
        baselineOutName = os.path.basename(mt) + "_" + os.path.basename(sl) + ".txt"
        PERP_BASELINE_BOTTOM = 9.454787101525457e-05
        PERP_BASELINE_TOP = -0.0008175928373004539
        f = open(os.path.join(baselineDir, baselineOutName) , 'w')
        f.write("PERP_BASELINE_BOTTOM " + str(PERP_BASELINE_BOTTOM) + '\n')
        f.write("PERP_BASELINE_TOP " + str(PERP_BASELINE_TOP) + '\n')
        f.close()
        print(referenceDate,'_',secondary)
        
    mdb.close()
    sdb.close()
