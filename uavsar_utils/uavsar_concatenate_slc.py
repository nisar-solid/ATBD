#!/usr/bin/env python3
############################################################
# Script to concatenate UAVSAR segmets for isce processing.#
# Author: Bhuvan Varugu,Talib Oliver, 2022                               #
############################################################

import os
import argparse
import sys
import json
import shelve
from osgeo import gdal
from osgeo.gdal import GA_ReadOnly
import numpy as np
import isce
import isceobj

def createParser():
    EXAMPLE = """example:
      uavsar_concatenate_slc.py -w ./ -s 1 2 3 -o SLC_merged
    """
    parser = argparse.ArgumentParser(description='Concatenate 2 UAVSAR segmets for isce processing',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE)

    parser.add_argument('-w', '--work_dir', dest='work_dir', type=str, required=True,
            help='work dir containing slc segments.')
    parser.add_argument('-s', '--segments', dest='segments', type=int, nargs='*',default=['1'],required=True,
            help='list of slc segments to be concatenated,default=1.')
    parser.add_argument('-o', '--slc_out', dest='slc_merged_dir', type=str, required=True,
            help='merged out directory.')
    return parser


def cmdLineParse(iargs=None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args = iargs)

GDAL2NUMPY_DATATYPE = {

1 : np.uint8,
2 : np.uint16,
3 : np.int16,
4 : np.uint32,
5 : np.int32,
6 : np.float32,
7 : np.float64,
10: np.complex64,
11: np.complex128,

}
def read(file, processor='ISCE' , bands=None , dataType=None):
    ''' reader based on GDAL.

    Args:
        * file      -> File name to be read
    Kwargs:
        * processor -> the processor used for the InSAR processing. default: ISCE
        * bands     -> a list of bands to be extracted. If not specified all bands will be extracted.
        * dataType  -> if not specified, it will be extracted from the data itself
    Returns:
        * data : A numpy array with dimensions : number_of_bands * length * width
    '''

    if processor == 'ISCE':
        cmd = 'isce2gis.py envi -i ' + file
        os.system(cmd)

    dataset = gdal.Open(file,GA_ReadOnly)

    ######################################
    # if the bands have not been specified, all bands will be extracted
    if bands is None:
        bands = range(1,dataset.RasterCount+1)
    ######################################
    # if dataType is not known let's get it from the data:
    if dataType is None:
        band = dataset.GetRasterBand(1)
        dataType =  GDAL2NUMPY_DATATYPE[band.DataType]

    ######################################
    # Form a numpy array of zeros with the the shape of (number of bands * length * width) and a given data type
    data = np.zeros((len(bands), dataset.RasterYSize, dataset.RasterXSize),dtype=dataType)
    ######################################
    # Fill the array with the Raster bands
    idx=0
    for i in bands:
       band=dataset.GetRasterBand(i)
       data[idx,:,:] = band.ReadAsArray()
       idx+=1

    dataset = None
    return data


def write(raster, fileName, nbands, bandType):

    ############
    # Create the file
    driver = gdal.GetDriverByName( 'ENVI' )
    dst_ds = driver.Create(fileName, raster.shape[1], raster.shape[0], nbands, bandType )
    dst_ds.GetRasterBand(1).WriteArray( raster, 0 ,0 )

    dst_ds = None

def getShape(file):

    dataset = gdal.Open(file,GA_ReadOnly)
    return dataset.RasterXSize , dataset.RasterYSize

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
    print(fname)
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

def write_xml(shelveFile, slcFile, length, width):
    with shelve.open(shelveFile,flag='r') as db:
        frame = db['frame']

    # length = frame.numberOfLines
    # width = frame.numberOfSamples

    print (frame) ## To test
    print (width,length)

    slc = isceobj.createSlcImage()
    slc.setWidth(width)
    slc.setLength(length)
    slc.filename = slcFile
    slc.setAccessMode('write')
    slc.renderHdr()
    slc.renderVRT()

##### Main #####
def main(iargs=None):
    inps = cmdLineParse(iargs) # read inputs
    work_dir = os.path.expanduser(inps.work_dir) # go to work dir
    os.chdir(work_dir)
    print('Go to directory: '+ work_dir);
    slc_merged_dir = os.path.expanduser(inps.slc_merged_dir)
    segments = inps.segments;
    print('segments:',segments);
    if not os.path.exists(inps.slc_merged_dir):
        os.mkdir(inps.slc_merged_dir)
    seg1_dir = 'SLC_seg{}'.format(segments[0]);
    # Check slc segment folders
    #images_json = os.path.join(work_dir, 'images.json')
    #with open(images_json) as fp:
    #    images = json.load(fp)
    #id = list(images.keys())[0]
    slc_list = sorted(os.listdir(seg1_dir));
    # create shelve folder
    for a in slc_list:
        slc_segs=[];
        for s in range(len(segments)):
            slc_path = (os.path.join(seg1_dir.replace(str(segments[0]), str(segments[s])), a,a)+'.slc');
            slc = read(slc_path, processor='ISCE' , bands=None , dataType=None);
            slc_segs.append(slc)
        slc_merged_path= slc_merged_dir + '/' + a;
        if not os.path.exists(slc_merged_path):
            os.mkdir(slc_merged_path)
        slc_merged_file = slc_merged_path + '/' + a + '.slc';
        source_shelveFile = (os.path.join(seg1_dir, a)) + '/data';
        slc_merged = np.concatenate(tuple(slc_segs), axis=1);
        print ('merged slc size', slc_merged.dtype, slc_merged.size, slc_merged.shape)
        length, width = slc_merged.shape[1], slc_merged.shape[2]
        slc_merged.tofile(slc_merged_file)

        cmd = 'cp ' + source_shelveFile +'*'+ ' ' + slc_merged_path +'/'
        print (cmd)
        os.system(cmd)
        bottom_slc_shelve= os.path.expanduser(seg1_dir.replace(str(segments[0]), str(segments[-1])))+'/'+a+'/data';
        lrc,llc = get_corners(bottom_slc_shelve);
        shelveFile = slc_merged_path + '/data'
        update_shelve(shelveFile, length, lrc, llc);
        print ('shelve file -->', shelveFile)
        slcFile = os.path.join(slc_merged_path, os.path.basename(slc_merged_file))
        print ('slcfile--->', slcFile)
        write_xml(shelveFile, slcFile, length, width)
###################
if __name__ == "__main__":
    main(sys.argv[1:])
