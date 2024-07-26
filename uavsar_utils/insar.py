#!/usr/bin/env python

#from itertools import zip, zip_longest, tee
import itertools as mit
import numpy as np
import os
import isce
import isceobj


# Taken from raster.py by Brian Hawkins, JPL.
class ImageReader (object):
    """ImageReader(filename, n, dtype='float32')

    Open an image for iterating over its rows.
    """
    def __init__ (self, filename, n, dtype='float32', blocksize=1024):
        self.filename = filename
        n = int(n)
        self.dtype = np.dtype(dtype)
        self.blocksize = blocksize
        m = os.path.getsize(self.filename) // (n * self.dtype.itemsize) ## changed by talib to have an integer output added "//"
        self.shape = m, n
        self.fid = open(self.filename, mode='rb')

    def __iter__(self):
        # Big blocks speed up IO.
        m, n = self.shape
        self.fid.seek(0)
        for i in list(range(0, m, self.blocksize)):
            nrows = min(self.blocksize, m-i)
            offset = i * n * self.dtype.itemsize
            block = np.memmap(self.fid, dtype=self.dtype, shape=(nrows,n),
                              mode='r', offset=offset)
            for k in list(range(nrows)):
                yield block[k,:]
            del block
        self.fid.close()


def identity(x):
    return x


class Multilook(object):
    """Multilook(n, looks=(1,1), dtype='complex64')

    Boxcar filter a 2D image with n columns.
    Specify filter size with looks=(rows,columns).
    Specify data type with a standard numpy type/descriptor.

    Supply input data to the iterrows method to get a generator of output data.
    This scheme lets you avoid holding whole images in memory.

    Equivalent to Fortran program cpxlooks.
    """
    # This function runs on the input data before averaging.
    _pre = staticmethod(identity)
    # This function runs on the output data.
    _post = staticmethod(identity)

    def __init__(self, n, looks=(1,1), dtype='complex64'):
        self.n = n
        self.looks = looks
        self.scale = 1.0 / (looks[0] * looks[1])
        self.nout = n // looks[1]
        self.dtype = np.dtype(dtype)
        self.i = 0
        self.buf = np.zeros((looks[0], n), dtype=self.dtype)

    def iterrows(self, rows):
        """Accumulate rows of input data and generate rows of output data.

        NOTE: Care must be taken when repeating calls to iterrows().
        It can consume rows after its last yield statement, which can lead to
        problems when combining data streams with zip.  See
        http://stackoverflow.com/questions/33818422/can-i-yield-from-an-instance-method
        """
        for row in rows:
            if len(row) != self.n:
                raise ValueError('Expected input length = %d.' % self.n)
            self.buf[self.i,:] = row
            self.i += 1
            # If buffers are full, perform looks and yield data.
            if self.i == self.looks[0]:
                yield self._flush()

    def _flush(self):
        # Apply pre-processing function to data.
        x = self._pre(self.buf)
        # Sum in azimuth and discard samples past integer multiple of looks.
        nfull = self.nout * self.looks[1]
        y = np.sum(x, axis=0)[:nfull]
        # Sum in range by adding a dimension and summing rows.
        y.shape = (self.nout, self.looks[1])
        z = self.scale * np.sum(y, axis=1)
        # Apply post-processing function.
        out = self._post(z)
        # Reset buffers.
        self.i = 0
        self.buf[...] = 0.0
        return out


class AmplitudeLooks(Multilook):
    """AmplitudeLooks(n, looks=(1,1), dtype='complex64')

    Multilook class adapted for packed two-channel amplitude data, e.g.
        z_in = abs(z1) + 1j*abs(z2)
    Computes the boxcar filter over the powers, returning
        z_out = sqrt(E[abs(z1)**2]) + 1j*sqrt(E[abs(z2)**2])

    Equivalent to Fortran program rilooks.
    """
    @staticmethod
    def _pre(z):
        return z.real**2 + 1j*z.imag**2

    @staticmethod
    def _post(z):
        return np.sqrt(z.real) + 1j*np.sqrt(z.imag)


class PowerLooks(Multilook):
    """PowerLooks(n, looks=(1,1), dtype='complex64')

    Computes the boxcar filter over the image power, returning
        out = E[abs(z)**2]

    Equivalent to Fortran program powlooks.
    """
    @staticmethod
    def _pre(z):
        return np.abs(z)**2


class Interferogram(object):
    """Interferogram(n, looks=(1,1))

    Takes two-channels of n-column SLC data and produces interferogram and
    amplitude layers multilooked by (azimuth,range) looks.

    After initialization supply the SLC images to the iterrows method.

    NOTE: Time-domain multiplication of two images results in a signal with
    double the bandwidth.  This class assumes the input sample rate is
    sufficient to accomodate this without aliasing.
    """
    def __init__(self, n, looks=(1,1)):
        self.int = Multilook(n, looks, dtype='complex64')
        self.amp = AmplitudeLooks(n, looks, dtype='complex64')

    def iterrows(self, z1, z2):
        """Consumes rows of SLC images z1 and z2, generates rows of (int,amp)
        where
            int = E[z1 * z2.conj]
        and
            amp = sqrt(E[abs(z1)**2]) + 1j*sqrt(E[abs(z2)**2])
        """
        # Define generators to feed data to multilook filters.
        def gen_int_slc(z1, z2):
            for a, b in zip(z1, z2):
                yield a * b.conj()

        def gen_amp_slc(z1, z2):
            for a, b in zip(z1, z2):
                yield np.abs(a) + 1j*np.abs(b)

        # Clone the input streams so we can feed them to each of the above.
        z1int, z1amp = mit.tee(z1)
        z2int, z2amp = mit.tee(z2)
        # Feed multilook filters at the same time to avoid excessive caching.
        gen_int = self.int.iterrows(gen_int_slc(z1int, z2int))
        gen_amp = self.amp.iterrows(gen_amp_slc(z1amp, z2amp))
        # Need to use zip_longest because we need both int and amp to consume
        # all the input data, even after their last yield.
        for int, amp in mit.zip_longest(gen_int, gen_amp):
            assert int is not None and amp is not None
            yield int, amp


def cpxlooks(name_in, name_out, n, looks=(1,1)):
    """Equivalent to cpxlooks.f program for complex data.

    name_in     flat file containing an array of complex64 data in n columns.
    name_out    similar flat file with (n // looks[1]) columns.
    looks       tuple of (rows,columns) describing the size of the filter.

    See also the Multilook class.
    """
    img = ImageReader(name_in, n, dtype='complex64')
    with open(name_out, 'wb') as f:
        for row in Multilook(n, looks, dtype='complex64').iterrows(img):
            row.tofile(f)


def rilooks(name_in, name_out, n, looks=(1,1)):
    """Equivalent to rilooks.f program for two-channel amplitude data.

    name_in     flat file containing an array of complex64 data in n columns.
    name_out    similar flat file with (n // looks[1]) columns.
    looks       tuple of (rows,columns) describing the size of the filter.

    See also the AmplitudeLooks class.
    """
    img = ImageReader(name_in, n, dtype='complex64')
    with open(name_out, 'wb') as f:
        for row in AmplitudeLooks(n, looks, dtype='complex64').iterrows(img):
            row.tofile(f)


def correlation(intf, amp, cor):
    """Equivalent to makecc.f program.  Namely, compute the correlation
    coefficient
        cor = abs(E[z1 * conj(z2)]) / sqrt(E[abs(z1)**2] * E[abs(z2)**2])
    given the interferogram file containing values
        intf = E[z1 * conj(z2)]
    and the amplitude file containing values
        amp = sqrt(E[abs(z1)**2]) + 1j*sqrt(E[abs(z2)**2])

    Both input files are assumed to have type 'complex64', and the output
    file will have type 'float32'.
    """
    nbuf = 8192
    fint = open(intf, 'rb')
    famp = open(amp, 'rb')
    fcor = open(cor, 'wb')
    while True:
        # Read data in blocks.
        z = np.fromfile(fint, count=nbuf, dtype='complex64')
        a = np.fromfile(famp, count=nbuf, dtype='complex64')
        # Don't fail if one file is longer for some reason.
        n = min(len(z), len(a))
        z = z[:n]
        a = a[:n]
        # Avoid divide-by-zero, use zero as the null value.
        p = a.real * a.imag
        mask = p > 0.0
        c = np.zeros(n, dtype='float32')
        # Compute the correlation and save to file.
        c[mask] = np.abs(z[mask]) / p[mask]
        c.tofile(fcor)
        if n < nbuf:
            break
    for f in (fint, famp, fcor):
        f.close()
        
def isce_xml_ifg(ifg_file, amp_file, coh_file, samples):
    outInt = isceobj.Image.createIntImage()
    outInt.setFilename(ifg_file)
    outInt.setWidth(samples)
    outInt.setAccessMode('read')
    outInt.renderHdr()
    outInt.renderVRT()

    outAmp = isceobj.Image.createAmpImage()
    outAmp.setFilename(amp_file)
    outAmp.setWidth(samples)
    outAmp.setAccessMode('read')
    outAmp.renderHdr()
    outAmp.renderVRT()

    outCor = isceobj.Image.createImage()
    outCor.setFilename(coh_file)
    outCor.setWidth(samples)
    outCor.setAccessMode('read')
    outCor.setDataType('FLOAT')
    outCor.renderHdr()
    outCor.renderVRT()
