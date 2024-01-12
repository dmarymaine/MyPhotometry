
import os
import numpy as np
import matplotlib.pyplot as plt
import photutils as pht
import glob
import rawpy
import pyexiv2
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.utils import calc_total_error
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from datetime import datetime

import MYPhot_Logging as log

logger = log.getLogger("MYPhot_Core")

class MYPhot_Core:

   def __init__(self,args):

     self.workdir = args.workdir
     self.target = args.target
     self.apertures = args.apertures
     self.cr2 = args.cr2
     self.preprocessing = args.preprocessing
     self.bias = False
     self.filename = {}


   def do_fits_preprocessing(self):
     """ Perform pre-processing aka bias,dark and flat correction
         of input data in FITS format """
     pass

   def convert_cr2_fits(self):
     """ Perform pre-processing aka bias, dark and flat correction
         of input data in CR2 format """

     # transform CR2 files into FITS files and then proceed with
     # do_fits_preprocessing
     logger.info("Create BIAS FITS files")
     if os.path.exists(self.workdir+"/BIAS"):
       self.bias = True
       bfiles = glob.glob(self.workdir+"/BIAS/*cr2")
       bfiles.sort()

       for i,ifile in enumerate(bfile):
         logger.info(f"reading bias: {i+1} of {len(bfiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]

         # average the two channels and remove CANON offset
         green = 0.5*(g1+g2) - 2047

         # prepare to write fits
         image.pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green)
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[0] == 'Exif.Photo.ExposureTime'):
             first,second=it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/bias_{i+1}.fits',overwrite=True)

     logger.info("Create DARK FITS files")
     if os.path.exists(self.workdir+"/DARK"):
       dfiles = glob.glob(self.workdir+"/DARK/*.cr2")
       dfiles.sort()

       for i,ifile in enumerate(dfiles):
         logger.info(f"reading dark: {i+1} of {len(dfiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]

         green = 0.5*(g1+g2) - 2047

         # prepare to write fits
         image.pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green)
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/dark_{i+1}.fits',overwrite=True)

     logger.info("Create DARKFLAT FITS files")
     if os.path.exists(self.workdir+"/DARKFLAT"):
       dfiles = glob.glob(self.workdir+"/DARKFLAT/*.cr2")
       dfiles.sort()

       for i,ifile in enumerate(dfiles):
         logger.info(f"reading dark-flat: {i+1} of {len(dfiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]

         green = 0.5*(g1+g2) - 2047

         # prepare to write fits
         image.pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green)
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/darkflat_{i+1}.fits',overwrite=True)

     logger.info("Create FLAT FITS files")
     if os.path.exists(self.workdir+"/FLAT"):
       ffiles = glob.glob(self.workdir+"/FLAT/*.cr2")
       ffiles.sort()

       for i,ifile in enumerate(dfiles):
         logger.info(f"reading flat: {i+1} of {len(ffiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]

         green = 0.5*(g1+g2) - 2047

         # prepare to write fits
         image.pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green)
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/flat_{i+1}.fits',overwrite=True)

     logger.info("Create LIGHT FITS files")
     if os.path.exists(self.workdir+"/LIGHT"):
       lfiles = glob.glob(self.workdir+"/LIGHT/*.cr2")
       lfiles.sort()

       for i,ifile in enumerate(dfiles):
         logger.info(f"reading light: {i+1} of {len(lfiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]

         green = 0.5*(g1+g2) - 2047

         # prepare to write fits
         image.pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green)
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/light_{i+1}.fits',overwrite=True)


   def exec(self):
     """ main point with the actual execution of the main steps.
         if preprocessing has to be executed produce the results
         if not populate the variable filename with the already
         calibrated data
     """

     if self.preprocessing:
       if self.cr2:
         self.convert_cr2_fits()

       self.do_fits_preprocessing()

     else:
       rfiles = glob.glob(args.workdir+"Reduced/light*calib*.fits")
       rfiles.sort()
       self.filename = rfiles
