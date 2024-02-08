
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import photutils as pht
import glob
import rawpy
import pyexiv2
import json
import requests
import scipy
from scipy.optimize import curve_fit
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats, SigmaClip, mad_std
from photutils.utils import calc_total_error
from photutils.centroids import centroid_quadratic
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.profiles import RadialProfile
from PythonPhot import aper
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from datetime import datetime
matplotlib.use('TkAgg')

import MYPhot_Logging as log

logger = log.getLogger("MYPhot_Core")
pyexiv2.set_log_level(4)
obscode = "MDAS"
obs_location = EarthLocation(lat=45.53*u.deg,lon=9.4*u.deg,height=133*u.m)

def fn1(x,a,b):
  return a + b*x[0]

def fn2(x,a,b,c):
  return a + b*x[0] + c*x[1]

class MYPhot_Core:

   def __init__(self,args):

     self.workdir = args.workdir
     self.target = args.target
     self.apertures = args.apertures
     self.naper = None
     self.cr2 = args.cr2
     self.preprocessing = args.preprocessing
     self.bias = False
     self.filename = []
     self.showplots = args.showplots
     self.gain = None
     self.rdnoise = None
     self.out_target = None
     self.out_compar = None
     self.out_valide = None
     self.phot = None
     self.inn = None
     self.out = None
     self.x_t = None
     self.y_t = None
     self.x_c = None
     self.y_c = None
     self.x_v = None
     self.y_v = None
     self.magmax = args.magmax
     self.filters = ['V','B']
     self.maglim = args.maglim
     self.correct_bv = None
     self.t_mag_ave = None
     self.v_mag_ave = None
     self.t_mag_std = None
     self.v_mag_std = None
     self.JohnV_t = None
     self.JohnV_v = None
     self.jd = []
     self.data = []
     self.xmass = []
     self.result = []
     self.check_V = []
     self.list_obj = args.list_obj
     self.airmass = None
     self.chartid = None
     self.slope  = None
     self.slope1 = None
     self.slope2 = None
     self.ZPoint = None
     self.ZPoint1 = None
     self.target_bv = None
     self.target_compar = None


   def set_output_data(self,files):
     with open(self.apertures,'r') as g:
       aper_radii = json.load(g)

     naper = len(aper_radii)
     self.out_target = np.zeros((1+2*naper,len(files),len(self.filters)))
     self.out_compar = np.zeros((1+2*naper,len(files),len(self.filters)))
     self.out_valide = np.zeros((1+2*naper,len(files),len(self.filters)))
     self.x_t = np.zeros((len(files),len(self.filters)))
     self.y_t = np.zeros((len(files),len(self.filters)))
     self.x_c = np.zeros((len(files),len(self.filters)))
     self.y_c = np.zeros((len(files),len(self.filters)))
     self.x_v = np.zeros((len(files),len(self.filters)))
     self.y_v = np.zeros((len(files),len(self.filters)))
     self.phot = np.zeros((len(self.sources),len(self.filters)))
     self.inn = np.zeros((len(self.sources),len(self.filters)))
     self.out = np.zeros((len(self.sources),len(self.filters)))
     self.airmass = np.zeros((len(files)))

   def _get_green(self,data):
     """
     Extract the green channel from FITS image
     """
     h,w = data.shape 
     oh = 0.5*h 
     ow = 0.5*w 
  
     g1 = data[0::2,1::2]
     g2 = data[1::2,0::2]
     green = 0.5*(g1[:oh,:ow]+g2[:oh,:ow])

     return green

   def _get_blue(self,data):
     """
     Extract the blue channel from FITS image
     """  
     h,w = data.shape 
     oh = 0.5*h 
     ow = 0.5*w 
     b1 = data[1::2,1::2]
     blue = b1[:oh,:ow]

     return blue 


   def do_fits_preprocessing(self):
     """ Perform pre-processing aka bias,dark and flat correction
         of input data in FITS format """
     
     for filter in self.filters:
       # check if a master bias is already present in the Reduce folder     
       if os.path.isfile(f"{self.workdir}/Reduced/masterbias_{filter}.fits"):
         logger.info(f"Master Bias for filter {filter} already present - skipping master bias creation")
       else:
         os.system(f"mkdir {self.workdir}/Reduced")
         logger.info(f"Create Master Bias for filter {filter}")
         bfiles = glob.glob(f"{self.workdir}/BIAS/bias_{filter}_*.fits")
         bfiles.sort()
         allbias = []
         for i,ifile in enumerate(bfiles):
          logger.info(f"reading bias: {i+1}/{len(bfiles)} - {ifile}")
          data = fits.getdata(ifile)
          allbias.append(data)

         # stack bias together
         allbias = np.stack(allbias)
         superbias = np.median(allbias,axis=0)
         fits.writeto(f"{self.workdir}/Reduced/masterbias_{filter}.fits",superbias.astype('float32'),overwrite=True)

         if self.showplots:
           tvbias = fits.getdata(f"{self.workdir}/Reduced/masterbias_{filter}.fits")
           plt.figure(figsize=(8,8))
           plt.imshow(tvbias,origin='lower')
           plt.colorbar()
           plt.title(f"Master Bias derived from bias frames for filter {filter}")
           plt.show(block=True)
       
       # check if flat dark is present in the Reduce folder 
       if os.path.isfile(f"{self.workdir}/Reduced/masterdarkflat_{filter}.fits"):
         logger.info("Master Dark Flat already present - skipping creation")
       else:
         logger.info(f"Create Master Dark Flat for filter {filter}")
         dffiles = glob.glob(f"{self.workdir}/DARKFLAT/darkflat_{filter}_*.fits")
         dffiles.sort()
         alldarkflats = []
         for i,ifile in enumerate(dffiles):
           logger.info(f"reading dark-flat: {i+1}/{len(dffiles)} - {ifile}")
           data = fits.getdata(ifile)
           alldarkflats.append(data)

         # stack all dark flats together
         alldarkflats = np.stack(alldarkflats)
         mdarkflat = np.median(alldarkflats,axis=0)
         fits.writeto(f"{self.workdir}/Reduced/masterdarkflat_{filter}.fits",mdarkflat.astype('float32'),overwrite=True)
           
         if self.showplots:
          tvdflats = fits.getdata(f"{self.workdir}/Reduced/masterdarkflat_{filter}.fits")
          plt.figure(figsize=(8,8))
          plt.imshow(tvdflats,origin='lower')
          plt.colorbar()
          plt.title(f"Master Dark-Flat from dark-flat frames for filter {filter}")
          plt.show(block=True)
     
       # now combinte the flats: subtract BIAS and DARKFLAT
       # normalize the frames and create the master flat
       if os.path.isfile(f"{self.workdir}/Reduced/masterflat_{filter}.fits"):
         logger.info("Master Flat already exists - skipping creations")
       else:
         logger.info(f"Create Master Flat for filter {filter}")
         ffiles = glob.glob(f"{self.workdir}/FLAT/flat_{filter}_*fits")
         ffiles.sort()
         allflats = []

         masterbias = fits.getdata(f"{self.workdir}/Reduced/masterbias_{filter}.fits")
         masterdarkflat = fits.getdata(f"{self.workdir}/Reduced/masterdarkflat_{filter}.fits")

         for i,ifile in enumerate(ffiles):
           logger.info(f"reading flat: {i+1}/{len(ffiles)} - {ifile}")
           data = fits.getdata(ifile)- masterbias - masterdarkflat

           mflat = np.median(data)
           # normalize flat frame
           data/=mflat
           logger.info(f"median flat: {mflat}")
           allflats.append(data)
       
         allflats = np.stack(allflats)
         masterflat=np.median(allflats,axis=0)
         fits.writeto(f"{self.workdir}/Reduced/masterflat_{filter}.fits",masterflat.astype('float32'),overwrite=True)
      
         if self.showplots:
          tvflats = fits.getdata(f"{self.workdir}/Reduced/masterflat_{filter}.fits")
          plt.figure(figsize=(8,8))
          plt.imshow(tvflats)
          plt.colorbar()
          plt.title(f"Master Flat from Flat frames for filter {filter}")
          plt.show(block=True)
     
       # now combine the dark frames to create the master dark
       if os.path.isfile(f"{self.workdir}/Reduced/masterdark_{filter}.fits"):
         logger.info("Master Dark already present - skipping creation")
       else:
         logger.info(f"Creating Master Dark for filter {filter}")
         dfiles = glob.glob(f"{self.workdir}/DARK/dark_{filter}_*.fits")
         dfiles.sort()
         alldarks = []

         masterbias = fits.getdata(f"{self.workdir}/Reduced/masterbias_{filter}.fits")

         for i,ifile in enumerate(dfiles):
           logger.info(f"reading dark: {i+1}/{len(dfiles)} - {ifile}")
           data = fits.getdata(ifile) - masterbias
           alldarks.append(data)
       
         alldarks = np.stack(alldarks)
         masterdark = np.median(alldarks,axis=0)
         fits.writeto(f"{self.workdir}/Reduced/masterdark_{filter}.fits",masterdark.astype('float32'),overwrite=True)

         if self.showplots:
          tvdark = fits.getdata(f"{self.workdir}/Reduced/masterdark_{filter}.fits")
          plt.figure(figsize=(8,8))
          plt.imshow(tvdark)
          plt.colorbar()
          plt.title(f"Master Dark from Dark frames for filter {filter}")
          plt.show(block=True)

       # now get the Light frames and calibrate them with bias, dark and flats
       # also add keywords for auxiliari information
       # it takes the target (ra,dec) and set it to CRVAL1/CRVAL2
       # it computes gain and read-out noise from bias and flats

       if os.path.isfile(f"{self.workdir}/Reduced/p_light_{filter}_*fits"):
         logger.info("Light frames already calibrated - skipping light calibration")
       else:
         logger.info(f"Create calibrated Light frames for filter {filter}")
         lfiles = glob.glob(f"{self.workdir}/LIGHT/light_{filter}_*.fits")
         lfiles.sort()
       
         # compute gain and read-out noise
         self.gain,self.rdnoise = self._compute_gain_rnoise(filter)
       
         ra,dec = self.get_target_radec()

         masterbias = fits.getdata(f"{self.workdir}/Reduced/masterbias_{filter}.fits")
         masterflat = fits.getdata(f"{self.workdir}/Reduced/masterflat_{filter}.fits")
         masterdark = fits.getdata(f"{self.workdir}/Reduced/masterdark_{filter}.fits")

         for i,ifile in enumerate(lfiles):
           logger.info(f"reducing (debias,dark sub and flat-field) light: {i+1}/{len(lfiles)} - {ifile}")
           indir, infile = os.path.split(ifile)
           rootname,_ = os.path.splitext(infile)
           outfile = os.path.join(f"{self.workdir}/Reduced/p_{rootname}.fits")
           data = fits.getdata(ifile)
           head = fits.getheader(ifile,output_verifystr = "silentfix")

           # calibre light frames
           calib_data = (data[100:-100,100:-100] - masterbias[100:-100,100:-100] - masterdark[100:-100,100:-100])/masterflat[100:-100,100:-100]
           naxis1 = head['NAXIS1']-200
           naxis2 = head['NAXIS2']-200
           head['epoch'] = 2000.0
           head['NAXIS1'] = naxis1
           head['NAXIS2'] = naxis2
           head['CRVAL1'] = ra
           head['CRVAL2'] = dec
           head['CRPIX1'] = head['NAXIS1']/2
           head['CRPIX2'] = head['NAXIS2']/2
           head['FILTER'] = (filter,'Colour used from OSC bayer matrix')
           head['GAIN'] = (self.gain,'GAIN in e-/ADU')
           head['RDNOISE'] = (self.rdnoise,'read out noise in electron')

           fits.writeto(outfile,calib_data.astype('float32'),header=head,overwrite=True)
           self.filename.append(outfile)

   def convert_cr2_fits(self):
     """ Perform pre-processing aka bias, dark and flat correction
         of input data in CR2 format """

     # transform CR2 files into FITS files and then proceed with
     # do_fits_preprocessing
     logger.info("Create BIAS FITS files for Green channel")
     if os.path.exists(self.workdir+"/BIAS"):
       self.bias = True
       bfiles = glob.glob(self.workdir+"/BIAS/*cr2")
       bfiles.sort()

       for i,ifile in enumerate(bfiles):
         logger.info(f"reading bias: {i+1} of {len(bfiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]
         b1 = raw.raw_image[1::2,1::2]
         g1 = g1.astype('float32')
         g2 = g2.astype('float32')
         b1 = b1.astype('float32')

         green = 0.5*(g1+g2) - 2047
         b1 = b1 - 2047

         # prepare to write fits
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         hdu1 = fits.PrimaryHDU(b1.astype('float32'))

         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
             hdu1.header['ISO']=it[1]
           if (it[0] == 'Exif.Photo.ExposureTime'):
             first,second=it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)
             hdu1.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/BIAS/bias_V_{i+1}.fits',overwrite=True)
         hdu1.writeto(f'{self.workdir}/BIAS/bias_B_{i+1}.fits',overwrite=True)

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
         b1 = raw.raw_image[1::2,1::2]
         g1 = g1.astype('float32')
         g2 = g2.astype('float32')
         b1 = b1.astype('float32')

         green = 0.5*(g1+g2) - 2047
         b1 = b1 - 2047

         # prepare to write fits
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         hdu1 = fits.PrimaryHDU(b1.astype('float32'))
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
             hdu1.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)
             hdu1.header['EXPTIME']=np.float32(first)/np.float(second)

         hdu.writeto(f'{self.workdir}/DARK/dark_V_{i+1}.fits',overwrite=True)
         hdu1.writeto(f'{self.workdir}/DARK/dark_B_{i+1}.fits',overwrite=True)

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
         b1 = raw.raw_image[1::2,1::2] 
         g1 = g1.astype('float32')
         g2 = g2.astype('float32')
         b1 = b1.astype('float32')

         green = 0.5*(g1+g2) -2047
         b1 = b1 - 2047

         # prepare to write fits
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         hdu1 = fits.PrimaryHDU(b1.astype('float32'))
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
             hdu1.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)
             hdu1.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/DARKFLAT/darkflat_V_{i+1}.fits',overwrite=True)
         hdu1.writeto(f'{self.workdir}/DARKFLAT/darkflat_B_{i+1}.fits',overwrite=True)

     logger.info("Create FLAT FITS files")
     if os.path.exists(self.workdir+"/FLAT"):
       ffiles = glob.glob(self.workdir+"/FLAT/*.cr2")
       ffiles.sort()

       for i,ifile in enumerate(ffiles):
         logger.info(f"reading flat: {i+1} of {len(ffiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]
         b1 = raw.raw_image[1::2,1::2]
         g1 = g1.astype('float32')
         g2 = g2.astype('float32')
         b1 = b1.astype('float32')

         green = 0.5*(g1+g2) - 2047
         b1 = b1 - 2047

         # prepare to write fits
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         hdu1 = fits.PrimaryHDU(b1.astype('float32'))
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
             hdu1.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/FLAT/flat_V_{i+1}.fits',overwrite=True)
         hdu1.writeto(f'{self.workdir}/FLAT/flat_B_{i+1}.fits',overwrite=True)

     logger.info("Create LIGHT FITS files")
     if os.path.exists(self.workdir+"/LIGHT"):
       lfiles = glob.glob(self.workdir+"/LIGHT/*.cr2")
       lfiles.sort()

       for i,ifile in enumerate(lfiles):
         logger.info(f"reading light: {i+1} of {len(lfiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]
         b1 = raw.raw_image[1::2,1::2]
         g1 = g1.astype('float32')
         g2 = g2.astype('float32')
         b1 = b1.astype('float32')

         green = 0.5*(g1+g2) - 2047
         b1 = b1 - 2047

         # prepare to write fits
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         hdu1 = fits.PrimaryHDU(b1.astype('float32'))
         utcoffset = 2.*u.hour
         for it in items:
           if (it[0] == 'Exif.Image.DateTime'):
            b=datetime.strptime(it[1],'%Y:%m:%d %H:%M:%S')
            date_obs=f'{b.year}-{b.month}-{b.day}'
            timstart=f'{b.hour}:{b.minute}:{b.second}'
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
             hdu1.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)
             hdu1.header['EXPTIME']=np.float32(first)/np.float32(second)
         obs_location = EarthLocation(lat=45.53*u.deg,lon=9.4*u.deg,height=133*u.m)
         time = Time(f"{date_obs}T{timstart}",format='isot')-utcoffset
         time.format='fits'
         hdu.header['DATE-OBS']=(time.value,'[UTC] Start time of exposure')
         hdu1.header['DATE-OBS']=(time.value,'[UTC] Start time of exposure')
         rastr, decstr = self.get_target_radec()
         c=SkyCoord([f"{rastr} {decstr}"],frame='icrs',unit=(u.hourangle,u.deg))
         caltaz = c.transform_to(AltAz(obstime=time,location=obs_location))
         m = np.float32(caltaz.secz[0])
         airmass = self._compute_airmass(m)
         hdu.header['AIRMASS']=(airmass,'Airmass value')
         hdu1.header['AIRMASS']=(airmass,'Airmass value')

         # add other useful keywords to LIGHT frames
         hdu.header['OBJECT'] = (self.target[0],'Object Name')
         hdu1.header['OBJECT']= (self.target[0],'Object Name')
         hdu.writeto(f'{self.workdir}/LIGHT/light_V_{i+1}.fits',overwrite=True)
         hdu1.writeto(f'{self.workdir}/LIGHT/light_B_{i+1}.fits',overwrite=True)

   def _compute_airmass(self,m):
    """ 
    Return airmass 
    """
    return m - 0.0018167*(m-1) - 0.002875*(m-1)**2 - 0.0008083*(m-1)**3

   def _compute_gain_rnoise(self,filter):
    """
      Compute Gain and Read-out noise from Bias and Flat frames
    """
    logger.info("Computing GAIN and ReadOut Noise")
    biasfile1 = f"{self.workdir}/BIAS/bias_{filter}_1.fits"
    biasfile2 = f"{self.workdir}/BIAS/bias_{filter}_3.fits"
    flatfile1 = f"{self.workdir}/FLAT/flat_{filter}_1.fits"
    flatfile2 = f"{self.workdir}/FLAT/flat_{filter}_3.fits"

    bias1 = fits.getdata(biasfile1)
    bias2 = fits.getdata(biasfile2)
    flat1 = fits.getdata(flatfile1)
    flat2 = fits.getdata(flatfile2)

    mean_flat1 = np.median(flat1)
    mean_flat2 = np.median(flat2)
    mean_bias1 = np.median(bias1)
    mean_bias2 = np.median(bias2)

    _,_,std_biasdiff = sigma_clipped_stats(bias1-bias2,sigma=4.0,maxiters=2)
    _,_,std_flatdiff = sigma_clipped_stats(flat1-flat2,sigma=4.0,maxiters=2)
    gain = ((mean_flat1+mean_flat2) - (mean_bias1+mean_bias2))/((std_flatdiff**2 - std_biasdiff**2))
    rdnoise = gain *std_biasdiff/np.sqrt(2.)

    logger.info(f"Gain = {gain} - Read-Out Noise = {rdnoise}")
    return gain, rdnoise

   def get_target_radec(self):
     """
       Get target ra and dec from the provided json file
     """
     with open(self.target,'r') as f:
        info = json.load(f)

     ra = info[0][1]
     dec = info[0][2]
     
     return ra,dec
   
   def compute_allobject_photometry_image(self,filter):
    with open(self.apertures,'r') as f:
      list_aper = json.load(f)

    circle = list_aper[0]
    inner = list_aper[1]
    outer = list_aper[2]

    logger.info("Computing Aperture Photometry for all objects in the user provided list")
    imfiles = glob.glob(f"{self.workdir}/Solved/wcs*{filter}*.fits")
    imfiles.sort()

    phot = {}
    xmass = {}    
    
    for i,ifile in enumerate(imfiles):
       head = fits.getheader(ifile)
       t = Time(head['DATE-OBS'],format='isot',scale='utc')
       jd = t.jd
       t.format = 'fits'
       wcs = WCS(head)
       img = fits.getdata(ifile)
       for iobj in range(0,self.sources):
         skycoord = SkyCoord(f"{self.result[iobj]['ra']} {self.result[iobj]['dec']}",frame='icrs',unit=(u.hourangle,u.deg))

         # this is for computing airmass for the selected object - computed only once
         if (filter == 'V'):
           caltaz = skycoord.transform_to(AltAz(obstime=t,location=obs_location))
           m = np.float32(caltaz.secz)
           airmass = self._compute_airmass(m)
           xmass = {'jd': jd, 'name':self.result[iobj]['name'],'airmass': airmass}
           self.xmass.append(xmass)

         xy = SkyCoord.to_pixel(skycoord,wcs=wcs,origin=1)
         mag, magerr, flux, fluxerr, sky, skyerr, badflag, outstr = aper.aper(img,xy[0],xy[1],phpadu=self.gain,
                            apr=9,zeropoint=0.,skyrad=[12,16],exact=True) 
         phot = {'jd': jd, 'name':self.result[iobj]['name'],f'{filter}_ins': mag}
         self.data.append(phot)

   def _compute_mean_mag(self):
    """
     Compute mean of instrumental magnitudes for both V and B filters
    """
    nsources = self.sources
    ntimes = len(self.xmass)/nsources
    mean_V = np.zeros(nsources)
    mean_B = np.zeros(nsources)
    mean_X = np.zeros(nsources)

    self.target_compar = np.zeros([5,np.int32(ntimes)])
    for iobj in range(0,2):
      name = self.data[iobj]['name']
      i = 0
      for id in self.data:
        if id['name'] == name and ('V_ins' in id):
          self.target_compar[iobj][i] = np.float32(id.get('V_ins'))
          i = i + 1

    for iobj in range(0,nsources):
      name = self.data[iobj]['name']
      for id in self.data:
        if id['name'] == name:
          key = id.get('V_ins')
          if key is not None:
            mean_V[iobj] = mean_V[iobj] + id['V_ins']
          key = id.get('B_ins')
          if key is not None:
            mean_B[iobj] = mean_B[iobj] + id['B_ins']   

    for iobj in range(0,2):
      name = self.data[iobj]['name']
      i = 0
      for id in self.xmass:
        if id['name'] == name:
          self.target_compar[iobj+2][i] = np.float32(id.get('airmass'))
          i = i+1

    for iobj in range(0,nsources):
      name = self.xmass[iobj]['name']
      for id in self.xmass:
        if id['name'] == name:
          mean_X[iobj] = mean_X[iobj]+ id['airmass']

    return mean_V/ntimes, mean_B/ntimes, mean_X/ntimes

   def _extract_b_v(self):
    """
     extract B-V for all sources
    """
    nsources = self.sources
    B_V = np.zeros(nsources)
    V_cat = np.zeros(nsources)
    for iobj in range(0,nsources):
      name = self.result[iobj]['name']
      for id in self.result:
        if id['name'] == name:
          B_V[iobj] = id.get('B-V')
          V_cat[iobj] = id.get('V')
    return V_cat,B_V

   def get_final_estimation(self):
    """
     Get the final estimation of the transformed magnitudes for Target and Comparison
     Do this for all the observations acquired and then report mean + std of the results
    """
    V_Cat, B_V_Cat = self._extract_b_v()

    v_inst_target = self.target_compar[0][:]
    v_inst_compar = self.target_compar[1][:]
    airmass_target = self.target_compar[2][:]
    airmass_compar = self.target_compar[3][:]
    name = self.xmass[0]['name']
    for id in self.xmass:
      if id['name'] == name:
        self.jd.append((id.get('jd')))
    
    v_target = self.slope1 * self.target_bv + self.slope2 * airmass_target + v_inst_target + self.ZPoint1
    v_compar = self.slope1 * B_V_Cat[1] + self.slope2 * airmass_compar + v_inst_compar + self.ZPoint1

    logger.info("Report Target and Comparison star transformed magnitudes")
    logger.info("Target                      Compar")
    for i in range(0,len(v_target)):
      print (f"       {self.jd[i]}   {airmass_target[i]}   {v_target[i]}    {airmass_compar[i]}    {v_compar[i]}")

    self.JohnV_t = np.median(v_target)
    self.JohnV_v = np.median(v_compar)
    self.JohnV_t_std = 1.253*np.std(v_target)/np.sqrt(len(v_target))
    self.JohnV_v_std = 1.253*np.std(v_compar)/np.sqrt(len(v_compar))

    logger.info(f"Mean/Std values for jd {np.mean(self.jd)}")
    logger.info(f"Target: {np.median(v_target)} +/- {1.253*np.std(v_target)/np.sqrt(len(v_target))}")
    logger.info(f"Comparison: {np.median(v_compar)} +/- {1.253*np.std(v_compar)/np.sqrt(len(v_compar))}")


   def do_transf_extin(self):
    """ 
     Perform magnitude transformation and extinction
     In order to do linear fits compute the mean value of all the quantities (V_ins, B_ins and Xmass)
    """

    mean_V, mean_B, mean_X = self._compute_mean_mag()
    
    # the first two soures are target and comparison. These are not used for the
    # derivation of transformation

    V_Cat, B_V_Cat = self._extract_b_v()
    
    # estimate B-V of the target
    b_v = mean_B - mean_V
    y = np.array(B_V_Cat[1:])
    A = np.vstack([b_v[1:],np.ones(len(y))]).T
    b,a = np.linalg.lstsq(A,y,rcond=None)[0]
    self.target_bv = a + b*b_v[0]
    logger.info(f"Estimated Target Colour Index (B-V) = {self.target_bv}")

    # fit only transformation
    y_TX = np.array(V_Cat[2:] - mean_V[2:])
    x_TX = np.array(B_V_Cat[2:])
    A = np.vstack([x_TX,np.ones(len(x_TX))]).T
    self.slope,self.ZPoint = np.linalg.lstsq(A,y_TX,rcond=None)[0]
    logger.info(f"Results for Transformation only: slope = {self.slope} and ZeroPoint = {self.ZPoint}")

    # fit with extintion
    x_TX = np.array([B_V_Cat[2:],mean_X[2:]])
    A = x_TX.T
    A = np.c_[A,np.ones(A.shape[0])]
    self.slope1,self.slope2,self.ZPoint1 = np.linalg.lstsq(A,y_TX,rcond=None)[0]
    logger.info(f"Results for Transformation and Extintion : slope1 = {self.slope1}, slope2 ={self.slope2} and ZeroPoint = {self.ZPoint1}")

    # print validation check on the validation stars
    for i in range(2,len(V_Cat)):
      v = B_V_Cat[i]*self.slope1 + mean_X[i]*self.slope2 + self.ZPoint1 + mean_V[i]
      self.check_V.append(v)
      v_str = "{:.4f}".format(v)
      VC = "{:.4f}".format(V_Cat[i])
      DV = "{:.4f}".format(np.fabs(V_Cat[i] - v))
      logger.info(f"{self.data[i]['name']}: V_mag = {v_str}  Cat V_mag = {VC}  ABS = {DV}")

   def get_ra_dec_for_objects(self):
    """
    This function return as a list the reference stars within the observed field
    around the target star. Select at lest 10 stars that will be used to
    compute the transformation from TG to V knowing the color index of the 
    measured stars
    """

    with open(self.target,'r') as f:
      info = json.load(f)

    star_name = info[0][0]
    star_ra = info[0][1]
    star_dec = info[0][2]
    
    res_dict = {}
    tdict = {'name':star_name,'ra':star_ra,'dec':star_dec,'V': 0.0, 'B': 0.0, 'B-V': 0.0}
    self.result.append(tdict)
    logger.info(f"Get comparison/check stars for {star_name}")
    vsp_template = 'https://www.aavso.org/apps/vsp/api/chart/?format=json&fov=180&star={}&maglimit={}'
    query = vsp_template.format(star_name,self.maglim)
    record = requests.get(query).json()
    self.chartid = record['chartid']
    for id in self.list_obj:
      for item in record['photometry']:
        if item['auid'] == id:
          for key in item:
            if key == 'bands':
              for d in item[key]:
                if d['band'] == 'V':
                  magv = d['mag']
                if d['band'] == 'B':
                  magb = d['mag']
          res_dict = {'name': item['auid'],'ra':item['ra'],'dec':item['dec'],'V':magv,'B':magb,'B-V':np.float32("{:.3f}".format(magb-magv))}
      self.result.append(res_dict)

    self.sources = len(self.result)

   def show_apertures(self,filter):
     """
      Show one image with apertures on the target, comparison and
      check stars
     """
     cfiles = glob.glob(f"{self.workdir}/Solved/wcs*{filter}*.fits")
     cfiles.sort()
     data = fits.getdata(cfiles[0])
       
     if filter == 'V':
      idfilter = 0
     if filter == 'B':
       idfilter = 1
     
     pos_tar = [self.x_t[0,idfilter],self.y_t[0,idfilter]]
     pos_com = [self.x_c[0,idfilter],self.y_c[0,idfilter]]
     pos_che = [self.x_v[0,idfilter],self.y_v[0,idfilter]]

     #pos_tar=[(ix-1,iy-1) for ix,iy in zip(xt,yt)]
     #pos_com=[(ix-1,iy-1) for ix,iy in zip(xc,yc)]
     #pos_che=[(ix-1,iy-1) for ix,iy in zip(xv,yv)]
     aper_targ = pht.CircularAperture(pos_tar,r=10)
     aper_comp = pht.CircularAperture(pos_com,r=10)
     aper_chec = pht.CircularAperture(pos_che,r=10)
     plt.figure(figsize=(8,8))
     plt.imshow(data,cmap='Greys_r',origin='lower',vmin=160,vmax=250,interpolation='nearest')
     aper_targ.plot(color='red',lw=2,alpha=0.5)
     aper_comp.plot(color='cyan',lw=2,alpha=0.5)
     aper_chec.plot(color='yellow',lw=2,alpha=0.5)
     plt.title(f'red:target, cyan: comparison, yellow: validation - Filter {filter}')
     plt.show(block=False)

   def show_radial_profiles(self,filter):
     """
       Show radial profiles to check which aperture is the correct one
     """
     cfiles = glob.glob(f"{self.workdir}/Solved/wcs*{filter}*.fits")
     cfiles.sort()
     data = fits.getdata(cfiles[0])
     if filter == 'V':
      idfilter = 0
     if filter == 'B':
      idfilter = 1

     xycen_t = centroid_quadratic(data,xpeak=self.x_t[0,idfilter],ypeak=self.y_t[0,idfilter])
     xycen_c = centroid_quadratic(data,xpeak=self.x_c[0,idfilter],ypeak=self.y_c[0,idfilter])
     xycen_v = centroid_quadratic(data,xpeak=self.x_v[0,idfilter],ypeak=self.y_v[0,idfilter])
     edge_radii = np.arange(25)
     rp_t = RadialProfile(data,xycen_t,edge_radii,error=None,mask=None)
     rp_c = RadialProfile(data,xycen_c,edge_radii,error=None,mask=None)
     rp_v = RadialProfile(data,xycen_v,edge_radii,error=None,mask=None)

     plt.figure(figsize=(24,8))
     plt.plot(rp_t.radius,rp_t.profile/np.max(rp_t.profile),label='Target')
     plt.plot(rp_c.radius,rp_c.profile/np.max(rp_c.profile),label='Comparison')
     plt.plot(rp_v.radius,rp_v.profile/np.max(rp_v.profile),label='Validation')
     plt.xlim([0,25])
     plt.xlabel('Radius [pixels]',fontsize=20)
     plt.ylim([0,1.2])
     plt.ylabel(f'Normalize Radial Profile for filter {filter}',fontsize=20)
     plt.show(block=False)

   def plot_light_curve(self,iaper):
    """
     Plot the light curves of target, comparison and validation
    """
    rlc_targ = self.out_target[iaper+1,:]/self.out_compar[iaper+1,:]
    rlc_vali = self.out_valide[iaper+1,:]/self.out_compar[iaper+1,:]

    a1=1.0/self.out_compar[iaper+1,:]
    e1=self.out_target[iaper+self.naper+1,:]
    a2=self.out_target[iaper+1,:]/self.out_compar[iaper+1,:]**2
    e2=self.out_compar[iaper+self.naper+1,:]
    rlcerr_target=np.sqrt(a1**2*e1**2+a2**2*e2**2)
    e1=self.out_valide[iaper+self.naper+1,:]
    a2=self.out_valide[iaper+1,:]/self.out_compar[iaper+1,:]**2
    e2=self.out_compar[iaper+self.naper+1,:]
    rlcerr_validate=np.sqrt(a1**2*e1**2+a2**2*e2**2)

    logger.info(f"Photometric Error for target/comparison : {np.median(rlcerr_target)}")
    logger.info(f"Photometric Error for validation/comparison: {np.median(rlcerr_validate)}")

    norm_targ = np.median(rlc_targ)
    norm_vali = np.median(rlc_vali)
    plt.figure(figsize=(16,8))
    plt.plot(self.out_target[0,:],rlc_targ/norm_targ,'r.')
    plt.plot(self.out_target[0,:],rlc_vali/norm_vali-0.08,'b.')
    plt.xlabel('MJD',fontsize=20)
    plt.ylabel('Relative $m$',fontsize=20)
    plt.show(block=True)

   def aavso(self):
    """
    Create the report for AAVSO submittion of the observation
    """
    header_template = """
    #TYPE = EXTENDED
    #OBSCODE = {0}
    #SOFTWARE = {1}, Python Script - Tested against ASTAP
    #DELIM = ,
    #DATE = JD
    #OBSTYPE = DSLR
    #NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES
    """
    with open(self.target,'r') as f:
      info = json.load(f)

    result_template = "{0},{1:1.6f},{2:1.6f},{3:1.6f},{4},YES,STD,{5},{6:1.6f},{7},{8:1.6f},{9},NA,{10},{11}\n"
    with open(f"{self.workdir}/webobs_{info[0][0]}_{np.mean(self.jd)}.csv",'w') as webobs:
      webobs.write(header_template.format(obscode,"MYPhotometry"))
      webobs.write(result_template.format(info[0][0],np.mean(self.jd),self.JohnV_t,self.JohnV_t_std,"V",
                           info[1][0],self.JohnV_v,info[2][0],self.check_V[0],np.mean(self.target_compar[3][:]),self.chartid," "))

   
   
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
       rfiles = glob.glob(self.workdir+"/Reduced/p_light*.fits")
       rfiles.sort()
       self.filename = rfiles

       # since pre-processing is not performed get the GAIN and RDNOISE from
       # FITS keywords
       head = fits.getheader(rfiles[0])
       self.gain = head['GAIN']
       self.rdnoise = head['RDNOISE']

     # do astrometry
     if os.path.exists(f"{self.workdir}/Solved"):
      logger.info("Solved folder already exists")
     else:
      os.system(f"mkdir {self.workdir}/Solved")

     for i,ifile in enumerate(self.filename):
       logger.info(f"Get astrometric solution {i+1}/{len(self.filename)} - {ifile}")
       basename = os.path.basename(ifile)
       rootname,_ = os.path.splitext(basename)
    
       if os.path.exists(f"{self.workdir}/Solved/wcs_{rootname}.fits"):
        logger.info("Image already Solved - skip it") 
       else:  
         # it uses astap to create astrometric solution
         rastr, decstr = self.get_target_radec()
         ra = np.int16(rastr[:2])
         dec = 90 + np.int16(decstr[:3])

         #os.system(f"astap -f {ifile} -ra {ra} -spd {dec} - r 30 -fov 1.48 -o {self.workdir}/Reduced/test")
         os.system(f"solve-field --scale-units arcsecperpix --scale-low 3. --scale-high 3.5 {ifile} -D {self.workdir}/Solved --no-plots")
         os.system(f"mv {self.workdir}/Solved/{rootname}.new {self.workdir}/Solved/wcs_{rootname}.fits")
         os.system(f"rm {self.workdir}/Solved/*.corr {self.workdir}/Solved/*.axy {self.workdir}/Solved/*.match {self.workdir}/Solved/*.wcs")
         os.system(f"rm {self.workdir}/Solved/*xyls {self.workdir}/Solved/*.rdls {self.workdir}/Solved/*solved")