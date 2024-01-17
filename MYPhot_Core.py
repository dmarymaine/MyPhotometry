
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
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.utils import calc_total_error
from photutils.centroids import centroid_quadratic
from photutils.profiles import RadialProfile
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from datetime import datetime
matplotlib.use('TkAgg')

import MYPhot_Logging as log

logger = log.getLogger("MYPhot_Core")

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
     self.x_t = []
     self.y_t = []
     self.x_c = []
     self.y_c = []
     self.x_v = []
     self.y_v = []


   def do_fits_preprocessing(self):
     """ Perform pre-processing aka bias,dark and flat correction
         of input data in FITS format """

     # check if a master bias is already present in the Reduce folder     
     if os.path.isfile(self.workdir+"/Reduced/masterbias.fits"):
       logger.info("Master Bias already present - skipping master bias creation")
     else:
       os.system(f"mkdir {self.workdir}/Reduced")
       logger.info("Create Master Bias")
       bfiles = glob.glob(self.workdir+"/BIAS/bias*.fits")
       bfiles.sort()
       allbias = []
       for i,ifile in enumerate(bfiles):
         logger.info(f"reading bias: {i+1}/{len(bfiles)} - {ifile}")
         data = fits.getdata(ifile)
         allbias.append(data)

       # stack bias together
       allbias = np.stack(allbias)
       superbias = np.median(allbias,axis=0)
       fits.writeto(self.workdir+"/Reduced/masterbias.fits",superbias.astype('float32'),overwrite=True)

     if self.showplots:
       tvbias = fits.getdata(self.workdir+"/Reduced/masterbias.fits")
       plt.figure(figsize=(8,8))
       plt.imshow(tvbias,origin='lower')
       plt.colorbar()
       plt.title("Master Bias derived from bias frames")
       plt.show(block=True)
       
     # check if flat dark is present in the Reduce folder 
     if os.path.isfile(self.workdir+"/Reduced/masterdarkflat.fits"):
       logger.info("Master Dark Flat already present - skipping creation")
     else:
       logger.info("Create Master Dark Flat")
       dffiles = glob.glob(self.workdir+"/DARKFLAT/darkflat*.fits")
       dffiles.sort()
       alldarkflats = []
       for i,ifile in enumerate(dffiles):
         logger.info(f"reading dark-flat: {i+1}/{len(dffiles)} - {ifile}")
         data = fits.getdata(ifile)
         alldarkflats.append(data)

       # stack all dark flats together
       alldarkflats = np.stack(alldarkflats)
       mdarkflat = np.median(alldarkflats,axis=0)
       fits.writeto(self.workdir+"/Reduced/masterdarkflat.fits",mdarkflat.astype('float32'),overwrite=True)
           
     if self.showplots:
       tvdflats = fits.getdata(self.workdir+"/Reduced/masterdarkflat.fits")
       plt.figure(figsize=(8,8))
       plt.imshow(tvdflats,origin='lower')
       plt.colorbar()
       plt.title("Master Dark-Flat from dark-flat frames")
       plt.show(block=True)
     
     # now combinte the flats: subtract BIAS and DARKFLAT
     # normalize the frames and create the master flat
     if os.path.isfile(self.workdir+"/Reduced/masterflat.fits"):
       logger.info("Master Flat already exists - skipping creations")
     else:
       logger.info("Create Master Flat")
       ffiles = glob.glob(self.workdir+"/FLAT/flat*fits")
       ffiles.sort()
       allflats = []

       masterbias = fits.getdata(self.workdir+"/Reduced/masterbias.fits")
       masterdarkflat = fits.getdata(self.workdir+"/Reduced/masterdarkflat.fits")

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
       fits.writeto(self.workdir+"/Reduced/masterflat.fits",masterflat.astype('float32'),overwrite=True)
      
     if self.showplots:
       tvflats = fits.getdata(self.workdir+"/Reduced/masterflat.fits")
       plt.figure(figsize=(8,8))
       plt.imshow(tvflats)
       plt.colorbar()
       plt.title("Master Flat from Flat frames")
       plt.show(block=True)
     
     # now combine the dark frames to create the master dark
     if os.path.isfile(self.workdir+"/Reduced/masterdark.fits"):
       logger.info("Master Dark already present - skipping creation")
     else:
       logger.info("Creating Master Dark")
       dfiles = glob.glob(self.workdir+"/DARK/dark*.fits")
       dfiles.sort()
       alldarks = []

       masterbias = fits.getdata(self.workdir+"/Reduced/masterbias.fits")

       for i,ifile in enumerate(dfiles):
         logger.info(f"reading dark: {i+1}/{len(dfiles)} - {ifile}")
         data = fits.getdata(ifile) - masterbias
         alldarks.append(data)
       
       alldarks = np.stack(alldarks)
       masterdark = np.median(alldarks,axis=0)
       fits.writeto(self.workdir+"/Reduced/masterdark.fits",masterdark.astype('float32'),overwrite=True)

     if self.showplots:
       tvdark = fits.getdata(self.workdir+'/Reduced/masterdark.fits')
       plt.figure(figsize=(8,8))
       plt.imshow(tvdark)
       plt.colorbar()
       plt.title("Master Dark from Dark frames")
       plt.show(block=True)

     # now get the Light frames and calibrate them with bias, dark and flats
     # also add keywords for auxiliari information
     # it takes the target (ra,dec) and set it to CRVAL1/CRVAL2
     # it computes gain and read-out noise from bias and flats

     if os.path.isfile(self.workdir+"/Reduced/p_light*fits"):
       logger.info("Light frames already calibrated - skipping light calibration")
     else:
       logger.info("Create calibrated Light frames")
       lfiles = glob.glob(self.workdir+"/LIGHT/light*.fits")
       lfiles.sort()
       
       # compute gain and read-out noise
       self.gain,self.rdnoise = self._compute_gain_rnoise()
       
       ra,dec = self.get_target_radec()

       masterbias = fits.getdata(self.workdir+"/Reduced/masterbias.fits")
       masterflat = fits.getdata(self.workdir+"/Reduced/masterflat.fits")
       masterdark = fits.getdata(self.workdir+"/Reduced/masterdark.fits")

       for i,ifile in enumerate(lfiles):
         logger.info(f"reducing (debias,dark sub and flat-field) light: {i+1}/{len(lfiles)} - {ifile}")
         indir, infile = os.path.split(ifile)
         rootname,_ = os.path.splitext(infile)
         outfile = os.path.join(self.workdir+f"/Reduced/p_{rootname}.fits")
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
         head['GAIN'] = (self.gain,'GAIN in e-/ADU')
         head['RDNOISE'] = (self.rdnoise,'read out noise in electron')

         fits.writeto(outfile,calib_data.astype('float32'),header=head,overwrite=True)
         self.filename.append(outfile)

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

       for i,ifile in enumerate(bfiles):
         logger.info(f"reading bias: {i+1} of {len(bfiles)} - {ifile}")
         raw = rawpy.imread(ifile)

         # get the two green channels
         g1 = raw.raw_image[1::2,::2]
         g2 = raw.raw_image[::2,1::2]

         # average the two channels and remove CANON offset
         green = 0.5*(g1+g2) - 2047

         # prepare to write fits
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[0] == 'Exif.Photo.ExposureTime'):
             first,second=it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/BIAS/bias_{i+1}.fits',overwrite=True)

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
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/DARK/dark_{i+1}.fits',overwrite=True)

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
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/DARKFLAT/darkflat_{i+1}.fits',overwrite=True)

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

         green = 0.5*(g1+g2) - 2047

         # prepare to write fits
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         for it in items:
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)

         hdu.writeto(f'{self.workdir}/FLAT/flat_{i+1}.fits',overwrite=True)

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

         green = 0.5*(g1+g2) - 2047

         # prepare to write fits
         image=pyexiv2.Image(ifile)
         exif = image.read_exif()
         items = exif.items()
         hdu = fits.PrimaryHDU(green.astype('float32'))
         utcoffset = 2.*u.hour
         for it in items:
           if (it[0] == 'Exif.Image.DateTime'):
            b=datetime.strptime(it[1],'%Y:%m:%d %H:%M:%S')
            date_obs=f'{b.year}-{b.month}-{b.day}'
            timstart=f'{b.hour}:{b.minute}:{b.second}'
           if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
             hdu.header['ISO']=it[1]
           if (it[1] == 'Exif.Photo.ExposureTime'):
             first,second = it[1].split("/")
             hdu.header['EXPTIME']=np.float32(first)/np.float32(second)
         obs_location = EarthLocation(lat=45.53*u.deg,lon=9.4*u.deg,height=133*u.m)
         time = Time(f"{date_obs}T{timstart}",format='isot')-utcoffset
         time.format='fits'
         hdu.header['DATE-OBS']=(time.value,'[UTC] Start time of exposure')
         rastr, decstr = self.get_target_radec()
         c=SkyCoord([f"{rastr} {decstr}"],frame='icrs',unit=(u.hourangle,u.deg))
         caltaz = c.transform_to(AltAz(obstime=time,location=obs_location))
         hdu.header['AIRMASS']=(np.float32(caltaz.secz[0]),'Airmass value')

         # add other useful keywords to LIGHT frames
         hdu.header['OBJECT'] = (self.target[0],'Object Name')
         hdu.writeto(f'{self.workdir}/LIGHT/light_{i+1}.fits',overwrite=True)

   def _compute_gain_rnoise(self):
    """
      Compute Gain and Read-out noise from Bias and Flat frames
    """
    logger.info("Computing GAIN and ReadOut Noise")
    biasfile1 = f"{self.workdir}/BIAS/bias_1.fits"
    biasfile2 = f"{self.workdir}/BIAS/bias_3.fits"
    flatfile1 = f"{self.workdir}/FLAT/flat_1.fits"
    flatfile2 = f"{self.workdir}/FLAT/flat_3.fits"

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
   
   def compute_allobject_photometry(self):
    """ 
      Compute all objects photometry. For each calibrated frames
      - detector sources
      - compute background and errors
      - defined apertures
      - get photometry for all objects 
      - save results to a catalog
    """
    with open(self.apertures,'r') as f:
      aper_radii = json.load(f)

    self.naper = len(aper_radii)

    checkfiles = glob.glob(f"{self.workdir}/Solved/*-cat.fits")
    if len(checkfiles) > 0:
      logger.info("Photometry already computed  - skipping catalog creation")
    else:
      logger.info("Computing Aperture Photometry for all the objects")
      cfiles = glob.glob(f"{self.workdir}/Solved/*wcs.fits")
      cfiles.sort()

      for i,ifile in enumerate(cfiles):
        logger.info(f"aperture photometry for {i+1}/{len(cfiles)} - {ifile}")
        rootname,_ = os.path.splitext(ifile)
        catfile = rootname+'-cat.fits'
        data = fits.getdata(ifile)

        # mask to get background estimation
        sigma_clip = SigmaClip(sigma=3.)
        mask = pht.make_source_mask(data,nsigma=3,npixels=5,dilate_size=11)
        bkg_estimator = pht.SExtractorBackground()
        bkg = pht.Background2D(data,(64,64),mask=mask,filter_size=(3,3),sigma_clip=sigma_clip,
              bkg_estimator=bkg_estimator)

        if (self.showplots and i == 1):
          f,axs = plt.subplots(1,2,figsize=(16,8))
          axs[0].imshow(bkg.background,origin='lower')
          axs[0].set_title("background")
          axs[1].imshow(bkg.background_rms,origin='lower')
          axs[1].set_title("background rms")
          plt.show(block=True)

        daofind = pht.IRAFStarFinder(fwhm=3.0,threshold=5.*bkg.background_rms_median,exclude_border=True,
                  sharplo=0.5,sharphi=2.0,roundlo=0.0,roundhi=0.7)
        sources = daofind(data - bkg.background)
        positions = [(ix,iy) for ix,iy in zip(sources['xcentroid'],sources['ycentroid'])]
        apert = [pht.CircularAperture(positions,r=r) for r in aper_radii]
        error = calc_total_error(data-bkg.background,bkg.background_rms,self.gain)
        aper_phot = pht.aperture_photometry(data - bkg.background,apert,error=error)
      
        aper_phot.write(catfile,overwrite=True)

   def get_target_comp_valid_photometry(self):
     """
     Get photometry for the target, comparison and validation stars
     It uses the catalog produced in comput_allobject_photometry
     and the WCS from the solved light frames.
     From WCS and T,C and V stars coordinates it gets the (x,y) on the frame
     and select the nearest object in the catalog
     It returns arrays for T,C and V with JD and photometry results
     """

     catfiles = glob.glob(f"{self.workdir}/Solved/*-cat.fits")
     catfiles.sort()

     with open(self.target,'r') as f:
        info = json.load(f)
    
     target_name = info[0][0]
     compar_name = info[1][0]
     valide_name = info[2][0]

     t_coord = SkyCoord([f"{info[0][1]} {info[0][2]}"],frame='icrs',unit=(u.hourangle,u.deg))
     c_coord = SkyCoord([f"{info[1][1]} {info[1][2]}"],frame='icrs',unit=(u.hourangle,u.deg))
     v_coord = SkyCoord([f"{info[2][1]} {info[2][2]}"],frame='icrs',unit=(u.hourangle,u.deg))

     logger.info("Selected objects with coordinates:")
     logger.info(f"Target {target_name} = {t_coord}")
     logger.info(f"Comparison {compar_name} = {c_coord}")
     logger.info(f"Validation {valide_name} = {v_coord}")

     with open(self.apertures,'r') as g:
       aper_radii = json.load(g)

     naper = len(aper_radii)
     self.out_target = np.zeros((1+2*naper,len(catfiles)))
     self.out_compar = np.zeros((1+2*naper,len(catfiles)))
     self.out_valide = np.zeros((1+2*naper,len(catfiles)))

     # 
     # now loop over the catalogs, read WCS from calibrated frames and get required stars
     for i,ifile in enumerate(catfiles):
        logger.info(f"reading catfiles {i+1}/{len(catfiles)} - {ifile}")
        rootname,_ = os.path.splitext(ifile)
        #remove the last 4 spaces from file name to get the calibrated frame
        sciframe = rootname[:-4]

        # read sci header, create WCS and get (x,y) for T,C and V stars
        logger.info("creating WCS for the selected catalog")
        head = fits.getheader(sciframe+".fits")
        w = WCS(head)
        x,y = w.world_to_pixel(t_coord)
        self.x_t.append(x)
        self.y_t.append(y)
        x,y = w.world_to_pixel(c_coord)
        self.x_c.append(x)
        self.y_c.append(y)
        x,y= w.world_to_pixel(v_coord)
        self.x_v.append(x)
        self.y_v.append(y)
        
        head_cat = fits.getheader(ifile)
        datestr = head['DATE-OBS']
        t = Time(datestr,format='isot',scale='utc')
        jd = t.mjd
        self.out_target[0,i] = jd
        self.out_compar[0,i] = jd
        self.out_valide[0,i] = jd

        # now find the nearest to T,C and V star in catalog
        # Target
        cat = fits.getdata(ifile)
        x = cat['xcenter']
        y = cat['ycenter']
        d_t = np.sqrt((x-self.x_t[i])**2 + (y-self.y_t[i])**2)
        idx = np.argmin(d_t)
        icat=cat[idx]
        dt = d_t[idx]
        if d_t[idx]<2:
          for j in range(naper):
            self.out_target[j+1,i]=icat['aperture_sum_'+str(j)]
            self.out_target[naper+j+1,i]=icat['aperture_sum_err_'+str(j)]
        else:
          self.out_target[1:,i]=np.nan
        #
        # Comparison
        d_c = np.sqrt((x-self.x_c[i])**2 + (y-self.y_c[i])**2)
        idx = np.argmin(d_c)
        icat=cat[idx]
        dc = d_c[idx]
        if d_c[idx]<2:
          for j in range(naper):
            self.out_compar[j+1,i]=icat['aperture_sum_'+str(j)]
            self.out_compar[naper+j+1,i]=icat['aperture_sum_err_'+str(j)]
        else:
          self.out_compar[1:,i]=np.nan
        #  
        #  Validation
        d_v = np.sqrt((x-self.x_v[i])**2+(y-self.y_v[i])**2)
        idx = np.argmin(d_v)
        icat = cat[idx]
        dv = d_v[idx]
        if d_v[idx] < 2:
          for j in range(naper):
            self.out_valide[j+1,i]=icat['aperture_sum_'+str(j)]
            self.out_valide[naper+j+1,i]=icat['aperture_sum_err_'+str(j)]
        else:
          self.out_valide[1:,i] = np.nan


   def show_apertures(self):
     """
      Show one image with apertures on the target, comparison and
      check stars
     """
     cfiles = glob.glob(f"{self.workdir}/Solved/p*wcs.fits")
     cfiles.sort()
     data = fits.getdata(cfiles[0])

     pos_tar=[(ix-1,iy-1) for ix,iy in zip(self.x_t[0],self.y_t[0])]
     pos_com=[(ix-1,iy-1) for ix,iy in zip(self.x_c[0],self.y_c[0])]
     pos_che=[(ix-1,iy-1) for ix,iy in zip(self.x_v[0],self.y_v[0])]
     aper_targ = pht.CircularAperture(pos_tar,r=10)
     aper_comp = pht.CircularAperture(pos_com,r=10)
     aper_chec = pht.CircularAperture(pos_che,r=10)
     plt.figure(figsize=(8,8))
     plt.imshow(data,cmap='Greys_r',origin='lower',vmin=160,vmax=250,interpolation='nearest')
     aper_targ.plot(color='red',lw=2,alpha=0.5)
     aper_comp.plot(color='cyan',lw=2,alpha=0.5)
     aper_chec.plot(color='yellow',lw=2,alpha=0.5)
     plt.title('red:target, cyan: comparison, yellow: validation')
     plt.show(block=False)

   def show_radial_profiles(self):
     """
       Show radial profiles to check which aperture is the correct one
     """
     cfiles = glob.glob(f"{self.workdir}/Solved/p_*wcs.fits")
     cfiles.sort()
     data = fits.getdata(cfiles[0])
     xycen_t = centroid_quadratic(data,xpeak=self.x_t[0],ypeak=self.y_t[0])
     xycen_c = centroid_quadratic(data,xpeak=self.x_c[0],ypeak=self.y_c[0])
     xycen_v = centroid_quadratic(data,xpeak=self.x_v[0],ypeak=self.y_v[0])
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
     plt.ylabel('Normalize Radial Profile',fontsize=20)
     plt.show(block=False)

   def plot_light_curve(self,iaper):
    """
     Plot the light curves of target, comparison and validation
    """
    print (self.out_target[iaper+1,:],self.out_compar[iaper+1,:])
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

     if os.path.exists(f"{self.workdir}/Solved/p_light_1_wcs.fits"):
       logger.info("Images already Plate Solved - go to photometry")
     else:
       # do astrometric solution
       for i,ifile in enumerate(self.filename):
         logger.info(f"Get astrometric solution {i+1}/{len(self.filename)} - {ifile}")
         # it uses astap to create astrometric solution
         rastr, decstr = self.get_target_radec()
         ra = np.int16(rastr[:2])
         dec = 90 + np.int16(decstr[:3])

         indir,infile = os.path.split(ifile)
         rootname,_ = os.path.splitext(infile)

         if os.path.exists(f"{self.workdir}/Solved"):
          logger.info("Solved folder already exists - skipping creation")
         else:
          os.system(f"mkdir {self.workdir}/Solved")

         #os.system(f"astap -f {ifile} -ra {ra} -spd {dec} - r 30 -fov 1.48 -o {self.workdir}/Reduced/test")
         os.system(f"solve-field --scale-units arcsecperpix --scale-low 3. --scale-high 3.5 {ifile} -D {self.workdir}/Solved --no-plots")
         os.system(f"mv {self.workdir}/Solved/{rootname}.new {self.workdir}/Solved/{rootname}_wcs.fits")