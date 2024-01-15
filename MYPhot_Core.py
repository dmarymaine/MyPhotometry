
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
     self.showplots = args.showplots
     self.gain = None
     self.rdnoise = None
     self.out_target = None
     self.out_compar = None
     self.out_valide = None
     self.x_t = 0.0
     self.y_t = 0.0
     self.x_c = 0.0
     self.y_c = 0.0
     self.x_v = 0.0
     self.y_v = 0.0

   def do_fits_preprocessing(self):
     """ Perform pre-processing aka bias,dark and flat correction
         of input data in FITS format """

     # check if a master bias is already present in the Reduce folder
     if os.path.isfile(self.workdir+"/Reduced/masterbias.fits"):
       logger.info("Master Bias already present - skipping master bias creation")
     else:
       logger.info("Create Master Bias")
       bfiles = glob.glob(self.workdir+"BIAS/bais*.fits")
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
       plt.show(block=False)
       
     # check if flat dark is present in the Reduce folder 
     if os.path.isfile(self.workdir+"/Reduced/masterdarkflat.fits"):
       logger.info("Master Dark Flat already present - skipping creation")
     else:
       logger.info("Create Master Dark Flat")
       dffiles = glob.glob(self.workdir+"DARKFLAT/DarkFlat*.fits")
       ddfiles.sort()
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
       plt.show(block=False)
     
     # now combinte the flats: subtract BIAS and DARKFLAT
     # normalize the frames and create the master flat
     if os.path.isfile(self.workdir+"/Reduced/masterflat.fits"):
       logger.info("Master Flat already exists - skipping creations")
     else:
       logger.info("Create Master Flat")
       ffiles = glob.glob(self.workdir+"/FLAT/Flat*fits")
       ffiles.sort()
       allflats = []

       masterbias = fits.getdata(self.workdir+"/Reduced/masterbias.fits")
       masterdarkflat = fits.getdata(self.workdir+"/Reduced/masterdarkflat.fits")

       for i,ifiles in enumerate(ffiles):
         logger.info(f"reading flat: {i+1}/{len(ffiles)} - {ifile}")
         data = fits.getdata(ifile)- masterbias - masterdarkflat
         mflat = np.median(data)
         # normalize flat frame
         data/=mflat
         logger.info(f"median flat: {mflat}")
         allflat.append(data)
       
       allflat = np.stack(allflat)
       masterflat=np.median(allflat,axis=0)
       fits.writeto(self.workdir+"/Reduced/masterflat.fits",masterflat.astype('float32'),overwrite=True)
      
     if showplots:
       tvflats = fits.getdata(self.workdir+"/Reduced/masterflat.fits")
       plt.figure(figsize=(8,8))
       plt.imshow(tvflats)
       plt.colorbar()
       plt.title("Master Flat from Flat frames")
       plt.show(block=False)

     # now get the Light frames and calibrate them with bias, dark and flats
     # also add keywords for auxiliari information
     # it takes the target (ra,dec) and set it to CRVAL1/CRVAL2
     # it computes gain and read-out noise from bias and flats

     if os.path.isfile(self.workdir+"/Reduced/light*calib*fits"):
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
         data = (data - masterbias - masterdark)/masterflat
         head['epoch'] = 2000.0
         head['CRVAL1'] = ra
         head['CRVAL2'] = dec
         head['CRPIX1'] = head['NAXIS1']/2.0
         head['CRPIX2'] = head['NAXIS2']/2.0
         head['GAIN'] = (gain,'GAIN in e-/ADU')
         head['RDNOISE'] = (rdnoise,'read out noise in electron')

         fits.writeto(outfile,data,header=head,overwrite=True)

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

         hdu.writeto(f'{self.workdir}/DARKFLAT/darkflat_{i+1}.fits',overwrite=True)

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

         hdu.writeto(f'{self.workdir}/FLAT/flat_{i+1}.fits',overwrite=True)

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

         # add other useful keywords to LIGHT frames
         hdu.header['OBJECT'] = (self.target[0],'Object Name')
         hdu.writeto(f'{self.workdir}/LIGHT/light_{i+1}.fits',overwrite=True)

   def _compute_gain_rnoise(self):
    """
      Compute Gain and Read-out noise from Bias and Flat frames
    """
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
    logger.info("Computing Aperture Photometry for all the objects")
    cfiles = glob.glob(f"self.{workdir}/Solved/*.fits")
    cfiles.sort()

    with open(self.apertures,'r') as f:
      aper_radii = json.load(f)

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
      daofind = pht.IRAFStarFinder(fwhm=3.0,threshold=5.*bkg.background_rms_median,exclude_border=True,
                plo=0.5,sharphi=2.0,roundlo=0.0,roundhi=0.7)
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

     t_coord = SkyCoord([f"{info[0][1]} {info[0][2]}"],frame='fk5',unit=(u.hourangle,u.deg))
     c_coord = SkyCoord([f"{info[1][1]} {info[1][2]}"],frame='fk5',unit=(u.hourangle,u.deg))
     v_coord = SkyCoord([f"{info[2][1]} {info[2][2]}"],frame='fk5',unit=(u.hourangle,u.deg))

     with open(self.apertures,'r') as g:
       aper_radii = json.load(g)

     naper = len(aper_radii)
     out_target = np.zeros((1+2*naper,len(catfiles)))
     out_compar = np.zeros((1+2*naper,len(catfiles)))
     out_valide = np.zeros((1+2*naper,len(catfiles)))

     # 
     # now loop over the catalogs, read WCS from calibrated frames and get required stars
     for i,ifile in enumerate(catfiles):
        logger.info(f"reading catfiles {i+1}/{len(catfiles)} - {ifile}")
        rootame,_ = os.path.splitext(ifile)
        #remove the last 4 spaces from file name to get the calibrated frame
        sciframe = rootname[:-4]
        # read sci header, create WCS and get (x,y) for T,C and V stars
        logger.info("creating WCS for the selected catalog")
        head = fits.getheader(sciframe+".fits")
        w = WCS(head)
        self.x_t,self.y_t = w.world_to_pixel(t_coord)
        self.x_c,self.y_c = w.world_to_pixel(c_coord)
        self.x_v,self.y_v = w.world_to_pixel(v_coord)

        head_cat = fits.getheader(ifile)
        datestr = head['DATE-OBS']
        t = Time(datestr,format='isot',scale='utc')
        jd = t.mjd
        out_target[0,i] = jd
        out_compar[0,i] = jd
        out_valide[0,i] = jd

        # now find the nearest to T,C and V star in catalog
        # Target
        cat = fits.getdata(ifile)
        x = cat['xcenter']
        y = cat['ycenter']
        d = np.sqrt((x-self.x_t)**2 + (y-self.y_t)**2)
        idx = np.argmin(d)
        icat=cat[idx]
        dt = d[idx]
        if d[idx]<2:
          for j in range(naper):
            out_target[j+1,i]=icat['aperture_sum_'+str(j)]
            out_target[naper+j+1,i]=icat['aperture_sum_err_'+str(j)]
        else:
          out_target[1:,i]=np.nan
        #
        # Comparison
        d = np.sqrt((x-self.x_c)**2 + (y-self.y_c)**2)
        idx = np.argmin(d)
        dc = d[idx]
        if d[idx]<2:
          for j in range(naper):
            out_compar[j+1,i]=icat['aperture_sum_'+str(j)]
            out_compar[naper+j+1,i]=icat['aperture_sum_err_'+str(j)]
        else:
          out_compar[1:,i]=np.nan
        #  
        #  Validation
        d = np.sqrt((x-self.x_v)**2+(y-self.y_v)**2)
        idx = np.argmin(d)
        icat = cat[idx]
        dv = d[idx]
        if d[idx] < 2:
          for j in range(naper):
            out_valide[j+1,i]=icat['aperture_sum_'+str(j)]
            out_valide[naper+j+1,i]=icat['aperture_sum_err_'+str(j)]
        else:
          out_valide[1:,i] = np.nan

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
       rfiles = glob.glob(args.workdir+"Reduced/plight*.fits")
       rfiles.sort()
       self.filename = rfiles

     # do astrometric solution
     for i,ifile in enumerate(self.filename):
       logger.info(f"Get astrometric solution {i+1}/{len(self.filename)} - {ifile}")
       # it uses astap to create astrometric solution
       rastr, decstr = self._get_target_radec()
       ra = np.int16(rastr[:2])
       dec = 90 + np.int16(decstr[:3])

       indir,infile = os.path.split(ifile)
       rootname,_ = os.path.splitext(infile)

       os.system(f"astap -f {ifile} -ra {ra} -spd {dec} - r 30 -o {self.workdir}/Reduced/test")
       head_wcs = fits.getheader(f"{self.workdir}/Reduced/test.wcs")
       data = fits.getdata(ifile)
       fits.writeto(f"{self.workdir}/Solved/{rootname}_wcs.fits",data,header=head_wcs,overwrite=True)
