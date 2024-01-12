import numpy as np
import photutils as pht
import matplotlib.pyplot as plt
import os
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

"""
  This code performes a very basic evaluation variable stars mag using 
  aperture photometry.
  It supposes to have in input raw data or already pre-processed data 
  (i.e dark subtracted and flat corrected).
  Input data could be either in RAW CR2 format or already in FITS format
  It performs astrometry solution using astromty.net and then uses astropy.WCS
  to go from (RA,Dec) to (x,y).

  Input data
   -- workdir (string)
	base dir where to find data. It assumes to have raw data in 
        (LIGHT,DARK,FLAT,DARKFLAT) folders and/or already pre-processed data
        in Reduced

   -- pre-processing (bool) 
	whether to perform pre-processing (assumes the previous dir structures)
	it creates master darks and master flat dark corrected in store them
        into the Reduced folder. Together with reduce data in FITS format with
	header enrighed by raw data metadata

   -- plot_sprofile (bool)
	whether to compute and plots the target star profile

  Output data
   -- outfile (string)
	output file with photometry computed for target, comparison and third bonus
	star for different apertures
"""

class MyPhotometry(file):
   """ Photometry main class
   """
   def __init__(self):

     self.input_json = file
     self.input_params = {}
     with open(self.input_json,'r') as f:
        self.input_params = json.load(f)

     self.workdir = self.input_params[0]
     self.name = self.input_params[1]
     # ra and dec in string format
     self.ra = self.input_params[2]
     self.dec = self.input_params[3]

def compure_radial_profile(img,xpeak,ypeak):
    """
     function to compute and plot the radial profile
     of a selected source.

     Input:
      img: 2D image data
      xpeak: x position of the source peak
      ypeak: y position of the source peak

     Output:
      profile: an RadialProfile object 
    """
    xycen = centroid_quadratic(img,xpeak,ypeak)
    edge_radii = np.arange(25)
    rp = RadialProfile(img,xycen,edge_radii,error=error,mask=None)

    # plot the radial profile
    plt.figure(figsize=(8,8))
    plt.xlabel(r'Radius [pixel]',fontsize=12)
    plt.ylabel(r'Profile',fontsize=12)
    plt.plot(rp.radius,rp.profile) 
    plt.show(block=False)

    # compute a gaussian fit to the profile and extract FWHM
    rp.gaussian_fit
    print (f"Selected star FWHM [pix]: {rp.gaussian_fwhm}")

    return rp

def compute_pre_processing(workdir):
    """
     perform master dark and master flat dark corrected and store them.
     create reduce science data
    """
    # check if Reduced directory exists
    if os.path.exists(workdir+"Reduced"):
      print("directory already exists")
    else:
      os.system(f"mkdir {workdir}/Reduced") 

    # star with dark flat master creation
    print ("combining darks flats...")
    dfiles = glob.glob(workdir+"DARKFLAT/*.cr2")
    dfiles.sort()
    alldarks = []

    for i,ifile in enumerate(dfiles):
      print ("reading flat darks:",i+1,len(dfiles),ifile)
      raw=rawpy.imread(ifile)

      # now get the two green channels
      g1=raw.raw_image[1::2,::2]
      g2=raw.raw_image[::2,1::2]
      green = 0.5*(g1+g2)
      alldarks.append(green-2047)

    alldarks = np.stack(alldarks)
    print (alldarks.shape)

    # create darkflat and correct for bias
    darkflat=np.median(alldarks,axis=0)

    print ("combining flats ...")
    ffiles = glob.glob(workdir+"FLAT/*cr2")
    ffiles.sort()
    allflats = []

    for i,ifile in enumerate(ffiles):
      print ("reading flats:",i+1,len(ffiles),ifile)
      raw=rawpy.imread(ifile)
      g1=raw.raw_image[1::2,::2]
      g2=raw.raw_image[::2,1::2]
      green = 0.5*(g1+g2)
      allflats.append(green-2047)

    allflats = np.stack(allflats)
    print (allflats.shape)

    # create calibrated flat master
    flat=allflats-darkflat
    flat = flat/np.median(flat)
    flat = np.median(flat,axis=0)

    # new store the master flat into FITS file
    image=pyexiv2.Image(ffiles[0])
    exif=image.read_exif()
    items = exif.items()
    hdu = fits.PrimaryHDU(flat)
    for it in items:
      if (it[0] == 'Exif.Photo.ISOSpeedRatings'):
        hdu.header['GAIN'] = it[1]
      if (it[0] == 'Exif.Photo.ExposureTime'):
        first,second=it[1].split("/")
        hdu.header['EXPTIME'] = np.float32(first)/np.float32(second)
      if (it[0] == 'Exif.Image.Make'):
        hdu.header['CAMMAKER'] = it[1]
      if (it[0] == 'Exif.Image.Model'):
        hdu.header['INSTRUM'] = it[1]

    hdu.writeto(workdir+'/Reduced/masterflat.fits',overwrite=True)

    # now create master dark
    dfiles = glob.glob(workdir+"DARK/*.cr2")
    dfiles.sort()
    alldarks = []

    for i,ifile in enumerate(dfiles):
      print ("reading darks:",i+1,len(dfiles),ifile)
      raw=rawpy.imread(ifile)

      # now get the two green channels
      g1=raw.raw_image[1::2,::2]
      g2=raw.raw_image[::2,1::2]
      green = 0.5*(g1+g2)
      alldarks.append(green-2047)

    alldarks = np.stack(alldarks)
    print (alldarks.shape)

    # create darkflat and correct for bias
    dark=np.median(alldarks,axis=0)

    image=pyexiv2.Image(dfiles[0])
    exif = image.read_exif()
    items = exif.items()
    hdu = fits.PrimaryHDU(dark)
    for it in items:
      if (it[0] == 'Exif.Photo.ISOSPeedRatings'):
        hdu.header['GAIN']=it[1]
      if (it[0] == 'Exif.Photo.ExposureTime'):
        first,second=it[1].split("/")
        hdu.header['EXPTIME'] = np.float32(first)/np.float32(second)
      if (it[0] == 'Exif.Image.Make'):
        hdu.header['CAMMAKER'] = it[1]
      if (it[0] == 'Exif.Image.Model'):
        hdu.header['INSTRUM'] = it[1]

    hdu.writeto(workdir+'/Reduced/masterdark.fits',overwrite=True) 


def compute_gain_rnoise(workdir):
    """
     compute gain (e-/ADU) and read-out noise from
     flat and bias. If Bias are not present skip it
    """
    if os.path.exists(workdir+"/BIAS"):
     bfiles = glob.glob(workdir+"/BIAS/*cr2")
     bfiles.sort()

     biasfile1=bfiles[0]
     biasfile2=bfiles[1]

     ffiles = glob.glob(workdir+"/FLAT/*cr2")
     ffiles.sort()

     flatfile1=ffiles[0]
     flatfile2=ffiles[1]

     raw=rawpy.imread(biasfile1)
     g1=raw.raw_image[1::2,::2]
     g2=raw.raw_image[::2,1::2]

     bias1 = 0.5*(g1+g2)-2047

     raw=rawpy.imread(biasfile2)
     g1=raw.raw_image[1::2,::2]
     g2=raw.raw_image[::2,1::2]

     bias2 = 0.5*(g1+g2)-2047

     raw=rawpy.imread(flatfile1)
     g1=raw.raw_image[1::2,::2]
     g2=raw.raw_image[::2,1::2]

     flat1=0.5*(g1+g2)-2047

     raw=rawpy.imread(flatfile2)
     g1=raw.raw_image[1::2,::2]
     g2=raw.raw_image[::2,1::2]

     flat2=0.5*(g1+g2)-2047

     mean_bias1=np.median(bias1)
     mean_bias2=np.median(bias2)
     mean_flat1=np.median(flat1)
     mean_flat2=np.median(flat2)

     _,_,std_biasdiff=sigma_clipped_stats(bias1-bias2,sigma=4.0,maxiters=2)
     _,_,std_flatdiff=sigma_clipped_stats(flat1-flat2,sigma=4.0,maxiters=2)

     print (mean_bias1,mean_bias2,mean_flat1,mean_flat2,std_biasdiff,std_flatdiff)

     gain=((mean_flat1+mean_flat2)-(mean_bias1+mean_bias2))/((std_flatdiff**2-std_biasdiff**2))
     rdnoise = gain*std_biasdiff/np.sqrt(2.)
     print("gain: ",gain, "readout noise:",rdnoise)

    else:
     print ("neither bias nor flat are provided")
     gain = 1.0
     rdnoise = 999

    return gain,rdnoise

def reduct_data(workdir,pre_processing):
   """
    this routine performs science data reduction
    either it calls the pre-processing routine to create master dark and 
    master flats or it read the already computed master files
   """

   if (pre_processing):
     compute_pre_processing(workdir)

   hdu_masterdark = fits.open(workdir+"/Reduced/masterdark.fits")
   masterdark = hdu_masterdark[0].data

   hdu_masterflat = fits.open(workdir+"/Reduced/masterflat.fits")
   masterflat = hdu_masterflat[0].data

   # now read science data, calibrate them and store them into FITS files
   sfiles = glob.glob(workdir+"/LIGHT/*.cr2")
   sfiles.sort()

   for i,ifile in enumerate(sfiles):
      print("reducing (dark subtraction, flat-field and clipping) :",i+1,len(sfiles),ifile)
      raw=rawpy.imread(ifile)
      g1=raw.raw_image[1::2,::2]
      g2=raw.raw_image[::2,1::2]
      g=0.5*(g1+g2)-2047

      print (g.shape,masterdark.shape,masterflat.shape)
      # trim the image removing the first and last 50 pixels for each
      # row and column. Masters have same dimensions
      g = g[50:g.shape[0]-50,50:g.shape[1]-50]
      md = masterdark[50:masterdark.shape[0]-50,50:masterdark.shape[1]-50]
      mf = masterflat[50:masterflat.shape[0]-50,50:masterflat.shape[1]-50]

      g = (g - md)/mf

      # save the reduce science data into FITS file
      hdu = fits.PrimaryHDU(data=g)

      # read meta data from CR2 file and populate FITS header
      image=pyexiv2.Image(ifile)
      exif = image.read_exif()
      items = exif.items()
      for it in items:
        if (it[0] == 'Exif.Image.DateTime'):
          b=datetime.strptime(it[1],'%Y:%m:%d %H:%M:%S')
          date_obs=f'{b.year}-{b.month}-{b.day}'
          timstart=f'{b.hour}:{b.minute}:{b.second}'
        if (it[0] == 'Exif.Photo.ISOSPeedRatings'):
          hdu.header['GAIN']=it[1]
        if (it[0] == 'Exif.Photo.ExposureTime'):
          first,second=it[1].split("/")
          hdu.header['EXPTIME'] = np.float32(first)/np.float32(second)
        if (it[0] == 'Exif.Image.Make'):
          hdu.header['CAMMAKER'] = it[1]
        if (it[0] == 'Exif.Image.Model'):
          hdu.header['INSTRUM'] = it[1]

      hdu.header['EQUINOX']=2000.0
      hdu.header['FILTER'] = 'TG'

      # compute the air mass given obs time, observatory location and
      # object coordinates. Assumes center of image is at object coordinates

      coords = ["05:04:13.42 -03:47:14.2"]
      c = SkyCoord(coords,frame='fk5',unit=(u.hourangle,u.deg))
      # Put 2 hours shift since Image.DateTime is increased by 1 hour
      utcoffset = 2.*u.hour 
      obs_location = EarthLocation(lat=45.53*u.deg,lon=9.4*u.deg,height=133*u.m)
      time = Time(f"{date_obs}T{timstart}",format='isot')-utcoffset
      time.format='fits'
      hdu.header['DATE-OBS'] = (time.value,'[UTC] Start time of exposure')
      caltaz = c.transform_to(AltAz(obstime=time,location=obs_location))
      hdu.header['AIRMASS']=(np.float32(caltaz.secz[0]),'Airmass value')

      # save the un-calibrated science frames into FITS files
      hdu.writeto(f'{workdir}/Reduced/light_{i+1}_uncalib.fits',overwrite=True)

def compute_background(file,gain):
    """ compute background for the selected image
    Input
     file: input image

    Output
     bkg_median and bkg_median_rms
    """
    sigma_clip = SigmaClip(sigma=3.)
    data = fits.getdata(file)
    mask = pht.make_source_mask(data,nsigma=3,npixels=5,dilate_size=11)
    bkg_estimator = pht.SExtractorBackground()
    bkg = pht.Background2D(data, (64,64), mask=mask, filter_size=(3,3), sigma_clip = sigma_clip, bkg_estimator=bkg_estimator)

    f,axs = plt.subplots(1,2,figsize=(16,8))
    axs[0].imshow(bkg.background,origin='lower')
    axs[0].set_title("background")
    axs[1].imshow(bkg.background_rms,origin='lower')
    axs[1].set_title("background rms")
    plt.show()

    error = calc_total_error(data-bkg.background,bkg.background_rms,gain)
    return bkg.background_median,bkg.background_rms_median,error

def get_astrometry(reduce_dir,ra,dec):
    """
    Compute astrometric solution using astap. Update file FITS header
    Inputs:
        reduce_dir :  directory with calibrated data
        ra : hour of the start location
        dec: declination of the start location
    """

    # now read science data, calibrate them and store them into FITS files
    sfiles = glob.glob(reduce_dir+"/light*.fits")
    sfiles.sort()

    dec = 90+dec
    for i,ifile in enumerate(sfiles):
      print("compute astromtric solution :",i+1,len(sfiles),ifile)

      os.system(f'astap -f {ifile} -r 10 -ra {ra} -spd {dec} -update')


"""
      # new solve the astrometry. It requires guess input ra/dec coordinates
----sono qui ----
   # create output file name
   indir,infile=os.path.split(ifile)
   rootname,_=os.path.splitext(infile)
   outfile=os.path.join(outdir,"p_"+rootname+".fits")
   hdu = fits.PrimaryHDU(data=g)

   hdu.header['GAIN']=(gain,'GAIN in e-/ADU')
   hdu.header['RDNOISE']=(rdnoise,'readout noise in electron')
   hdu.header['CTYPE1']='RA---TAN'
   hdu.header['CTYPE2']='DEC--TAN'
   hdu.header['CDELT1']=pixscale/3600.0
   hdu.header['CDELT2']=pixscale/3600.0
   hdu.header['CRPIX1']=hdu.header['NAXIS1']/2.0
   hdu.header['CRPIX2']=hdu.header['NAXIS2']/2.0
   hdu.header['CRVAL1']=ra
   hdu.header['CRVAL2']=dec

   
   g1 = (g[550:1050,550:1050]-superbias[550:1050,550:1050])/flat[550:1050,550:1050]

   f,axs = plt.subplots(1,2,figsize=(16,8))
   axs[0].imshow(g[550:1050,550:1050],vmin=np.percentile(g[550:1050,550:1050],1),vmax=np.percentile(g[550:1050,550:1050],99))
   axs[0].set_title("raw data")
   axs[1].imshow(g1,vmin=np.percentile(g1,1),vmax=np.percentile(g1,99))
   axs[1].set_title("calib data")
   plt.show()
"""
