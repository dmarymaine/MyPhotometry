# MyPhotometry
Python code for photometry of variable stars and exo-planet transits


You can perform data pre-processing or directly go to photometry

## Pre-Processing
It assumes that you have a data directory structure as follows:

    workdir 
    ├── LIGHT 	    # your light frames 
    ├── BIAS 	    # your bias frames 
    ├── DARK 	    # your dark frames 
    ├── DARKFLAT 	    # your darkflat frames (same exp-time as flats)
    ├── FLAT            # your flat frames

You can also work with Canon RAW data file (.CR2). The code firstly
extract the two green channels and create FITS files for all your data.
Then it goes to pre-processing with FITS only files.

