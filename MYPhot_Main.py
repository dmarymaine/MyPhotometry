#
# Copyright (C) 2024 - Davide Maino
#
# File: MYPhot_Main.py
#
# Created: 2024-01-11
#
# Author: Davide Maino
#

""" This module is the driver for the photometry package """

import argparse
import json

import MYPhot_Logging as log
from MYPhot_Core import MYPhot_Core

"""
Defines the specific program options

"""


parser = argparse.ArgumentParser(prog='MYPhot_Main')

parser.add_argument('--ver', action='version',version='0.1')
parser.add_argument('--workdir',type=str,required=True,
                    help='The root directory where data files are stored')
parser.add_argument('--target',type=str,required=True,
                    help='A JSON file with information of the target/comparison')

parser.add_argument('--preprocessing',dest='preprocessing',action='store_true',
                    help='whether perform preprocessing of the data')

# input data format
parser.add_argument('--cr2',dest='cr2',action='store_true',
                    help='whether use CR2 input files, if not FITS assumed')

parser.add_argument('--apertures',type=str,required=True,
                    help='A JSON file with list of apertures for photometry')

parser.add_argument('--showplots',dest='showplots',action='store_true',
                    help='wheter produce plots during processing')
# output report
parser.add_argument('--aavso',dest='aavso',action='store_true',
                    help='whether produce an aavso output file')


args = parser.parse_args()

logger = log.getLogger("MYPhot_Main")
logger.info("Entering MYPhot mainMethod()")

module_name = "MYPhot_Main"

# first of all define main variables and read the parameters files
workdir = args.workdir
target_json = args.target
apertures_json = args.apertures

# read target, comparison and bonus star
with open(target_json,'r') as f:
   target_list = json.load(f)

# read list of apertures for photometry
with open(apertures_json,'r') as f:
   apertures = json.load(f)

myphot = MYPhot_Core(args)

# the exec is only either for the preprocessing step or to 
# populate the local variable with already calibrated frame

myphot.exec()

# compute photometry for all objects (creating catalogs)
myphot.compute_allobject_photometry()

# now create circular apertures around the target, comparison and 
# validation stars. 
myphot.get_target_comp_valid_photometry()

# now plot radial profile to check which aperture is optimal



logger.info("Done")

