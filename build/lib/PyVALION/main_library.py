#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains components for PyIRI software.

References
----------
Forsythe et al. (2023), PyIRI: Whole-Globe Approach to the
International Reference Ionosphere Modeling Implemented in Python,
Space Weather, ESS Open Archive, September 28, 2023,
doi:10.22541/essoar.169592556.61105365/v1.

Bilitza et al. (2022), The International Reference Ionosphere
model: A review and description of an ionospheric benchmark, Reviews
of Geophysics, 60.

Nava et al. (2008). A new version of the NeQuick ionosphere
electron density model. J. Atmos. Sol. Terr. Phys., 70 (15),
doi:10.1016/j.jastp.2008.01.015.

Jones, W. B., Graham, R. P., & Leftin, M. (1966). Advances
in ionospheric mapping by numerical methods.

"""

import datetime as dt
from fortranformat import FortranRecordReader
import math
import numpy as np
import os

import PyVALION
from PyVALION import logger


def download_GIRO_parameters(time_start,
                             time_finish,
                             save_dir='',
                             clean_directory=True,
                             use_subdirs=True,
                             overwrite=False):
    """Retrieve GIRO ionospheric parameters from fromlgdc.uml.edu.

    Parameters
    ----------
    time_start : datetime.datetime
        Start date and time for the validation period
    time_finish : datetime.datetime
        End date and time for the validation period
    save_dir : str
        Directory for GIRO data, or '' to use package directory.
        (default='')
    use_subdirs : bool
        If True, adds YYYY/MMDD subdirectories to the filename path, if False
        assumes that the entire path to the coefficient directory is provided
        by `save_dir` (default=True)
    overwrite : bool
        Allow overwriting of existing parameter files if True (default=False)

    Returns
    -------
    dstat : bool
        Download status: True if file was downloaded, False if not
    fstat : bool
        File status: True if parameter coefficient file exists, False if not
    msg : str
        Potential message with more details about status flags, empty if both
        are True

    """
    PyVALION.logger.info('Downloading data from GIRO for: '
                         + time_start.strftime('%Y%m%dT%H%MZ')
                         + '-' + time_finish.strftime('%Y%m%dT%H%MZ'))

    # Initalize output
    dstat = False
    fstat = False
    msg = ''

    # Read included pickle file that has the names and the locations of all
    # GIRO ionosondes
    
    file_ion_name = open(os.path.join(PyVALION.giro_dir, 'GIRO_Ionosondes.p'), mode='r')
    ion_info = pickle.load(file_ion_name)
    print(ion_info)