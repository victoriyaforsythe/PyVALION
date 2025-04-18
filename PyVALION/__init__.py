"""Core library imports for PyVALION."""

# Define a logger object to allow easier log handling
import logging
logging.raiseExceptions = False
logger = logging.getLogger('pyvalion_logger')

osflag = False
try:
    from importlib import metadata
    from importlib import resources
except ImportError:
    import importlib_metadata as metadata
    import os
    osflag = True

# Import the package modules and top-level classes
from PyVALION import library  # noqa F401
from PyVALION import plotting  # noqa F401

# Set version
__version__ = metadata.version('PyVALION')

# Determine the coefficient root directory
if osflag:
    giro_names_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                  'giro_names')
else:
    giro_names_dir = str(resources.files(__package__).joinpath('giro_names'))

if osflag:
    giro_data_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                 'giro_data')
else:
    giro_data_dir = str(resources.files(__package__).joinpath('giro_data'))

del osflag
