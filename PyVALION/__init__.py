"""Core library imports for PyVALION."""

from importlib import metadata
from importlib import resources
import logging

# Define a logger object to allow easier log handling
logging.raiseExceptions = False
logger = logging.getLogger('pyvalion_logger')

# Import the package modules and top-level classes
from PyVALION import library  # noqa F401
from PyVALION import plotting  # noqa F401

# Set version
__version__ = metadata.version('PyVALION')

# Determine the coefficient root directory
giro_names_dir = str(resources.files(__package__).joinpath('giro_names'))
giro_data_dir = str(resources.files(__package__).joinpath('giro_data'))
