# Change Log
All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org/).

## 0.2.0 (04-14-2026)
* Updeted the manifest file to comply with current Jason format
* Added bilinear interpolation to the Jason and GIRO routines
* Added the flag to include Jason negative TEC values
* Updated CI testing to use up-to-date NEP29 and Action versions
* Fixed bug in the docs installation
* Fixed bugs in the docstrings

## 0.1.0 (10-29-2025)
* Implemented Jason TEC functionality in `library.py`
* Implemented Jason TEC plotting functions in `plotting.py`
* Updated `pyproject.toml` to include new dependencies
* Fixed typos in the supporting documentation
* Implemented logging in plot functions
* Implemented `with` statements for all file openings
* Removed logic needed in `__init__.py` for older Python versions
* Added SZA function to use for day-night separation in obs space

## 0.0.1 (05-05-2025)
* Alpha release
