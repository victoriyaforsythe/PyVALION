#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains components for PyVALION software.

"""

import datetime
import netCDF4
import os
import pickle
import re
from siphon.catalog import TDSCatalog
import subprocess
from tqdm import tqdm

import numpy as np
import pandas as pd

import PyVALION
from PyVALION import logger

from PyIRI.main_library import adjust_longitude as adjust_lon


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def download_GIRO_parameters(time_start,
                             time_finish,
                             ion_name,
                             data_save_dir,
                             save_dir,
                             name_run,
                             clean_directory=True,
                             filter_CS=90):
    """Retrieve GIRO ionospheric parameters from fromlgdc.uml.edu.

    Parameters
    ----------
    time_start : datetime.datetime
        Start date and time for the validation period
    time_finish : datetime.datetime
        End date and time for the validation period
    ion_name : array-like
        String arrays of GIRO ionosondes names
    data_save_dir : str
        Directory where to save the downloaded files
    save_dir : str
        Directory where to save the downloaded files
    name_run : str
        String to add to the name of the files for saved results
    clean_directory : bool
        If set to True the downloaded .txt ionosonde file will be deleted
        (default=True)
    filter_CS : float
        Minimum accepted Autoscaling Confidence Score. Expects values from 0 to
        100, with flag values of 999 for manual scaling and -1 if unknown.
        (default=90)

    Returns
    -------
    data_all : dict
        Dictionary with all the data combined.

    """
    PyVALION.logger.info(''.join(['Downloading data from GIRO for: ',
                                  time_start.strftime('%Y%m%dT%H%MZ'), '-',
                                  time_finish.strftime('%Y%m%dT%H%MZ')]))

    output_flag = np.empty((ion_name.size), dtype=bool)

    # Open a file that has the names and locations of all GIRO stations
    # This is important because the user might want to reduce the number
    # of the ionosondes for the validation
    file_ion_name = os.path.join(PyVALION.giro_names_dir, 'GIRO_Ionosondes.p')
    with open(file_ion_name, 'rb') as fopen:
        giro_name = pickle.load(fopen)

    # Download parameter for each ionosonde for the duration of the val period
    for iname in range(0, ion_name.size):

        ionosonde = ion_name[iname]

        # Index of the ionosonde in GIRO table
        ind_giro = np.where(giro_name['name'] == ionosonde)[0]

        # Save downloaded file as
        file_name_str = (ionosonde
                         + '_' + time_start.strftime('%Y%m%dT%H%MZ')
                         + '_' + time_finish.strftime('%Y%m%dT%H%MZ'))

        output_file_txt = os.path.join(data_save_dir,
                                       file_name_str + '.txt')
        output_file_pic = os.path.join(data_save_dir,
                                       file_name_str + '.p')

        # String for wget in GIRO-desired format
        url = ''.join((
            "https://lgdc.uml.edu/common/DIDBGetValues?ursiCode=", ionosonde,
            "&charName=foF2,foF1,hmF2,hmF1,B0,B1&fromDate=",
            time_start.strftime('%Y/%m/%d+%H:%M:%S'), "&toDate=",
            time_finish.strftime('%Y/%m/%d+%H:%M:%S')))

        # Run wget
        subprocess.run(["wget", "-O", output_file_txt, url, '-q'])

        # Empty arrays to concatenate
        adtime = np.empty((0), dtype=datetime.datetime)
        ascore = np.empty((0), dtype=float)
        afof2 = np.empty((0), dtype=float)
        ahmf2 = np.empty((0), dtype=float)
        aB0 = np.empty((0), dtype=float)
        aB1 = np.empty((0), dtype=float)
        alon = np.empty((0), dtype=float)
        alat = np.empty((0), dtype=float)
        acode = np.empty((0), dtype=object)

        # Open downloaded .txt GIRO file and read it
        with open(output_file_txt, 'r') as file:
            lines = file.readlines()
        str_arr = np.array(lines[25:-1])

        for line in str_arr:
            line_arr = line.split()
            if len(line_arr) == 14:
                stamp, score, fof2, hmf2, B0, B1 = read_GIRO_line(line_arr)
                # Concatenate variables into arrays
                adtime = np.concatenate((adtime, stamp), axis=None)
                ascore = np.concatenate((ascore, score), axis=None)
                afof2 = np.concatenate((afof2, fof2), axis=None)
                ahmf2 = np.concatenate((ahmf2, hmf2), axis=None)
                aB0 = np.concatenate((aB0, B0), axis=None)
                aB1 = np.concatenate((aB1, B1), axis=None)
                alon = np.concatenate((alon, giro_name['lon'][ind_giro]),
                                      axis=None)
                alat = np.concatenate((alat, giro_name['lat'][ind_giro]),
                                      axis=None)
                acode = np.concatenate((acode, giro_name['name'][ind_giro]),
                                       axis=None)

        # Combine parameters to a dictionary
        data = {'dtime': adtime, 'score': ascore, 'fof2': afof2, 'hmf2': ahmf2,
                'B0': aB0, 'B1': aB1, 'lon': alon, 'lat': alat, 'name': acode}

        # If there are elements in adtime array pickle the output
        if adtime.size > 0:
            # Pickle file for each ionosonde
            with open(output_file_pic, 'wb') as fclose:
                pickle.dump(data, fclose)
            output_flag[iname] = True
        else:
            output_flag[iname] = False

        # Delete the .txt file if clean_directory is True
        if clean_directory:
            # Remove the .txt file
            os.remove(output_file_txt)

    # output_flag is an array of bool format, has True if data was present,
    # False if not.
    # The size is the same as the given array ion_name.

    # Create new dictionary to store filtered data
    empty_flt = np.empty((0), dtype=float)
    empty_str = np.empty((0), dtype=str)
    giro_name_good = {'name': empty_str, 'city': empty_str, 'lat': empty_flt,
                      'lon': empty_flt}

    # Filter out data that was not downloaded
    for key in giro_name:
        giro_name_good[key] = giro_name[key][output_flag]

    # Combine and filter out data using CS score
    data_filtered = filter_GIRO_parameters(time_start,
                                           time_finish,
                                           giro_name_good['name'],
                                           data_save_dir,
                                           filter_CS=filter_CS)
    # Pickle file
    output_file_str = ('Filtered_independent_ionosondes_' + name_run + '.p')
    write_file = os.path.join(save_dir, output_file_str)
    with open(write_file, 'wb') as fwrite:
        pickle.dump(data_filtered, fwrite)

    return data_filtered


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def read_GIRO_line(line_arr):
    """Read 14-element line from GIRO ionosonde file.

    Parameters
    ----------
    line_arr : str
        String from the GIRO file.

    Returns
    -------
    stamp : datetime.datetime
        Datetime stamp from the line.
    score : float
        Confidence score CS.
    fof2 : float
        Critical frequency of F2 layer in MHz.
    hmf2 : float
        Height of the F2 layer in km.
    B1 : float
        Bottom-side thickness of F2 layer in km.
    B0 : float
        Bottom-side shape parameter unitless.
    """

    try:
        data_stamp = '%Y-%m-%dT%H:%M:%S.000Z'
        stamp = datetime.datetime.strptime(line_arr[0], data_stamp)
    except Exception as err:
        stamp = np.nan
        logger.error(''.join(['Invalid datetime format in line_arr[0], ',
                              'failed with Exception: ', str(err)]))

    score = safe_float(line_arr, 1, 'score')
    fof2 = safe_float(line_arr, 2, 'fof2')
    hmf2 = safe_float(line_arr, 6, 'hmf2')
    B0 = safe_float(line_arr, 10, 'B0')
    B1 = safe_float(line_arr, 12, 'B1')

    return stamp, score, fof2, hmf2, B0, B1


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def safe_float(line_arr, index, label):
    """Check if element is float.

    Parameters
    ----------
    line_arr : str
        String from the GIRO file.
    index : ind
        Index of the array to check.
    label : str
        Label of the array to print out.

    Returns
    -------
    res : flt
        If given element is can be converted to float return it,
        if not, returns nan.
    """

    try:
        res = float(line_arr[index])
    except (ValueError, TypeError):
        logger.error(f'{label} is not a valid float')
        res = np.nan
    return res


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def make_empty_dict_data():
    """Make empty dictionary to collect GIRO data.

    Returns
    -------
    data : dict
        Dictionary with empty elements.
    """

    empty = np.empty((0))
    data = {'dtime': empty, 'score': empty,
            'fof2': empty, 'hmf2': empty,
            'B0': empty, 'B1': empty,
            'lon': empty, 'lat': empty, 'name': empty}
    return data


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def filter_GIRO_parameters(time_start,
                           time_finish,
                           ion_name,
                           save_dir,
                           filter_CS=90):
    """Download data from all GIRO ionosondes for the whole validation period.

    Parameters
    ----------
    time_start : datetime.datetime
        Start date and time for the validation period
    time_finish : datetime.datetime
        End date and time for the validation period
    ion_name : array-like
        String arrays of GIRO ionosondes names
    save_dir : str
        Directory where the downloaded files are
    filter_CS : float
        Minimum accepted Autoscaling Confidence Score. Expects values from 0 to
        100, with flag values of 999 for manual scaling and -1 if unknown.
        (default=90)

    Returns
    -------
    data_all : dict
        Dictionary with all the data combined.

    """

    data_all = make_empty_dict_data()

    for iion in range(0, ion_name.size):
        ionosonde = ion_name[iion]

        # Save downloaded file as
        file_name_str = (ionosonde
                         + '_'
                         + time_start.strftime('%Y%m%dT%H%MZ')
                         + '_'
                         + time_finish.strftime('%Y%m%dT%H%MZ') + '.p')

        input_file_pic = os.path.join(save_dir, file_name_str)
        with open(input_file_pic, 'rb') as fin:
            data = pickle.load(fin)

        # Filter out nans in hmf2
        data_new = make_empty_dict_data()
        a = np.where(np.isfinite(data['hmf2']) == 1)[0]
        for key in data_new:
            data_new[key] = data[key][a]

        # Filter out nans in fof2
        data_new2 = make_empty_dict_data()
        a = np.where(np.isfinite(data_new['fof2']) == 1)[0]
        for key in data_new2:
            data_new2[key] = data_new[key][a]

        # Filter out small scores (below filter_CS)
        data_new3 = make_empty_dict_data()
        a = np.where(data_new2['score'] > filter_CS)[0]
        for key in data_new3:
            data_new3[key] = data_new2[key][a]

        # Filter out small-size files (< 10 data points)
        if data_new3['fof2'].size > 10:
            for key in data_all:
                data_all[key] = np.concatenate((data_all[key],
                                                data_new3[key]),
                                               axis=None)

    # Convert datetime to datetime in array form using pandas
    data_all['dtime'] = pd.to_datetime(data_all['dtime'])

    return data_all


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def freq2den(freq):
    """Convert ionospheric frequency to plasma density.

    Parameters
    ----------
    freq : array-like
        ionospheric freqeuncy in MHz.

    Returns
    -------
    dens : array-like
        plasma density in m-3.

    Notes
    -----
    This function converts ionospheric frequency to plasma density.

    """
    dens = 1.24e10 * (freq)**2
    return dens


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def nearest_element(array, value):
    """Locate nearest element in array.

    Parameters
    ----------
    array : array-like
        Given array.
    valeu : flt
        Given value.

    Returns
    -------
    smallest_difference_index : int
        Index of the element.

    """
    absolute_val_array = np.abs(array - value)
    smallest_difference_index = absolute_val_array.argmin()
    return smallest_difference_index


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_G_and_y(adtime,
                 alon, alat,
                 data,
                 save_dir,
                 name_run,
                 pickle_outputs=True):
    """Create geometry matrix G and observation vector.

    Parameters
    ----------
    adtime : datetime.datetime
        Array of time
    alon : array-like
        Model array of longitude in degrees (from -180 to 180)
    alat : array-like
        Model array of latitudes in degrees (from -90 to 90)
    data : dict
        Dictionary of GIRO data produced by filter_GIRO_parameters.
    save_dir : str
        Directory where to save the downloaded files
    name_run : str
        String to add to the name of the files for saved results
    pickle_outputs : bool
        It true the outputs are also pickled in the root directory, default
        is True.

    Returns
    -------
    y : dict
        Dictionary that contains observation vectors:
        ay_nmf2 : array-like
            Array of GIRO NmF2 parameter in m-3.
        ay_hmf2 : array-like
            Array of GIRO hmF2 parameter in km.
        ay_B0 : array-like
            Array of GIRO B0 parameter in km.
        ay_B1 : array-like
            Array of GIRO B1 parameter, unitless.
        ay_lon : array-like
            Array of observation longitude in degrees.
        ay_lat : array-like
            Array of observation latitudes in degrees.
        ay_name : array-like
            Array of ionosonde names.
        ay_time : array-like
            Array of observation time.
    units : dict
        Dictionary with strings of units for the keys in dict y
    G : array-like
        Geometry matrix (N_obs, N_time, N_lat, N_lon)
    ion_info : dict
        Dictionary with ionosonde information
        lon : array-like
            Array of longitudes for the stations in degrees.
        ay_lat : array-like
            Array of latitudes latitudes for the stations in degrees.
        name : array-like
            Array of names of the stations.
    """

    # Look for 15/2 min data around
    adtime0 = adtime - datetime.timedelta(minutes=15) / 2.
    adtime1 = adtime + datetime.timedelta(minutes=15) / 2.

    # Empty arrays for concatenation
    ay_hmf2 = np.empty((0))
    ay_fof2 = np.empty((0))
    ay_B0 = np.empty((0))
    ay_B1 = np.empty((0))
    ay_lon = np.empty((0))
    ay_lat = np.empty((0))
    ay_time = np.empty((0))
    ay_name = np.empty((0))

    # Cycle through the array of time frames in a selected day
    PyVALION.logger.info('Forming observation vectors:')
    for it in range(0, adtime.size):
        a = np.where((data['dtime'] >= adtime0[it])
                     & (data['dtime'] < adtime1[it]))[0]

        # Unique ionosondes for this time frame
        un_ion = np.unique(data['name'][a])

        # Cycle through unique ions in the selected time period
        for iion in range(0, un_ion.size):
            b = np.where(data['name'][a] == un_ion[iion])[0]

            # If there is data populate arrays with mean ionosonde obs
            if data['dtime'][a[b]].size > 0:
                mean_fof2 = np.nanmean(data['fof2'][a[b]])
                mean_hmf2 = np.nanmean(data['hmf2'][a[b]])
                mean_B0 = np.nanmean(data['B0'][a[b]])
                mean_B1 = np.nanmean(data['B1'][a[b]])

                # Concatenate obs
                ay_hmf2 = np.concatenate((ay_hmf2, mean_hmf2), axis=None)
                ay_fof2 = np.concatenate((ay_fof2, mean_fof2), axis=None)
                ay_B0 = np.concatenate((ay_B0, mean_B0), axis=None)
                ay_B1 = np.concatenate((ay_B1, mean_B1), axis=None)

                # Concatenate position and time arrays
                ion_lon = adjust_lon(np.unique(data['lon'][a[b]])[0], 'to180')
                ion_lat = np.unique(data['lat'][a[b]])[0]
                ay_lon = np.concatenate((ay_lon, ion_lon), axis=None)
                ay_lat = np.concatenate((ay_lat, ion_lat), axis=None)
                ay_name = np.concatenate((ay_name, un_ion[iion]), axis=None)
                ay_time = np.concatenate((ay_time, adtime[it]), axis=None)

    # Convert frequency to density in cm-3
    ay_nmf2 = PyVALION.library.freq2den(ay_fof2)

    PyVALION.logger.info('Observation vector y has N_obs = ', ay_nmf2.size)
    # Make array for geometry matrix
    G = np.zeros((ay_name.size, adtime.size, alat.size, alon.size))

    PyVALION.logger.info('Forming G:')
    for iob in range(0, ay_name.size):
        it = np.where(adtime == ay_time[iob])[0][0]
        ind_lon = PyVALION.library.nearest_element(alon, ay_lon[iob])
        ind_lat = PyVALION.library.nearest_element(alat, ay_lat[iob])
        G[iob, it, ind_lat, ind_lon] = 1.

    PyVALION.logger.info('G has shape [N_obs, N_time, N_lat, N_lon] = ',
                         G.shape)

    # Write observation vectors to the dictionary
    y = {'fof2': ay_fof2, 'NmF2': ay_nmf2, 'hmF2': ay_hmf2, 'B0': ay_B0,
         'B1': ay_B1, 'lon': ay_lon, 'lat': ay_lat,
         'time': ay_time, 'name': ay_name}

    units = {'fof2': 'Hz', 'NmF2': 'm$^{-3}$', 'hmF2': 'km', 'B0': 'km',
             'B1': 'unitless', 'lon': '°', 'lat': '°',
             'time': 'datetime object', 'name': 'unitless'}

    # Save information about ionosonde stations
    un_names = np.unique(y['name'])

    # Open GIRO information file to pull lon and lat for these stations
    file_ion_name = os.path.join(PyVALION.giro_names_dir, 'GIRO_Ionosondes.p')
    with open(file_ion_name, 'rb') as fopen:
        giro_name = pickle.load(fopen)
    # Find elements from GIRO array of names as in un_names array
    a = np.where(np.isin(giro_name['name'], un_names))[0]
    un_lon = giro_name['lon'][a]
    un_lat = giro_name['lat'][a]

    ion_info = {'name': un_names, 'lon': un_lon, 'lat': un_lat}

    return y, units, G, ion_info


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_model_data(field, G):
    """Find model data for the given field and forward operator.

    Returns
    -------
    field : array-like
        2-D filed of a model parameter in shape [N_time, N_lat, N_lon].
    G : array-like
        Forward operator matrix for the ionosonde data in shape [N_obs, N_time,
        N_lat, N_lon].

    Returns
    -------
    model_data : array-like
        Array of model data, the expected data according to the model.

    """

    # Check that G is compatible with field
    if (field.shape[0: 2] == G.shape[1:3]):
        # Before G and filed can be multiplied, we need to reshape them into
        # [N_obs, N_filed] and [N_field] where N_field is a size of a flattened
        # array that combines time and horizontal grid dimmentions
        nht = G.shape[1] * G.shape[2] * G.shape[3]

        G_reshaped = np.reshape(G, (G.shape[0], nht))
        field_reshaped = np.reshape(field, (nht))

        # Replace nans with zeros (NIMO1 forecast has nans)
        field_reshaped = np.nan_to_num(field_reshaped)

        # Multiply G matrix and model field matrix
        model_data = np.matmul(G_reshaped, field_reshaped)
    else:
        flag = 'Error: G and filed are not compatable.'
        logger.error(flag)

    return model_data


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def find_residuals(model, G, obs_data, obs_info, units):
    """Find model data for the given field and forward operator.

    Returns
    -------
    model : dict
        Dictionary with model parameters in shape [N_time, N_lat, N_lon].
    G : array-like
        Forward operator matrix for the ionosonde data in shape [N_obs, N_time,
        N_lat, N_lon].
    obs_data : dict
        Dictionary that contains observation vectors:
        ay_nmf2 : array-like
            Array of GIRO NmF2 parameter in m-3.
        ay_hmf2 : array-like
            Array of GIRO hmF2 parameter in km.
        ay_B0 : array-like
            Array of GIRO B0 parameter in km.
        ay_B1 : array-like
            Array of GIRO B1 parameter, unitless.
        ay_lon : array-like
            Array of observation longitude in degrees.
        ay_lat : array-like
            Array of observation latitudes in degrees.
        ay_name : array-like
            Array of ionosonde names.
        ay_time : array-like
            Array of observation time.
    obs_info : dict
        Dictionary with ionosonde information
        lon : array-like
            Array of longitudes for the stations in degrees.
        ay_lat : array-like
            Array of latitudes latitudes for the stations in degrees.
        name : array-like
            Array of names of the stations.
    units : dict
        Dictionary with strings of units for the keys in dict y


    Returns
    -------
    model_data : dict
        Array of model data, the expected data according to the model.
    residuals : dict
        Array of residuals.
    model_units : dict
        Dictionary with strings of units for the keys in dict model_data
    res_ion : dict
        Array of mean residuals for each ionosonde station.

    """

    # The dictionary model_data will be used to store model predictions at
    # observation points
    model_data = {}

    # The dictionary residuals will store the differences between the observed
    # data and the model predictions
    residuals = {}

    # The dictionary model_units will store the units (as strings) for each key
    # in the model_data dictionary
    model_units = {}

    # Loop through all parameters in the model dictionary to extract model data
    # at observation points
    for key in model:
        model_data[key] = PyVALION.library.find_model_data(model[key], G)
        residuals[key] = obs_data[key] - model_data[key]
        model_units[key] = units[key]

    # Create dictionary to save individual mean residuals for each ionosonde
    res_ion = dict.fromkeys(residuals)

    # Loop through all parameters in the model dictionary
    for key in residuals:
        # Make an array with the size of the number of ionosondes
        res_ion[key] = np.zeros((obs_info['name'].size))

        # Loop through all ionosondes
        for i in range(0, obs_info['name'].size):
            # Find mean residuals for each ionosonde
            a = np.where(obs_data['name'] == obs_info['name'][i])[0]
            res_ion[key][i] = np.nanmean(residuals[key][a])

    return model_data, residuals, model_units, res_ion


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def download_Jason_TEC(time_start,
                       time_finish,
                       save_dir,
                       name_run='',
                       save_data_option=False,
                       sat_names=np.array(["JA2", "JA3"]),
                       jason_manifest_filename="jason_manifest.txt"):
    """
    Retrieve Jason ionospheric TEC from from www.ncei.noaa.gov THREDDS

    Parameters
    ----------
    time_start : datetime.datetime
        Start date and time for the validation period
    time_finish : datetime.datetime
        End date and time for the validation period
    save_dir : str
        Directory where to save the downloaded data and where Jason file
        manifest is stored.
    name_run : str
        String to add to the name of the files for saved results.
        Defaults to empty string.
    save_data_option : bool
        Option to save data as a pickle (.p) file into save_dir. Defaults to
        False.
    sat_names : array-like
        String arrays of Jason satellite names ("JA2" and/or "JA3").
    jason_manifest_filename : str
        String to designate an alternative Jaosn file manifest filename.

    Returns
    -------
    data_all : dict
        Dictionary with all the data combined.

    """

    # Create or update Jason file manifest
    jason_manifest_path = os.path.join(save_dir, jason_manifest_filename)
    create_or_update_manifest(jason_manifest_path)

    # Cross-reference user-inputted dates with file manifest
    matching_urls = filter_urls_by_timerange(jason_manifest_path, time_start,
                                             time_finish, sat_names)

    # Initialize data dictionary
    data_all = make_empty_dict_data_jason()

    # Loop and read through all matching urls
    # for row in matching_urls:
    print(f"\nProcessing Jason TEC data from {len(matching_urls)} files.")
    for row in tqdm(matching_urls, desc="Jason file progress",
                    unit="file"):
        url = row[0]

        # Perform different file reading routine for Jason-2 and Jason-2 files
        if "JA3" in url and any(sat_names == "JA3"):
            # data_file = read_jason3_file(url)
            data_file = read_jason3_file(url)
        elif "JA2" in url and any(sat_names == "JA2"):
            data_file = read_jason2_file(url)

        # Concatenate all fields in the dictionary
        # data_all = {k: np.concatenate([data_all[k], data_file[k]])
        #             for k in data_all}
        data_all = concat_data_dicts(data_all, data_file)

    # Convert dtime type
    data_all['dtime'] = pd.to_datetime(data_all['dtime'])
    # Indices for data within timespan
    t_match = ((data_all['dtime'] >= time_start)
               & (data_all['dtime'] <= time_finish))
    # Clear out data not within time spec
    data_all = mask_dict(data_all, t_match)

    if save_data_option:
        # Jason raw data filename
        file_str_raw = ('Jason_TEC_raw_' + name_run + '.p')
        file_path_raw = os.path.join(save_dir, file_str_raw)
        print(f"Saving Raw Jason TEC data to: '{file_path_raw}'")
        pickle.dump(data_all, open(file_path_raw, "wb"))

    return data_all


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def create_or_update_manifest(output_file):

    if not os.path.exists(output_file):
        create_manifest(output_file)
    else:
        update_manifest(output_file)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def create_manifest(output_file):
    # Print message to user
    print(f"Creating Jason data file manifest as '{output_file}'.")

    # Jason-2 and Jason-3 catalog URLs (THREDDS):
    catalog_urls = [
        "https://www.ncei.noaa.gov/thredds-ocean/catalog/jason2/gdr/gdr/"
        "catalog.xml",
        "https://www.ncei.noaa.gov/thredds-ocean/catalog/jason3/gdr/gdr/"
        "catalog.xml",
        "https://www.ncei.noaa.gov/thredds-ocean/catalog/jason3/gdr/gdr/gdr/"
        "catalog.xml"
    ]

    print("Counting total cycle catalogs across all Jason-2 and Jason-3 "
          "sources...")
    total_cycles = sum(count_cycle_subcatalogs(url) for url in
                       catalog_urls)
    print(f"Total of {total_cycles} cycle catalogs will be scanned.")

    all_nc_urls = []

    # Use progress bar for combined processing
    with tqdm(total=total_cycles, desc="Processing cycles") as progress:
        for url in catalog_urls:
            all_nc_urls.extend(list_nc_files(url, progress=progress))

    # Deduplicate by filename
    seen = {}
    for url in all_nc_urls:
        fname = os.path.basename(url)
        if (fname not in seen) and "JA2_GPN_2Pf" not in fname:
            seen[fname] = url  # Save only first occurrence

    # Sort by cycle number
    sorted_urls = sorted(seen.values(), key=extract_first_date)

    with open(output_file, "w") as f:
        for url in sorted_urls:
            f.write(url + "\n")

    print(f"{len(sorted_urls)} unique .nc URLs saved to '{output_file}'")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def update_manifest(output_file):
    """
    Update an existing manifest by scanning for .nc files in newer cycles.
    Only scans 'cycleXXX' folders with equal or higher cycle number than the
    last entry. Appends only files with timestamps newer than the most recent
    entry.
    """
    # Print message to user
    print(f"File '{output_file}' already exists. Checking for updates.")

    # Load existing URLs
    with open(output_file, "r") as f:
        existing_urls = [line.strip() for line in f if line.strip()]

    if not existing_urls:
        print("Jason file manifest is empty. Creating new manifest")
        create_manifest(output_file)
        return

    # Get cycle number and timestamp from last URL in manifest
    last_url = existing_urls[-1]
    last_cycle = extract_cycle_number(last_url)
    last_timestamp = extract_first_date(last_url)

    # Check that the final file is not a Jason-2 entry
    if "JA2" in last_url:
        print("WARNING: Last listed file is a Jason-2 entry. Rebuilding "
              "manifest...")
        create_manifest(output_file)
        return

    # Hardcoded threshold datetime fir Jason-2 file end
    # (do not update Jason-2 manifest listings)
    threshold_str = "20191001_065045"
    threshold_dt = datetime.datetime.strptime(threshold_str, "%Y%m%d_%H%M%S")
    if last_timestamp < threshold_dt:
        print(f"WARNING: Last listed file timestamp {last_timestamp} is older "
              f"than end of Jason-2 record. Rebuilding manifest...")
        create_manifest(output_file)
        return

    print(f"Last recorded data: cycle: {last_cycle}, "
          f"timestamp: {last_timestamp}")

    # Jason-3 THREDDS catalog URLs
    catalog_urls = [
        "https://www.ncei.noaa.gov/thredds-ocean/catalog/jason3/gdr/gdr/"
        "catalog.xml",
        "https://www.ncei.noaa.gov/thredds-ocean/catalog/jason3/gdr/gdr/gdr/"
        "catalog.xml"
    ]

    all_new_urls = []
    all_cycles = []

    # Collect all newer cycle references first
    for root_url in catalog_urls:
        try:
            cat = TDSCatalog(root_url)
            for ref in cat.catalog_refs.values():
                if "cycle" in ref.title.lower():
                    cycle_num = extract_cycle_number(ref.title)
                    if cycle_num >= last_cycle:
                        all_cycles.append((cycle_num, ref.href))
        except Exception as e:
            print(f"Failed to access {root_url}: {e}")

    print(f"{len(all_cycles)} total cycles will be scanned up to/including "
          f"cycle {last_cycle})")

    # Process each cycle with progress bar
    with tqdm(total=len(all_cycles), desc="Processing cycles") as progress:
        for cycle_num, cycle_url in sorted(all_cycles):
            found_urls = list_nc_files(cycle_url)
            for url in found_urls:
                timestamp = extract_first_date(url)
                if timestamp > last_timestamp:
                    all_new_urls.append(url)
            progress.update(1)

    # Deduplicate by filename
    seen_filenames = set(os.path.basename(u) for u in existing_urls)
    new_unique_urls = []
    for url in all_new_urls:
        fname = os.path.basename(url)
        if fname not in seen_filenames:
            new_unique_urls.append(url)
            seen_filenames.add(fname)

    # Sort by extracted date
    new_unique_urls.sort(key=extract_first_date)

    # Append to manifest
    if new_unique_urls:
        with open(output_file, "a") as f:
            for url in new_unique_urls:
                f.write(url + "\n")
        print(f"Added {len(new_unique_urls)} new entries to manifest.")
    else:
        print("No new files to add. Manifest is up to date.")


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def count_cycle_subcatalogs(url):
    """
    Count the number of cycle sub-catalogs in a THREDDS catalog.

    Parameters
    ----------
    url : str
        Base catalog url from www.ncei.noaa.gov THREDDS containing "cycle"
        sub-catalogs.

    Returns
    -------
    cycle_num : int
        Total number of cycle sub-catalogs in base url

    """

    try:
        cat = TDSCatalog(url)
        return sum("cycle" in ref.title.lower() for ref in
                   cat.catalog_refs.values())
    except Exception:
        return 0


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def list_nc_files(url, root_base=None, progress=None):
    """
    Recursively list all .nc file HTTPServer URLs under a THREDDS catalog.
    """
    nc_urls = []

    try:
        cat = TDSCatalog(url)
        if root_base is None:
            root_base = url.rsplit('catalog.xml', 1)[0]

        for _, ds in cat.datasets.items():
            if 'HTTPServer' in ds.access_urls:
                full_url = ds.access_urls['HTTPServer']
                # Specify dodsC for subsequent downloads
                full_url = full_url.replace("/fileServer/", "/dodsC/")
                nc_urls.append(full_url)

        for subcat_ref in cat.catalog_refs.values():
            if "cycle" in subcat_ref.title.lower():
                nc_urls.extend(list_nc_files(subcat_ref.href, root_base,
                                             progress))
                if progress:
                    progress.update(1)

    except Exception:
        pass

    return nc_urls


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def extract_first_date(url):
    """
    Extract the first occurrence of a datetime stamp in the format
    YYYYMMDD_HHMMSS from the filename.
    Returns a datetime object if found, otherwise a fallback datetime far in
    the future.

    Parameters
    ----------
    url : str
        URL containing Jason filename from www.ncei.noaa.gov THREDDS catalog

    Returns
    -------
    first_date : datetime.datetime
        Start date of the Jason file

    """

    fname = os.path.basename(url)
    match = re.search(r'(\d{8}_\d{6})', fname)
    if match:
        try:
            return datetime.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    return datetime.datetime.max  # fallback if no valid date is found


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def extract_cycle_number(url):
    """
    Extracts the cycle number from a URL string (e.g., 'cycle345' → 345). Used
    to update Jason file manifest from last recorded cycle.
    Returns -1 if not found.

    Parameters
    ----------
    url : str
        URL containing Jason filename from www.ncei.noaa.gov THREDDS catalog

    Returns
    -------
    cycle_num : int
        Cycle number from Jason filename URL

    """
    match = re.search(r'cycle(\d{3})', url)
    return int(match.group(1)) if match else -1


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def filter_urls_by_timerange(manifest_filepath,
                             user_start,
                             user_finish,
                             sat_names):
    """
    Reads a .txt file of URLs and returns only those where the file's time
    range overlaps with the user-specified datetime window.

    Parameters
    ----------
        manifest_filepath : str
            Path to Jason file manifest (.txt file containing URLs)
        user_start : datetime.datetime
            Start date and time for the validation period
        user_finish : datetime.datetime
            End date and time for the validation period
        sat_names : np.ndarray
            Array of satellite name substrings to include (e.g., "JA2", "JA3")

    Returns
    -------
        matching_urls : list of tuples
            Each tuple contains (url, start_datetime, end_datetime) for files
            that overlap with the specified time window.
    """
    # Regex pattern to extract start and end datetimes from the filename
    pattern = r'_(\d{8})_(\d{6})_(\d{8})_(\d{6})\.nc'

    # Initialize list to hold matching URL entries
    matching_urls = []

    # Open and read the input file line by line
    with open(manifest_filepath, 'r') as file:
        # Print out progress statements to user
        print(f"\nFiltering for overlap with user-specified time window:"
              f"{user_start} to {user_finish}.")
        lines = file.readlines()
        print(f"Searching {len(lines)} URLs from manifest.")

        for line in tqdm(lines, desc="Filtering URLs", unit="file"):
            url = line.strip()

            # Skip if the URL does not contain any of the desired sat names
            if not any(sat in url for sat in sat_names):
                continue

            # Try to extract date info from the filename
            match = re.search(pattern, url)
            if match:
                start_str = match.group(1) + match.group(2)
                end_str = match.group(3) + match.group(4)
                start_dt = datetime.datetime.strptime(start_str,
                                                      '%Y%m%d%H%M%S')
                end_dt = datetime.datetime.strptime(end_str, '%Y%m%d%H%M%S')

                # Include file if it overlaps with the specified time window
                if end_dt >= user_start and start_dt <= user_finish:
                    matching_urls.append((url, start_dt, end_dt))

    # Return the list of matching URLs and their associated datetimes
    print(f"Found {len(matching_urls)} matching files.")
    return matching_urls


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def make_empty_dict_data_jason():
    """Make empty dictionary to collect Jason-2 and -3 data.

    Returns
    -------
    data : dict
        Dictionary with empty elements.
    """

    empty = np.empty((0))
    data = {'dtime': np.empty(0, dtype='datetime64[s]'),
            'TEC': empty,
            'lon': empty,
            'lat': empty,
            'name': np.empty(0, dtype='<U3')}
    return data


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def read_jason2_file(j2url):
    """Read in relevant data from Jason-2 OPeNDAP URL
    See OSTM/Jason-2 Products Handbook Section 4.2.5 for details.

    Parameters
    ----------
    j2url : str
        Jason-2 OPeNDAP url

    Returns
    -------
    data_all : dict
        Dictionary with all the data from a single file combined.

    """

    # Open Jason-2 netCDF with OPeNDAP URL
    j2file = netCDF4.Dataset(j2url, 'r')

    # Read spatiotemporal variables
    time = j2file.variables['time'][:]
    time = time.filled(np.nan)
    # Define the reference (epoch)
    epoch = np.datetime64('2000-01-01T00:00:00')
    time = (epoch + time.astype('timedelta64[s]'))

    lon = j2file.variables['lon'][:]
    lon = lon.filled(np.nan)
    lon = adjust_lon(lon, 'to180')
    lat = j2file.variables['lat'][:]
    lat = lat.filled(np.nan)

    # Read Jason-2 ku-band delay and fill missing
    iono_ku = j2file.variables['iono_corr_alt_ku'][:]
    iono_ku = iono_ku.filled(np.nan)

    # Flags
    # Surface type flag
    surf_flag = j2file.variables['surface_type'][:]
    surf_flag = surf_flag.filled(np.nan)
    # Open sea ice flag
    ice_flag = j2file.variables['ice_flag'][:]
    ice_flag = ice_flag.filled(np.nan)
    # RMS of the "ocean" altimeter range
    rms_flag = j2file.variables['range_rms_ku'][:]
    rms_flag = rms_flag.filled(np.nan)
    # Number of valid points used to compute the “ocean” altimeter range
    points_flag = j2file.variables['range_numval_ku'][:]
    points_flag = points_flag.filled(np.nan)

    # Only measurements collected a) over the ocean, b) not on ice, c) for
    # which the range at 1 Hz was computed with enough valid range observations
    # at 20 Hz (RMS below a threshold 0.2 and a number of 20 Hz observations
    # above a threshold of 10) and d) for which the ionospheric correction
    # could be computed using equation 1 (its values different than Default
    # Value (DV)) are selected.
    valid = (
        (surf_flag == 0)
        & (ice_flag == 0)
        & (rms_flag > 0)
        & (rms_flag < 0.2)
        & (points_flag > 10)
    )

    iono_ku[~valid] = np.nan

    # Filter data
    iono_ku_filt, filt_flag = robust_iterative_filter(iono_ku)
    iono_ku_filt[filt_flag] = np.nan

    # Compute TEC from Jason-2 ku-band delay
    tec = compute_jason_tec(iono_ku_filt)
    j2_sat_tec_bias = -3.5
    tec = tec + j2_sat_tec_bias
    tec[tec < 0] = 0

    # Initialize empy data dictionary
    data_all = make_empty_dict_data_jason()

    # Remove entries with nans
    delInd = np.isnan(time) | np.isnan(lat) | np.isnan(lon) | np.isnan(tec)
    time = np.delete(time, delInd)
    lat = np.delete(lat, delInd)
    lon = np.delete(lon, delInd)
    tec = np.delete(tec, delInd)

    # Fill data dictionary
    N = len(lat)
    data_all['dtime'] = time
    data_all['lat'] = lat
    data_all['lon'] = lon
    data_all['TEC'] = tec
    data_all['name'] = np.full(N, 'JA2', dtype='<U3')

    return data_all


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def read_jason3_file(j3url):
    """Read in relevant data from Jason-3 OPeNDAP URL.
    See OSTM/Jason-3 Products Handbook for details.

    Parameters
    ----------
    j3url : str
        Jason-3 OPeNDAP url

    Returns
    -------
    data_all : dict
        Dictionary with all the data from a single file combined.

    """

    #  Open Jason-3 netCDF with OPeNDAP URL
    j3file = netCDF4.Dataset(j3url, 'r')

    # Read spatiotemporal variables
    time = j3file.variables['data_01%2ftime'][:]
    time = time.filled(np.nan)
    # Define the reference (epoch)
    epoch = np.datetime64('2000-01-01T00:00:00')
    time = (epoch + time.astype('timedelta64[s]'))

    lon = j3file.variables['data_01%2flongitude'][:]
    lon = lon.filled(np.nan)
    lon = adjust_lon(lon, 'to180')
    lat = j3file.variables['data_01%2flatitude'][:]
    lat = lat.filled(np.nan)

    # Read Jason-3 ku-band delay and fill missing
    iono_ku = j3file.variables['data_01%2fku%2fiono_cor_alt'][:]
    iono_ku = iono_ku.filled(np.nan)

    # Flags
    # Surface type flag
    surf_flag = j3file.variables['data_01%2fsurface_classification_flag'][:]
    surf_flag = surf_flag.filled(np.nan)
    # Open sea ice flag
    ice_flag = j3file.variables['data_01%2fice_flag'][:]
    ice_flag = ice_flag.filled(np.nan)
    # RMS of the "ocean" altimeter range
    rms_flag = j3file.variables['data_01%2fku%2frange_ocean_rms'][:]
    rms_flag = rms_flag.filled(np.nan)
    # Number of valid points used to compute the “ocean” altimeter range
    points_flag = j3file.variables['data_01%2fku%2frange_ocean_numval'][:]
    points_flag = points_flag.filled(np.nan)

    # Only measurements collected a) over the ocean, b) not on ice, c) for
    # which the range at 1 Hz was computed with enough valid range observations
    # at 20 Hz (RMS below a threshold 0.2 and a number of 20 Hz observations
    # above a threshold of 10) and d) for which the ionospheric correction
    # could be computed using equation 1 (its values different than Default
    # Value (DV)) are selected.
    valid = (
        (surf_flag == 0)
        & (ice_flag == 0)
        & (rms_flag > 0)
        & (rms_flag < 0.2)
        & (points_flag > 10)
    )

    iono_ku[~valid] = np.nan

    # Filter data
    iono_ku_filt, filt_flag = robust_iterative_filter(iono_ku)
    iono_ku_filt[filt_flag] = np.nan

    # Compute TEC from Jason-3 ku-band delay
    tec = compute_jason_tec(iono_ku_filt)
    j3_sat_tec_bias = -1
    tec = tec + j3_sat_tec_bias
    tec[tec < 0] = 0

    # Initialize empy data dictionary
    data_all = make_empty_dict_data_jason()

    # Remove entries with nans
    delInd = np.isnan(time) | np.isnan(lat) | np.isnan(lon) | np.isnan(tec)
    time = np.delete(time, delInd)
    lat = np.delete(lat, delInd)
    lon = np.delete(lon, delInd)
    tec = np.delete(tec, delInd)

    # Fill Jason-3 data dictionary
    N = len(lat)
    data_all['dtime'] = time
    data_all['lat'] = lat
    data_all['lon'] = lon
    data_all['TEC'] = tec
    data_all['name'] = np.full(N, 'JA3', dtype='<U3')

    return data_all


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def robust_iterative_filter(
    data_raw,
    INIT_SIGMA=5,
    DATA_GAP_MAX=0,
    MEDIAN_NB_PTS=30,
    MEDIAN_NB_PTS_MIN=30,
    MEDIAN_NB_PTS_MEAN=0,
    LANCZOS_NB_PTS=50,
    LANCZOS_NB_PTS_CUTOFF=50,
    LANCZOS_NB_PTS_MIN=10,
    CONVERGENCE_THRESHOLD=0
):
    data = data_raw.copy()
    outlier_mask = np.zeros(len(data), dtype=bool)

    # Step 2: Initial global sigma outlier detection
    mean = np.nanmean(data)
    std = np.nanstd(data)
    init_outliers = np.abs(data - mean) > INIT_SIGMA * std
    outlier_mask |= init_outliers
    data[init_outliers] = np.nan

    # prev_outlier_count = np.sum(outlier_mask)

    # Iterate steps 3–7
    while True:
        # Step 3: Remove current outliers (already done by setting to NaN)

        # Step 4: Fill small gaps
        if DATA_GAP_MAX > 0:
            data = fill_small_gaps(data, DATA_GAP_MAX)

        # Step 5: Median Filter
        data_median = apply_median_filter(data, MEDIAN_NB_PTS,
                                          MEDIAN_NB_PTS_MIN,
                                          MEDIAN_NB_PTS_MEAN)

        # Step 6: Lanczos Filter
        data_lanczos = lanczos_filter(data_median, LANCZOS_NB_PTS,
                                      LANCZOS_NB_PTS_CUTOFF,
                                      LANCZOS_NB_PTS_MIN)

        # Step 7: Flag outliers based on residual
        residual = data_raw - data_lanczos
        residual_std = np.nanstd(residual)
        new_outliers = np.abs(residual) > 3 * residual_std
        new_outliers &= ~outlier_mask  # Don't double-flag

        outlier_mask |= new_outliers
        data = data_raw.copy()
        data[outlier_mask] = np.nan

        # Step 8: Check convergence
        percent_outliers = np.sum(outlier_mask) / len(data_raw)
        if percent_outliers <= CONVERGENCE_THRESHOLD:
            break
        if np.sum(new_outliers) == 0:
            break  # Converged

    return data_lanczos, outlier_mask


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def fill_small_gaps(data, max_gap_size):
    """Linearly interpolate gaps smaller than max_gap_size."""
    # data_filled = data.copy()
    isnan = np.isnan(data)
    idx = np.arange(len(data))
    valid = ~isnan

    # Interpolate all
    interp_all = np.interp(idx, idx[valid], data[valid])

    # Re-mask large gaps
    nan_runs = np.flatnonzero(np.diff(np.concatenate(([0], isnan.view(np.int8),
                                                      [0]))))
    gap_starts = nan_runs[::2]
    gap_ends = nan_runs[1::2]

    for start, end in zip(gap_starts, gap_ends):
        gap_len = end - start
        if gap_len > max_gap_size:
            interp_all[start:end] = np.nan

    return interp_all


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def apply_median_filter(data, half_width, min_valid, mean_width=0):
    """Median filter with optional averaging.
       Preserves NaNs in the original data.
    """
    result = np.full_like(data, np.nan)
    N = len(data)

    for i in range(N):
        if np.isnan(data[i]):
            continue  # preserve NaNs: don't filter at NaN locations

        start = max(0, i - half_width)
        end = min(N, i + half_width + 1)
        window = data[start:end]
        valid = window[~np.isnan(window)]

        if len(valid) >= min_valid:
            if mean_width > 0:
                center = len(valid) // 2
                s = max(0, center - mean_width)
                e = min(len(valid), center + mean_width + 1)
                result[i] = np.mean(np.sort(valid)[s:e])
            else:
                result[i] = np.median(valid)

    return result


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def lanczos_filter(signal, N, cutoff, min_valid=3):
    """Apply Lanczos filter with given parameters, ignoring NaNs."""
    kernel = compute_lanczos_kernel(N, cutoff)
    filtered = np.full_like(signal, np.nan)

    for i in range(len(signal)):
        start = max(i - N, 0)
        end = min(i + N + 1, len(signal))
        segment = signal[start:end]
        k_segment = kernel[N - (i - start):N + (end - i)]

        valid_mask = ~np.isnan(segment)
        if np.sum(valid_mask) >= min_valid:
            filtered[i] = (np.nansum(segment[valid_mask]
                           * k_segment[valid_mask])
                           / np.sum(k_segment[valid_mask]))
    return filtered


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compute_lanczos_kernel(N, cutoff):
    """Build a symmetric Lanczos kernel with half-width N and cutoff."""
    x = np.arange(-N, N + 1)
    sinc_filter = np.sinc(x / cutoff)
    lanczos_window = np.sinc(x / N)
    kernel = sinc_filter * lanczos_window
    return kernel / kernel.sum()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def compute_jason_tec(iono_ku_delay):
    """Jason TEC computation.
    See OSTM/Jason-2 Products Handbook Section 4.2.5 for details.

    Parameters
    ----------
    iono_ku_delay : float
        Jason-2 or Jason-3 ku-band ionospheric delay

    Returns
    -------
    tec : float
        Total electron content in TECU

    """

    scale_topex = 13.575E9 * 13.575E9 / 40.3  # f^2/40.3
    tec = -iono_ku_delay * scale_topex * 1E-16  # conversion to TECu
    return tec


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def concat_data_dicts(A, B):
    """Concatenate two dictionaries with the same field names.

    Parameters
    ----------
    A : dict
        Dictionary with identical field names to B
    B : dict
        Dictionary with identical field names to A

    Returns
    -------
    C : dict
        Dictionary with all the data from A and B combined (A preceeding B)

    """
    C = {}

    for k in A:
        a_val = A[k]
        b_val = B[k]

        if isinstance(a_val, np.ndarray) and isinstance(b_val, np.ndarray):
            if a_val.dtype == b_val.dtype:
                C[k] = np.concatenate([a_val, b_val])
            else:
                raise TypeError(f"Field '{k}' dtype mismatch: {a_val.dtype} vs"
                                f"{b_val.dtype}")
        else:
            raise ValueError(f"Field '{k}' is not a numpy array in both dicts")

    return C


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def mask_dict(data_raw, mask):
    """
    Masks data within all data dictionary fields

    Parameters:
    -----------
    data_raw : dict of arrays (np.ndarray or pd.Series)
        Dictionary of data fields to downsample.
    mask : 0s and 1s
        1 for keep, 0 for discard

    Returns:
    --------
    dict_resamp : dict
        Dictionary with cleaned out fields
    """

    data_clean = {}
    for key, data in data_raw.items():
        data_clean[key] = data[mask]
    return data_clean
