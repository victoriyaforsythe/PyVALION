#!/usr/bin/env python
# --------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# --------------------------------------------------------
"""This library contains components for PyVALION software.

"""

import datetime
import os
import pickle
import subprocess

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
    filter_CS : flt
        Minimum accepted Autoscaling Confidence Score (from 0 to 100, 999 if
        manual scaling, -1 if unknown)

    Returns
    -------
    data_all : dict
        Dictionary with all the data combined.

    """
    PyVALION.logger.info('Downloading data from GIRO for: '
                         + time_start.strftime('%Y%m%dT%H%MZ')
                         + '-' + time_finish.strftime('%Y%m%dT%H%MZ'))

    output_flag = np.empty((ion_name.size), dtype=bool)

    # Open a file that has the names and locations of all GIRO stations
    # This is important because the user might want to reduce the number
    # of the ionosondes for the validation
    file_ion_name = os.path.join(PyVALION.giro_names_dir, 'GIRO_Ionosondes.p')
    giro_name = pickle.load(open(file_ion_name, 'rb'))

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
        url = ("https://lgdc.uml.edu/common/DIDBGetValues?ursiCode="
               + ionosonde + "&charName=foF2,foF1,hmF2,hmF1,B0,B1&fromDate="
               + time_start.strftime('%Y/%m/%d+%H:%M:%S')
               + "&toDate=" + time_finish.strftime('%Y/%m/%d+%H:%M:%S'))

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
            pickle.dump(data, open(output_file_pic, "wb"))
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
    pickle.dump(data_filtered, open(write_file, "wb"))

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
    except Exception:
        stamp = np.nan
        logger.error('Invalid datetime format in line_arr[0]')

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
    except Exception:
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
    filter_CS : flt
        Minimum accepted Autoscaling Confidence Score (from 0 to 100, 999 if
        manual scaling, -1 if unknown)
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

        imput_file_pic = os.path.join(save_dir, file_name_str)
        data = pickle.load(open(imput_file_pic, 'rb'))

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
    giro_name = pickle.load(open(file_ion_name, 'rb'))
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

    # check that G is compatible with field
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
