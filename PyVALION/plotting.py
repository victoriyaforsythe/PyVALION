#!/usr/bin/env python
# ---------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# ---------------------------------------------------------
"""This library contains components visualization routines for PyIRI."""

import os

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from PyVALION import logger


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_ionosondes(y_instr_info,
                    dtime,
                    plot_dir,
                    plot_name='Ionosondes_Map.pdf'):
    """Plot map with GIRO ionosondes that were used for validation.

    Parameters
    ----------
    y_instr_info : dict
        Dictionary with ionosonde information
        lon : array-like
            Array of longitudes for the stations in degrees.
        ay_lat : array-like
            Array of latitudes latitudes for the stations in degrees.
        name : array-like
            Array of names of the stations.
    dtime : class:`datetime.datetime`
        Datetime of the validation day.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='Ionosondes_Map.pdf')

    """

    # Plot map with ionosondes before filtering
    projection1 = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9, 3),
                           constrained_layout=True,
                           subplot_kw={'projection': projection1})
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    ax_plot = ax
    ax_plot.set_xticks(np.arange(-180, 180 + 45, 90), crs=ccrs.PlateCarree())
    ax_plot.set_yticks(np.arange(-90, 90 + 45, 45), crs=ccrs.PlateCarree())
    ax_plot.set_xlabel('Geo Lon (°)')
    ax_plot.set_ylabel('Geo Lat (°)')
    ax_plot.set_facecolor('lightgray')
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax_plot.xaxis.set_major_formatter(lon_formatter)
    ax_plot.yaxis.set_major_formatter(lat_formatter)
    ax_plot.coastlines(lw=0.5, color='black', zorder=1)
    ax_plot.scatter(y_instr_info['lon'], y_instr_info['lat'],
                    zorder=2, linewidth=0.5, edgecolor='black', c='red')
    ax_plot.set_title('GIRO Ionosondes for Validation Period, '
                      + dtime.strftime('%Y%m%d'))

    # Save figure
    figname = os.path.join(plot_dir, plot_name)
    plt.savefig(figname, bbox_inches='tight', facecolor='white')
    logger.info('Figure Ionosonde Map is saved at: ', figname)
    return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_histogram(residuals,
                   model_units,
                   dtime,
                   plot_dir,
                   plot_name='Residuals.pdf'):
    """Plot residuals for each model parameter.

    Parameters
    ----------
    residuals : dict
        Dictionary with residuals.
    model_units: dict
        Dictionary with units for each key in residuals.
    dtime : class:`datetime.datetime`
        Datetime of the validation day.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='Residuals.pdf')

    """

    fig, ax = plt.subplots(1, len(residuals), sharex=False, sharey=False)
    fig.suptitle('Residuals, ' + dtime.strftime('%Y%m%d'))
    fig.set_size_inches(3 * len(residuals), 3)
    keys_list = np.array(list(residuals.keys()), dtype=str)

    for ikey in range(0, keys_list.size):
        ax_plot = ax[ikey]
        ax_plot.set_facecolor('lightgray')
        ax_plot.set_xlabel(keys_list[ikey]
                           + ' (' + model_units[keys_list[ikey]] + ')')
        ax_plot.set_ylabel('Number of obs')
        ax_plot.axvline(x=0, c='black', zorder=2, linestyle='dashed',
                        linewidth=0.5)
        # Plot the histogram of the model residuals
        ax_plot.hist(residuals[keys_list[ikey]], color='red',
                     bins=50, zorder=1)
        # Making x-axis symmetric
        low, high = ax_plot.get_xlim()
        bound = max(abs(low), abs(high))
        ax_plot.set_xlim((-bound, bound))
    plt.tight_layout()

    # Save figure
    figname = os.path.join(plot_dir, plot_name)
    plt.savefig(figname, bbox_inches='tight', facecolor='white')
    logger.info('Figure Residuals is saved at: ', figname)
    return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_individual_mean_residuals(res_ion,
                                   ion_info,
                                   model_units,
                                   dtime,
                                   plot_dir,
                                   plot_name='Ion_Residuals.pdf'):
    """Plot residuals for each ionosonde.

    Parameters
    ----------
    res_ion : dict
        Dictionary with mean residuals for individual ionosondes.
    ion_info : dict
        Dictionary with ionosonde information
        lon : array-like
            Array of longitudes for the stations in degrees.
        ay_lat : array-like
            Array of latitudes latitudes for the stations in degrees.
        name : array-like
            Array of names of the stations.
    model_units: dict
        Dictionary with units for each key in res_ion.
    dtime : class:`datetime.datetime`
        Datetime of the validation day.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='Ion_Residuals.pdf')

    Notes
    -----
    Each residuals parameter will be saved in a different plot, with that
    parameter appended to the start of the default plot name (e.g.,
    'ay_nmf2_Ion_Residuals.pdf')

    """
    # Loop through all parameters in the residuals dictionary
    for key in res_ion:
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False)
        fig.suptitle(key + ' Residuals, ' + dtime.strftime('%Y%m%d'))
        fig.set_size_inches(7, 3)

        ax_plot = ax
        ax_plot.set_facecolor('lightgray')
        ax_plot.set_ylabel(key + ' Residuals (' + model_units[key] + ')')
        ax_plot.set_xlabel('GIRO Ionosonde Names')
        ax_plot.bar(ion_info['name'], res_ion[key], color='red',
                    edgecolor='black', linewidth=0.5)
        plt.xticks(rotation='vertical')
        ax_plot.axhline(y=0, color='black', linewidth=0.5)

        # Making y-axis symmetric
        low, high = ax_plot.get_ylim()
        bound = max(abs(low), abs(high))
        ax_plot.set_ylim((-bound, bound))

        # Save figure
        figname = os.path.join(plot_dir, "_".join([key + plot_name]))
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
        logger.info('Figure Ionosonde Mean Residuals is saved at: ', figname)
    return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_TEC_residuals_map(alat,
                           alon,
                           residuals,
                           dtime,
                           save_option=False,
                           save_dir='',
                           plot_name='TEC_Residuals_Map'):
    """Plot residual map for TEC.

    Parameters
    ----------
    alon : array-like (float)
        Flattened array of longitudes in degrees.
    alat : array-like (float)
        Flattened array of latitudes in degrees.
    residuals : dict
        Dictionary with TEC residuals.
    model_units: dict
        Dictionary with units for each key in residuals.
    dtime : datetime.datetime
        Datetime of the validation day.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='TEC_Residuals_Map.pdf').
    """

    projection1 = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9, 3),
                           constrained_layout=True,
                           subplot_kw={'projection': projection1})
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    ax_plot = ax
    ax_plot.set_xticks(np.arange(-180, 180 + 45, 90), crs=ccrs.PlateCarree())
    ax_plot.set_yticks(np.arange(-90, 90 + 45, 45), crs=ccrs.PlateCarree())
    ax_plot.set_xlabel('Geo Lon (°)')
    ax_plot.set_ylabel('Geo Lat (°)')
    ax_plot.set_facecolor('lightgray')
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax_plot.xaxis.set_major_formatter(lon_formatter)
    ax_plot.yaxis.set_major_formatter(lat_formatter)
    vmax = 30
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)
    sc = ax_plot.scatter(alon, alat, c=residuals['TEC'], s=3,
                         cmap='seismic', norm=norm)
    ax_plot.coastlines(lw=0.5, color='black', zorder=1)
    if np.size(dtime) > 1:
        ax_plot.set_title('Residuals, ' + dtime[0].strftime('%Y%m%d') + ' - '
                          + dtime[-1].strftime('%Y%m%d'))
    else:
        ax_plot.set_title('Residuals, ' + dtime.strftime('%Y%m%d'))
    cbar = fig.colorbar(sc, ax=ax_plot)
    cbar.set_label('TEC Residuals (TECU)')

    if save_option:
        # Save figure
        figname = os.path.join(save_dir, plot_name + '.pdf')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
        print('Figure Residual Map is saved at: ', figname)
    return


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def plot_TEC_residuals_histogram(residuals,
                                 model_units,
                                 dtime,
                                 save_option=False,
                                 save_dir='',
                                 plot_name='TEC_Residuals'):
    """Plot residuals for TEC as a histogram.

    Parameters
    ----------
    residuals : dict
        Dictionary with residuals.
    model_units: dict
        Dictionary with units for each key in residuals.
    dtime : datetime.datetime
        Datetime of the validation day(s).
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='TEC_Residuals.pdf').
    """

    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False)
    if np.size(dtime) > 1:
        fig.suptitle('Residuals, ' + dtime[0].strftime('%Y%m%d') + ' - '
                     + dtime[-1].strftime('%Y%m%d'))
    else:
        fig.suptitle('Residuals, ' + dtime.strftime('%Y%m%d'))

    fig.set_size_inches(3, 3)
    keys_list = 'TEC'

    ax_plot = ax
    ax_plot.set_facecolor('lightgray')
    ax_plot.set_xlabel(keys_list
                       + ' (' + model_units[keys_list] + ')')
    ax_plot.set_ylabel('Number of Obs')
    ax_plot.axvline(x=0, c='black', zorder=2, linestyle='dashed',
                    linewidth=0.5)
    # Plot the histogram of the model residuals
    ax_plot.hist(residuals[keys_list], color='red',
                 bins=50, zorder=1)
    # Making x-axis symmetric
    bound = 30
    xticks = np.arange(-bound, bound + 1, 10)
    ax_plot.set_xticks(xticks)
    ax_plot.set_xlim((-bound, bound))
    plt.tight_layout()

    if save_option:
        # Save figure
        figname = os.path.join(save_dir, plot_name + '.pdf')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
        print('Figure Residuals is saved at: ', figname)

    return
