#!/usr/bin/env python
# ---------------------------------------------------------
# Distribution statement A. Approved for public release.
# Distribution is unlimited.
# This work was supported by the Office of Naval Research.
# ---------------------------------------------------------
"""This library contains components visualization routines for PyIRI."""

import matplotlib.pyplot as plt
import numpy as np
import os


def PyVALION_plot_ionosondes():
    """Plot magnetic dip latitude.

    Parameters
    ----------
    mag : dict
        Dictionary output of IRI_monthly_mean_parameters.
    alon : array-like
        Flattened array of geo longitudes in degrees.
    alat : array-like
        Flattened array of geo latitudes in degrees.
    alon_2d : array-like
        2-D array of geo longitudes in degrees.
    alat_2d : array-like
        2-D array of geo latitudes in degrees.
    plot_dir : str
        Direction where to save the figure.
    plot_name : str
        Output name, without directory, for the saved figure
        (default='PyIRI_mag_dip_lat.pdf')

    """
    figname = os.path.join(plot_dir, plot_name)
    fig, ax = plt.subplots(1, 1)
    ax.set_facecolor('grey')
    ax.set_xlabel('Geo Lon (°)')
    ax.set_ylabel('Geo Lat (°)')
    z = np.reshape(mag['mag_dip_lat'], alon_2d.shape)
    levels = np.linspace(-90, 90, 40)
    levels_cb = np.linspace(-90, 90, 5)
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.xticks(np.arange(-180, 180 + 45, 45))
    plt.yticks(np.arange(-90, 90 + 45, 45))
    contour = ax.contourf(alon_2d, alat_2d, z, levels=levels)
    for c in contour.collections:
        c.set_edgecolor("face")
    cbar = fig.colorbar(contour, ax=ax, ticks=levels_cb)
    cbar.set_label('Mag Dip Lat (°)')
    plt.title('Alt = 300 km')
    plt.savefig(figname)
    return

