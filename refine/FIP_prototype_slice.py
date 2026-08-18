#!/usr/bin/env python
# coding: utf-8

import sys
from optparse import OptionParser

try:
    import cupy
except ImportError:
    cupy = None
    print("ERROR: no cupy!")
    sys.exit()

import numpy as np
import cupy
import os
import time

import casacore.tables.table as Table
from ska_sdp_func.grid_data import GridderUvwEsFft as Gridder
from ska_sdp_func.visibility.weighting import get_uv_range, uniform_weights


from astropy.io import fits
from astropy import wcs


def run_ms2dirty(
    do_single,
    do_w_stacking,
    vis,
    model,
    freqs,
    uvw,
    weight,
    flags,
    im_size,
    pixel_size_arcsec,
    epsilon=1e-12,
):
    """Runs ms2dirty to invert visibility slice."""
    print(" ")  # just for separation of debug output
    print(" ")

    vis = vis - model
    vis[np.where(flags > 0)] = 0  # Apply flags.
    vis = vis[:, :, 0] + vis[:, :, 3]

    if do_single:
        vis = vis.astype(np.complex64)
        freqs = freqs.astype(np.float32)
        uvw = uvw.astype(np.float32)
        weight = weight.astype(np.float32)

    pixel_size_rad = pixel_size_arcsec / 3600.0 * np.pi / 180.0

    # Run gridder test on GPU, using cupy arrays.
    if cupy:
        vis_gpu = cupy.asarray(vis)
        freqs_gpu = cupy.asarray(freqs)
        uvw_gpu = cupy.asarray(uvw)
        weight_gpu = cupy.asarray(weight)
        dirty_image_gpu = cupy.zeros([im_size, im_size], uvw.dtype)

        # Create gridder
        gridder = Gridder(
            uvw_gpu,
            freqs_gpu,
            vis_gpu,
            weight_gpu,
            dirty_image_gpu,
            pixel_size_rad,
            pixel_size_rad,
            epsilon,
            do_w_stacking,
        )

        # Run gridder
        gridder.grid_uvw_es_fft(
            uvw_gpu, freqs_gpu, vis_gpu, weight_gpu, dirty_image_gpu
        )

        # Check output
        dirty_image = cupy.asnumpy(dirty_image_gpu)
    return dirty_image


def main():

    parser = OptionParser(usage="%prog [options] msname")
    parser.add_option(
        "--im_size", dest="im_size", default=1024, help="Image size (default = 1024)"
    )
    parser.add_option(
        "--pixel_size_arcsec",
        dest="pixel_size_arcsec",
        default=1.0,
        help="Pixel size (default = 1.0 arcsec)",
    )
    parser.add_option(
        "--do_single",
        dest="do_single",
        default=False,
        help="Make inverts in a single precision (default = False)",
    )
    parser.add_option(
        "--update_CORRECTED_DATA",
        dest="update_CORRECTED_DATA",
        default=False,
        help="Update CORRECTED_DATA column with flagged and model-subtracted visibilities (default = False)",
    )
    parser.add_option(
        "--epsilon",
        dest="epsilon",
        default=1e-12,
        help="Accuracy parameter (default = 1e-12 for FP64, should be 1e-5 for FP32)",
    )
    parser.add_option(
        "--do_w_stacking",
        dest="do_w_stacking",
        default=True,
        help="To perform w-stacking gridding (default = True)",
    )
    parser.add_option(
        "--time_idx_start",
        dest="time_idx_start",
        default=0,
        help="Starting snapshot index (default = 0)",
    )
    parser.add_option(
        "--time_idx_end",
        dest="time_idx_end",
        default=-1,
        help="Ending snapshot index (default = -1)",
    )
    parser.add_option(
        "--weighting",
        dest="weighting",
        default="natural",
        help="Weighting scheme, natural or uniform (default = natural)",
    )
    parser.add_option(
        "--fitsfile",
        dest="fitsfile",
        default=None,
        help="Output FITS file name (default name is based on the input MS name)",
    )

    (options, args) = parser.parse_args()
    do_single = options.do_single
    update_CORRECTED_DATA = options.update_CORRECTED_DATA
    epsilon = float(options.epsilon)
    do_w_stacking = options.do_w_stacking
    im_size = int(options.im_size)
    pixel_size_arcsec = float(options.pixel_size_arcsec)
    time_idx_start = int(options.time_idx_start)
    time_idx_end = int(options.time_idx_end)
    fits_image_filename = options.fitsfile
    weighting = options.weighting
    if weighting != "natural" and weighting != "uniform":
        print("Weighing scheme", weighting, "is neither natural nor uniform, reverting to natural")
        weighting = "natural"

    if len(args) != 1:
        print("Please specify a Measurement Set")
        sys.exit()
    else:
        MSfile = args[0].rstrip("/")

    MSname = os.path.basename(MSfile)
    print("Working with ", MSfile)

    if update_CORRECTED_DATA == True:
        table_all = Table(MSfile, readonly=False)
    else:
        table_all = Table(MSfile)        
    ColNames = table_all.colnames()
    for col in ColNames:
        print(col)

    table_freq = Table(MSfile + "/SPECTRAL_WINDOW")
    for col in table_freq.colnames():
        print(col)
    freqs = table_freq.getcol("CHAN_FREQ")[0]
    
    table_source = Table(MSfile+"/SOURCE")
    for col in table_source.colnames():
        print(col)
    direction = table_source.getcol("DIRECTION")
    try:
        print("Coordinates:", direction[0], direction.shape)
        source_name = table_source.getcol("NAME")
        print(source_name)
    except:
        print("WARNING: no source direction and name, using FILED/PHASE_DIR table.")
        table_field = Table(MSfile+"/FIELD")
        for col in table_field.colnames():
            print(col)
        direction = table_field.getcol("PHASE_DIR")[0]
        print("Coordinates:", direction[0], direction.shape)
        source_name = "OSKAR model"
        print(source_name)

    # Find unique times and their indices
    mstime = table_all.getcol("TIME")
    mstime_unique, time_idx = np.unique(mstime, return_index=True)

    # Reset the first snapshot index if required
    if time_idx_start < 0:
        time_idx_start = 0
    # Reset the last snapshot index if required
    if time_idx_end < 0 or time_idx_end >= len(mstime_unique):
        time_idx_end = len(mstime_unique) - 1

    tstart = time.time()
    image_cube_nz = int(time_idx_end - time_idx_start + 1)
    dirty_image_cube = np.zeros((image_cube_nz, im_size, im_size))
    
    if weighting == "uniform":
        # Read global UVW array
        uvw_all = table_all.getcol("UVW")
        print(uvw_all.shape)
        # Find global UV range for all snapshots
        max_abs_uv = get_uv_range(uvw_all[np.newaxis,:,:], freqs)
        # Free UVW array memory
        uvw_all = None

    # Loop over unique times and create inverts for each snapshot
    #for i in range(len(mstime_unique)):
    for i in range(time_idx_start, time_idx_end+1):
        image_cube_i = i - time_idx_start
        istart = time_idx[i]
        if i != len(mstime_unique) - 1:
            iend = time_idx[i + 1] - 1
        else:
            iend = len(mstime) - 1
        number_of_rows_to_read = iend - istart + 1
        vis_snap = table_all.getcol("DATA", istart, number_of_rows_to_read).astype(
            np.complex128
        )

        try:
            model_snap = table_all.getcol(
                "MODEL_DATA", istart, number_of_rows_to_read
            ).astype(np.complex128)
        except RuntimeError:
            model_snap = np.zeros(vis_snap.shape)

        uvw_snap = table_all.getcol("UVW", istart, number_of_rows_to_read)
        if weighting == "natural":
            weight_snap = np.ones(vis_snap.shape)
        else: # weigthing = "uniform"
            weight_snap = np.zeros(vis_snap.shape)
            grid_uv = np.zeros((im_size, im_size))
            uniform_weights(uvw_snap[np.newaxis,:,:], freqs, max_abs_uv, grid_uv, weight_snap[np.newaxis,:,:,:])
    
        flags_snap = table_all.getcol("FLAG", istart, number_of_rows_to_read)

        print(
            "\n",
            i,
            mstime_unique[i],
            istart,
            iend,
            number_of_rows_to_read,
            vis_snap.shape,
            freqs.shape,
            uvw_snap.shape,
            "\n",
        )
        dirty_image = run_ms2dirty(
            do_single,
            do_w_stacking,
            vis_snap,
            model_snap,
            freqs,
            uvw_snap,
            weight_snap,
            flags_snap,
            im_size,
            pixel_size_arcsec,
            epsilon=epsilon,
        )

        norm = np.sum(weight_snap) / 4
        dirty_image = dirty_image / norm
        dirty_image = (np.fliplr(dirty_image)).T
        dirty_image_cube[image_cube_i, :, :] = dirty_image

        # Write flagged data-model data back into "CORRECTED_DATA"
        if update_CORRECTED_DATA == True:
            corr_vis = vis_snap - model_snap
            corr_vis[np.where(flags_snap > 0)] = 0  # Apply flags.
            try:
                table_all.putcol("CORRECTED_DATA", corr_vis, istart, number_of_rows_to_read)
            except RuntimeError:
                print("CORRECTED_DATA does not exist in the MS, can not update the column!")

    telapsed = time.time() - tstart
    print("\nElapsed time = ", telapsed, "sec")
    if image_cube_nz > 1:
        time_per_snap = telapsed / (image_cube_nz -1)
        print("Time per snap = ", time_per_snap, "sec")

    # Write the results into the data cube FITS file data(ntimes, im_size, im_size)
    if fits_image_filename == None:
        fits_image_filename = MSname[:-3] + "_" + str(im_size) + "p_t" + str(time_idx_start) + "-" + str(time_idx_end) + "_" + weighting + ".fits"

    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=3)

    # Create fits file header
    nx = im_size
    ny = im_size
    cdelt = pixel_size_arcsec/3600./180.*np.pi # in rad
    tstart1 = mstime_unique[time_idx_start] 
    ntime1 = image_cube_nz
    if ntime1 != 1:
        tdelt1 = (mstime_unique[time_idx_end] - tstart1)/(ntime1-1)
    else:
        tdelt1 = 1.0	
    w.wcs.crpix = [nx/2, ny/2, 1.0]
    w.wcs.cdelt = np.array([-cdelt, cdelt, tdelt1])

    w.wcs.crval = [direction[0][0], direction[0][1], tstart1]
    w.wcs.cunit = ["rad", "rad", "s"]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "TIME"]

    header = w.to_header()
    hdumodel = fits.PrimaryHDU(data=dirty_image_cube.astype(np.float32), header=header)
    hdumodel.header['BUNIT'] = 'K'
    hdumodel.header['FREQ'] = np.mean(freqs)
    hdumodel.header['OBJECT'] = source_name[0]
    hdumodel.writeto(fits_image_filename, overwrite=True)


if __name__ == "__main__":

    main()
