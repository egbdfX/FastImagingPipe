# Fast Imaging Pipeline

We develop a GPU-accelerated Fast Imaging Pipeline (FIP) for transient detection and localisation in radio astronomy. Please see our paper in Section [Reference](https://github.com/egbdfX/FastImagingPipe/tree/main#reference) for more information. The FIP consists of two components: Transient-Oriented Imager ([TOI](https://github.com/egbdfX/SVDimager)) and Fast Imaging Trigger ([FITrig](https://github.com/egbdfX/FastImagingTrigger)).

There are two executable stages:

1. `svd_preprocess_gpu`: reads CASA Measurement Sets and writes inputs for the imager in FITS files (solely as data containers rather than image files).
2. `sharedlibrary_gpu`: reads the FITS inputs, runs the CUDA imaging and tLISI kernels, and writes the final tLISI matrix into a FITS file.

The older Python preprocessing script, `SVD_MFS.py`, is still included as a reference implementation. The maintained preprocessing path is the GPU/C++ implementation in `SVDPreprocessMain.cpp` and `SVDPreprocessKernels.cu`.

## Dependencies

Build-time dependencies:

- NVIDIA CUDA toolkit, including cuFFT, cuBLAS, and cuSOLVER
- CFITSIO
- casacore, for `svd_preprocess_gpu`
- C++ compiler with C++11 support
- GNU Make

Optional Python dependencies for the legacy `SVD_MFS.py` script:

- `casacore`
- `scipy`
- `numpy`
- `scikit-learn`
- `astropy`

The Makefiles assume:

- `CUDA_HOME=/usr/local/cuda` unless overridden
- `CFITSIO_HOME=$(HOME)/Libraries/cfitsio`
- `CASACORE_HOME=/usr` for `Makefile.gpu`

Override these paths on the `make` command line if your installation differs.

## Build

Build the GPU preprocessing executable:
```sh
make -f Makefile.gpu CUDA_ARCH=? svd_preprocess_gpu
```

Build the GPU pipeline executable:
```sh
make -f Makefile CUDA_ARCH=? sharedlibrary_gpu
```

## Preprocess Measurement Sets

The GPU preprocessor reads one or more Measurement Sets containing `UVW`, `DATA`, `FLAG`, `SPECTRAL_WINDOW/CHAN_FREQ`, and optionally `WEIGHT_SPECTRUM`.

**CPU version:** 
```
python SVD_MFS.py
```

**GPU version:**

To treat each distinct Measurement Set `TIME` value as a separate snapshot:
```
./svd_preprocess_gpu --group-by-time --snapshot-workers 16 --combined-output name_input.fits /path/to/ms
```

Options:

- `--output-prefix PREFIX`: prefix for the separate FITS outputs. If omitted, outputs are written as `Bin.fits`, `Vin.fits`, `Visreal.fits`, and `Visimag.fits`.
- `--combined-output FILE`: write a combined multi-extension FITS file. If omitted, this is `PREFIXinput.fits`. This is not used in the archive branch.
- `--group-by-time`: group rows in each Measurement Set by `TIME`; each group becomes one snapshot.
- `--snapshot-workers N`: number of snapshot workers, controlling how many snapshots are preprocessed concurrently. If omitted, the code uses up to 4 workers, limited by hardware concurrency and snapshot count.

Outputs:

`PREFIXBin.fits` contains centred 2D baseline coordinates, shape `2 x total_samples`; `PREFIXVin.fits` includes per-snapshot 3 x 3 transformation matrices; `PREFIXVisreal.fits` contains weighted real visibility values; `PREFIXVisimag.fits` includes weighted imaginary visibility values; combined FITS file with extensions `TRANSFORMATION MATRICES`, `VISIBILITIES`, and `R-COORDINATES`.

If snapshots have different sample counts, the preprocessor zero-pads shorter snapshots to the maximum snapshot size before writing aggregate outputs.

## Run the Imaging Pipeline

Run `sharedlibrary_gpu` with the separate FITS files generated above:

```sh
./sharedlibrary_gpu Visreal.fits Visimag.fits Bin.fits Vin.fits Image_Size Number_of_Baselines Cell_Size Number_of_Snapshots Tile_Size Output_Name.fits
```

Here, ```Visreal_input.fits```, ```Visimag_input.fits```, ```B_input.fits```, and ```V_input.fits``` are the input files (in FITS format) corresponding to the real components of visibilities, the imaginary components of visibilities, the (centred) SVDed baseline matrix, and the V matrix in the SVD, respectively. The remaining arguments are as their names suggest, where ```Image_Size``` is an integer (e.g., if you input 100, it means the image size is $100 \times 100$ pixels), ```Number_of_Baselines``` is an integer, ```Cell_Size``` is in units of radians, ```Number_of_Snapshots``` is an integer, ```Tile_Size``` is an integer (e.g., if you input 20, it means the tile size is $20 \times 20$ pixels), and the last argument is the name of the output file which should end with '.fits'. The code will output a FITS file named ```Output_Name.fits``` (as user defined), which is the output tLISI matrix.

## Contact
If you have any questions or need further assistance, please feel free to contact at [egbdfmusic1@gmail.com](mailto:egbdfmusic1@gmail.com).

## Reference

**When referencing this code, please cite our related paper:**

X. Li, K. Adámek, O. Bilaniuk, V. Stolyarov, W. Armour, "[FIP-TOI: Fast Imaging Pipeline for Pulsar Localisation with a Transient-Oriented Radio Astronomical Imager](https://arxiv.org/abs/2512.06254)," 2026.

## License

Shield: [![BSD 3-Clause][bsd-3-shield]][bsd-3]

This work is licensed under a
[BSD 3-Clause License][bsd-3].

[bsd-3]: https://opensource.org/licenses/BSD-3-Clause
[bsd-3-shield]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
