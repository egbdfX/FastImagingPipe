# Fast Imaging Pipeline

We have developed a GPU-accelerated Fast Imaging (FI) Pipeline for transient detection in radio astronomy. Please see our paper in Section [Reference](https://github.com/egbdfX/FastImagingPipe/tree/main#reference) for more information. The FI pipeline consists of two components: SVD imager (see [SVD Imager](https://github.com/egbdfX/SVDimager)) and FI trigger (see [Fast Imaging Trigger](https://github.com/egbdfX/FastImagingTrigger)).

## User guidance

**Step 1:**
Make sure GCCcore, CUDA, and CFITSIO are avaiable. If you see a warning saying ```/usr/bin/ld.gold: warning: /apps/system/easybuild/software/GCCcore/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o: unknown program property type 0xc0010002 in .note.gnu.property section```, you would need to make sure Python is also available.

**Step 2:**
Run the Makefile by ```make```. Note that this Makefile is written for NVIDIA H100. If you are using other GPUs, you would need to make sure the CUDA arch is matching.

**Step 3:**
Run the code by executing the following command:

```./sharedlibrary_gpu Visreal_input.fits Visimag_input.fits B_input.fits V_input.fits Image_Size Number_of_Baselines Frequency Cell_Size Number_of_Snapshots Tile_Size Output_Name.fits```.

Here, ```Visreal_input.fits```, ```Visimag_input.fits```, ```B_input.fits```, and ```V_input.fits``` are the input files (in FITS format) corresponding to the real components of visibilities, the imagery components of visibilities, the (centred) SVDed baseline matrix, and the V matrix in the SVD, respectively. The remaining arguments are as their names suggest, where ```Image_Size``` is an integer (e.g., if you input 100, it means the image size is $100 \times 100$ pixels), ```Number_of_Baselines``` is an integer, ```Frequency``` is in units of Hz, ```Cell_Size``` is in units of radians, ```Number_of_Snapshots``` is an integer, ```Tile_Size``` is an integer (e.g., if you input 20, it means the tile size is $20 \times 20$ pixels), and the last argument is the name of the output file which should end with '.fits'.

**Step 4:**
The code will output a FITS file named ```Output_Name.fits``` (as user defined), which is the output tLISI matrix.

## Contact
If you have any questions or need further assistance, please feel free to contact at [egbdfmusic1@gmail.com](mailto:egbdfmusic1@gmail.com).

## Reference

**When referencing this code, please cite our related paper:**

X. Li, K. Adámek, M. Giles, W. Armour, "Fast imaging pipeline for transient detection with GPU acceleration," 2024.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
