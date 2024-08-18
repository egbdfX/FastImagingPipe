# Fast Imaging Pipeline

We have developed a GPU-accelerated fast imaging pipeline for imaging and transient detection in radio astronomy. Please see our paper in Section [Reference](https://github.com/egbdfX/FastImagingPipe/tree/main#reference) for more information.

## User guidance

**Step 1:**
Make sure GCCcore, CUDA, and CFITSIO are avaiable. If you see a warning saying ```/usr/bin/ld.gold: warning: /apps/system/easybuild/software/GCCcore/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o: unknown program property type 0xc0010002 in .note.gnu.property section```, you would need to make sure Python is also available.

**Step 2:**
If you do not need reprojection in the imaging, download the files in 'NoReproj'; otherwise, download the files in 'WithReproj' (will release soon).

**Step 3:**
Run the Makefile by ```make```. Note that this Makefile is written for NVIDIA H100. If you are using other GPU, you would need to make sure the CUDA arch is matching.

**Step 4:**
Run the code by ```./sharedlibrary_gpu Visreal_input.fits Visimag_input.fits B_input.fits V_input.fits Image_Size Number_of_Baselines Frequency UV_Scale Phase_RA Phase_Dec Output_Name.fits```, where ```Visreal_input.fits```, ```Visimag_input.fits''', ```B_input.fits''', and ```V_input.fits``` are input files (in FITS) for real components of visibilities, imagery components of visibilities, SVDed baseline matrix, and the V matrix in the SVD. The rest arguments are as their names suggested, where Image_Size is an integer (e.g., if you input 100, it means the image size is $100 \times 100$ pixels), Number_of_Baselines is an integer, Frequency is in unit of Hz, UV_Scale is in unit of radian, Phase_RA and Phase_Dec are the phase centre in unit of degree, and the last argument is a file name of the output ended with '.fits'.

**Step 5:**
The code will output a FITS file named ```Output_Name.fits``` (as user defined), which is the output snapshot.

## Test
If you want to test the code, please download the files from 'ExampleInput' of the corresponding reprojection method. Run the code by ```./sharedlibrary_gpu Visreal0.fits Visimag0.fits Bin0.fits Vin0.fits 4096 2080 50000000 0.1310 0.0 0.0 dirty0.fits'''. You should obtain a dirty0.fits. If you open it (by SAOImageDS9, Fv or MATLAB etc), you will see a simulated sky brightness distribution of regular distributed sources. 

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
