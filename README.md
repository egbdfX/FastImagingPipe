# Fast Imaging Pipeline

We have developed a GPU-accelerated fast imaging pipeline for imaging and transient detection in radio astronomy. Please see our paper in Section [Reference](https://github.com/egbdfX/FastImagingPipe/tree/main#reference) for more information.

## User guidance

**Step 1:**
Make sure GCCcore, CUDA, and CFITSIO are avaiable. If you see a warning saying ```/usr/bin/ld.gold: warning: /apps/system/easybuild/software/GCCcore/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/crtbegin.o: unknown program property type 0xc0010002 in .note.gnu.property section```, you would need to make sure Python is also available.

**Step 2:**
If you do not need reprojection in the imaging, download the files in 'NoReproj'; otherwise, download the files in 'WithReproj' (will release soon).

Run the Makefile by ```make```. Note that this Makefile is written for NVIDIA H100. If you are using other GPU, you would need to make sure the CUDA arch is matching.

**Step 3:**
Run the code by ```./sharedlibrary_gpu Visreal0.fits Visimag0.fits Bin0.fits Vin0.fits 4096 2080 50000000 0.1310 0.0 0.0 dirty0.fits```, where ```dif1.fits``` and ```dif2.fits``` are the two difference images (FITS files), and ```snap1.fits``` is the reference snapshot image (FITS file). The three input images should have the same size.

**Step 4:**
The code will output a FITS file named ```output_tLISI.fits```, which is the output tLISI matrix.

## Test
If you want to test the code, please download everything from the 

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
