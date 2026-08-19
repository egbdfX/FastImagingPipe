# Fast Imaging Pipeline

We develop a GPU-accelerated Fast Imaging Pipeline (FIP) for transient detection and localisation in radio astronomy. Please see our paper in Section [Reference](https://github.com/egbdfX/FastImagingPipe/tree/main#reference) for more information. The FIP consists of two components: Transient-Oriented Imager ([TOI](https://github.com/egbdfX/SVDimager)) and Fast Imaging Trigger ([FITrig](https://github.com/egbdfX/FastImagingTrigger)).

## User guidance

**Step 1:**
Make sure CUDA 12.6, CFITSIO, casacore, Python, uv, Meson, CMake, Ninja, and NVCC are available.

**Step 2:**
Build the FIP command-line programme with Meson. See [here](https://github.com/egbdfX/FastImagingPipe/blob/engineering-archive/INSTALL.md) for more details.

```
uv run --with meson,cmake,ninja bash -c 'meson setup . build/Meson; meson compile -j1 -C build/Meson'
```

After building, the available commands can be checked by:

```
build/Meson/src/fip pipe --help
```
for pipeline running, and 
```
build/Meson/src/fip image --help
```
for image-only mode.

**Step 3:**
Run the GPU FIP to produce the 3D tLISI FITS cube:

```
build/Meson/src/fip pipe -vv -i /path/to/your/MeasurementSet.ms -o output.fits --cell-size {radians} --image-size {number of pixels}
```

This writes ```output.fits``` inside the ```FastImagingPipe``` directory.

**Step 4:**
Return to the outer run directory and build the CUDA harmonic trigger helper:

```
nvcc -O3 -std=c++11 -arch=sm_{yourGPU} -I$HOME/Libraries/cfitsio/include FastImagingPipe/refine/iqa_harfft_cuda.cu -L$HOME/Libraries/cfitsio/lib -lcfitsio -lcufft -lcudart -o iqa_harfft_cuda
```
**Step 5:**
Run the end-to-end periodic/non-periodic (without/with the ```--non-periodic```) source localisation refinement:

```
uv run -p 3.12 --with='numpy,astropy' FastImagingPipe/refine/fip_tlisi_pipeline.py ./FastImagingPipe/output.fits --non-periodic --r-orientation auto --fip-bin ./FastImagingPipe/build/Meson/src/fip --max-orientation-shift 16 --result-fits z_result.fits --ms /path/to/your/MeasurementSet.ms --image-size {number of pixels} --cell-size {radians}
```

Here, ```./FastImagingPipe/output.fits``` is the 3D tLISI cube from Step 3, ```--ms``` is the Measurement Set, ```--r-orientation auto``` automatically compares the FIP image against the SKA-SDP image to choose the image orientation, and ```--result-fits``` names the 2D tLISI result map.

**Step 6:**
The pipeline writes the main output files:

```z_result.fits``` is the 2D significance map.

```difference_image.fits``` is the FITS difference image used for source localisation.

```fip_image_t*.fits``` is the image produced by TOI in the outer run directory.

```FastImagingPipe/refine/MS_{ImageSize}p_t*-*_natural.fits``` is the corresponding SKA-SDP image produced by ```FIP_prototype_slice.py``` inside ```FastImagingPipe/refine```.

The terminal output prints detected transient positions as a table.

## Example
See ```fipexample.sh``` for an example.

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
