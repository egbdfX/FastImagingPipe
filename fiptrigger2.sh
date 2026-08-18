#!/usr/bin/env bash
set -euo pipefail

module load cuda/12.6.0
export UV_CACHE_DIR=/network/scratch/x/xla/FIP/olexa/runtest/debug/.uv-cache
export CPATH=$HOME/Libraries/cfitsio/include
export LIBRARY_PATH=$HOME/Libraries/cfitsio/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$HOME/Libraries/cfitsio/lib
export CPATH="$CPATH:/network/scratch/b/bilaniuo/rootfs/usr/include"
export LIBRARY_PATH="$LIBRARY_PATH:/network/scratch/b/bilaniuo/rootfs/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/network/scratch/b/bilaniuo/rootfs/usr/lib/x86_64-linux-gnu"

module load cudatoolkit/12.6
export CPATH="$CPATH:/network/scratch/b/bilaniuo/rootfs/usr/include"
export LIBRARY_PATH="$LIBRARY_PATH:/network/scratch/b/bilaniuo/rootfs/usr/lib/x86_64-linux-gnu"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}:/network/scratch/b/bilaniuo/rootfs/usr/lib/x86_64-linux-gnu/pkgconfig"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/network/scratch/b/bilaniuo/rootfs/usr/lib/x86_64-linux-gnu"

git clone -b engineering 'git@github.com:egbdfX/FastImagingPipe.git'
cd FastImagingPipe

if [[ ! -x build/Meson/src/fip || src/tools/main_pipe.cpp -nt build/Meson/src/fip || src/tools/main_image.cpp -nt build/Meson/src/fip || src/cuda_pipeline.c -nt build/Meson/src/fip || include/fip/cuda_pipeline.h -nt build/Meson/src/fip ]]; then
    rm -Rf build/Meson
    uv run --with meson,cmake,ninja bash -c 'meson setup . build/Meson; meson compile -j1 -C build/Meson'
fi
build/Meson/src/fip pipe --help
build/Meson/src/fip image --help

cp -a /network/scratch/x/xla/GX339.ms  $SLURM_TMPDIR/

build/Meson/src/fip pipe -vv -i $SLURM_TMPDIR/GX339.ms -o output.fits --cell-size 0.000004276057 --image-size 4096

cd ..
nvcc -O3 -std=c++11 -arch=sm_80 -I$HOME/Libraries/cfitsio/include FastImagingPipe/refine/iqa_harfft_cuda.cu -L$HOME/Libraries/cfitsio/lib -lcfitsio -lcufft -lcudart -o iqa_harfft_cuda
uv run -p 3.12 --with='numpy,astropy' FastImagingPipe/refine/fip_tlisi_pipeline.py ./FastImagingPipe/output.fits --non-periodic --r-orientation auto --fip-bin ./FastImagingPipe/build/Meson/src/fip --max-orientation-shift 16 --result-fits tlisi_result.fits --ms $SLURM_TMPDIR/GX339.ms --image-size 4096 --cell-size 0.000004276057
