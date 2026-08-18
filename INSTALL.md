# Installation

## 0. Preliminaries

- *All instructions are means to an end. They are not the ends themselves.*

- *Their adaptation is often necessary to a particular context. Thinking on two feet is required.*

- In compiling and running C, C++ and CUDA software, the unspoken secret is that:

    - Environment variable `$PATH` is where the _shell_ (and any tools that execute other programs) searches for named _programs_.
    - Environment variable `$CPATH` is a list of directories the _compiler_ adds to its search paths for headers (`-I /path/to/include`, looking for the `name.h` in `#include "name.h"`).
    - Environment variable `$LIBRARY_PATH` is a list of directories the _compiler_ adds to the _linker_'s search path for libraries (`-L /path/to/lib`, looking for `libname.so` given flag `-lname`).
    - Environment variable `$LD_LIBRARY_PATH` is a list of directories the _dynamic loader_ adds **at runtime** to the search path for libraries. It is not necessarily, or even commonly, the same value as `$LIBRARY_PATH`; One is compile-time and the other is run-time.
    - Other means of finding libraries at runtime exist.

- In loading software on HPC environments with modules:

    - `module load ...` and similar often manipulate the above environment variables to make software available/unavailable.
    - The names of modules do not always line up with the name of any particular library bundled in it.


## 1. Requirements

The Fast Imaging Pipeline is a mixed C, C++ and CUDA codebase. It requires the following to  build:

- **CUDA Toolkit.** Versions as recent as 13.3 and as old as 10.2 are believed to work.
    - `nvcc` must be visible in the `$PATH`.
- **GNU C & C++ compilers.** Any semi-modern version should work, but the CUDA Toolkit and CASACore dependencies may impose additional requirements. The codebase uses only such features of C++ as CASACore compels one to use.
    - Clang/LLVM have not been tested, but are likely to work.
- **Meson.** Version >= 1.10.0. This is the build system used by FIP.
    - _Transitively:_
        - **Python.** Version >= 3.7. Meson 1.12.0+ requires >= 3.10. Meson in implemented in Python. FIP itself does not require it.
        - **Ninja.** Version >= 1.8.2. Used by Meson for build target generation. Not used directly by FIP.
- **libpopt.** Practically any version. For parsing command-line arguments.
    - If not detected automatically, Meson automatically selects and compiles from source bundled version **1.19**.
- **CFITSIO.** Practically any version. For reading and writing FITS files.
    - If not detected automatically, Meson automatically selects and compiles from source bundled version **4.7.0**.
- **CASACore.** Any version >= 3.0.0 is believed to work, but only 3.4.0-3.7.1 has been tested. For reading astronomy data stored in MeasurementSet tables.
    - _NOTE:_ Versions 3.7.0+ of CASACore require C++17, and therefore also a C++ compiler that supports the C++17 standard and that can be configured in that mode. Historically:

        | GCC Version  | Default C++ Standard |
        |:------------:|:--------------------:|
        |   <6.1       |     `-std=c++98`     |
        |   *None* (a) |     `-std=c++11`     |
        |   6.1-10     |     `-std=c++14`     |
        |    11-15     |     `-std=c++17`     |
        |    16+       |     `-std=c++20`     |

      (a): GCC 4.8.1 was the first feature-complete implementation of C++11, but did not default to that standard.
- **CMake.** Optional. Only required if Meson uses fallback `libpopt` or `cfitsio` builds from source (both are themselves CMake projects).


## 2. Getting Started (Full)

1.  **Get source code (if not already possessed)**

        git clone -b engineering 'git@github.com:egbdfX/FastImagingPipe.git'
        cd FastImagingPipe

2.  ***(If using modules:)*** **Load any available modules for relevant dependencies.**

        module load cuda        # Anything >= 10.2 and maybe even older
        module load gcc         # Anything modern
        module load cfitsio     # Anything modern (optional)
        module load meson       # Anything >= 1.10.0
        module load casacore    # Anything >= 3.0.0

    Adapt module names to clusters. Look at alternative names, like "`compilers`", "`cmake`", "`ninja`", "`build`", ...

    If Meson is unavailable from any modules, create a virtual environment with

        virtualenv  .build-venv
        .           .build-venv/bin/activate
        pip install meson ninja cmake

3.  **Configure.**

    A Meson build must first be configured, then built. Meson _*requires* out-of-place builds_.

    This is where you configure build options such as target GPU architecture.

        # Tweak CUDA compute capabilities here if wanted, or leave at default
        #     '7.0 7.5 8.0 8.6 8.9 9.0+PTX'
        # which is adequate for most clusters.
        meson setup -Dgpu_arch="8.0 9.0"  .  build/Meson

    Pay attention to options selected and software versions detected here.

    `-Dgpu_arch="8.0 9.0"` is adequate for A100 and H100. For other GPUs, consult the
    [NVIDIA Compute Capability table](https://developer.nvidia.com/cuda/gpus).

4.  **Build.**

        meson compile           -C           build/Meson

5.  **Test in build directory.**

        build/Meson/src/fip pipe --help

    Should run without dynamic linker problems from the build directory and print
    supported flags.

    This is currently the conventional way to use FIP.

6.  **Install.**

    ***WARNING***, not currently advised to do this.

        meson install           -C           build/Meson


## 3. Getting Started (Quick)

For those who don't want to fiddle with environments, build systems and more,
you can use `uv` to create a throwaway environment to quickly compile the code.

    # One-time, if not installed UV already:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Download
    git clone -b engineering 'git@github.com:egbdfX/FastImagingPipe.git'
    cd FastImagingPipe
    
    # Configure and build
    uv run --with meson,cmake,ninja bash -c \
       'meson setup -Dgpu_arch="8.0 9.0" . build/Meson; meson compile -C build/Meson'
    
    # Run
    build/Meson/src/fip pipe --help

You still need at least the CUDA Toolkit, GCC and CASACore visible in the environment, or
from your package manager.
