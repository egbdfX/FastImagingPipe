# Collect source code for FIP
subdir('tools')


# Build targets
##  Legacy tool "sharedlibrary_gpu" that is, in fact, an executable
fip = executable('sharedlibrary_gpu',
    files('pipe_interface.cpp',
          'FIPkernels.cu'),
    cuda_args:     cuda_flags + ['-Xcompiler=-fPIC', '--use_fast_math', '-lineinfo'],
    dependencies:  [cfitsio, cufft],
    install:       false,
)


## Replacement tool "fip" with verbs interface.
fip = executable('fip',
    fip_tools_src + files(
        'utils.c',
        'FIPkernels.cu',
    ),
    cuda_args:     cuda_flags + ['-Xcompiler=-fPIC', '--use_fast_math', '-lineinfo'],
    dependencies:  [popt, cfitsio, cufft],
    install:       false,
)
