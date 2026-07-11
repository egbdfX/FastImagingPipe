# Collect source code for FIP
subdir('tools')


# Build targets
## Replacement tool "fip" with verbs interface.
fip = executable('fip',
    fip_tools_src + files(
        'utils.c',
        'FIPkernels.cu',
    ),
    cuda_args:     cuda_flags + ['-Xcompiler=-fPIC', '--use_fast_math', '-lineinfo'],
    dependencies:  [popt, cfitsio, cufft, nppc, nppial, nppist, nppitc],
    install:       false,
)
