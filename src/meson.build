# Build target
fip = executable('fip',
    files('pipe_interface.cpp',
          'FIPkernels.cu'),
    cuda_args:     cuda_flags + ['-Xcompiler=-fPIC', '--use_fast_math', '-lineinfo'],
    dependencies:  [cfitsio, cufft],
    install:       false,
)
