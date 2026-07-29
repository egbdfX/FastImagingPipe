#
# Determine if version scripts are supported.
#
# We simply make a hard assumption that certain linkers support the flag
# --version-script.
#
has_version_script = ld in ['ld.bfd', 'ld.gold', 'ld.lld', 'ld.mold']


# Build targets

## Library
libfip_soversion                = '0'
libfip_version_script           = files('libfip.ver')
libfip_version_script_path      = libfip_version_script[0].full_path()
libfip_version_script_link_args = not has_version_script ? [] : [
    '-Wl,--version-script='+libfip_version_script[0].full_path(),
]

libfip = shared_library('fip',
    files(
        'cpu_kernels.c',
        'cuda_kernels.cu',
        'cuda_pipeline.c',
    ) + visibility_h,
    gnu_symbol_visibility: 'hidden',
    include_directories:   [libfip_inc],
    c_args:                ['-DLIBFIP_COMPILATION'],
    cpp_args:              ['-DLIBFIP_COMPILATION'],
    cuda_args:             ['-DLIBFIP_COMPILATION'] + cuda_flags +
                           ['-Xcompiler=-fPIC', '--use_fast_math', '-lineinfo'],
    dependencies:          [cufft, nppc, nppist],
    link_language:         'cpp',
    link_depends:          libfip_version_script,
    link_args:             libfip_version_script_link_args,
    soversion:             libfip_soversion,
    install:               true,
)


## Replacement tool "fip" with verbs interface.
subdir('tools')
fip = executable('fip',
    fip_tools_src,
    gnu_symbol_visibility: 'hidden',
    dependencies:          [popt, cfitsio],
    link_with:             [libfip],
    include_directories:   [libfip_inc, include_directories('tools/')],
    install:               true,
)
