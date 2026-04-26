project('FastImagingPipe',
    ['cuda', 'c', 'cpp'],
    version:         '0.1',
    meson_version:   '>= 1.10.0',
    license:         'CC-BY-4.0',
    license_files:   ['LICENSE.txt'],
    default_options: [
        'c_std=gnu11,gnu99,c11,c99',
        'cpp_std=gnu++14,gnu++11,c++14,c++11',
        'cuda_std=c++14',
        'warning_level=3',
        'b_ndebug=if-release',
        'buildtype=release',
        'cudart=shared',
    ],
)

add_project_arguments     ('-Wno-deprecated-gpu-targets',   language: 'cuda')
add_project_link_arguments('-Wno-deprecated-gpu-targets',   language: 'cuda')


cuda_mod      = import('unstable-cuda')
nvcc_version  = meson.get_compiler('cuda').version()
cuda_target   = get_option('gpu_arch')
cuda_flags    = cuda_mod.nvcc_arch_flags   (nvcc_version, cuda_target)
cuda_readable = cuda_mod.nvcc_arch_readable(nvcc_version, cuda_target)
message('Building with NVCC '+nvcc_version)
message('Building for architectures ' + ' '.join(cuda_readable))



cudart  = get_option('cudart') == 'shared' ? 'cudart' : 'cudart_static'
cudart  = get_option('cudart') == 'none' ? [] : [cudart]
curand  = dependency('cuda',    modules: ['curand']+cudart,  disabler: true)
cufft   = dependency('cuda',    modules: ['cufft']+cudart,   required: true)
cfitsio = dependency('cfitsio',                              required: false,
                     fallback: ['cfitsio', 'cfitsio_dep'],
                     default_options: ['warning_level=0'])



subdir('src')
