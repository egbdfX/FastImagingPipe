project('FastImagingPipe',
    ['cuda', 'c', 'cpp'],
    version:         '0.1',
    meson_version:   '>= 1.10.0',
    license:         'BSD-3-Clause',
    license_files:   ['LICENSE'],
    default_options: [
        'c_std=gnu11,c11,gnu99,c99',
        'cpp_std=gnu++17,c++17,gnu++14,c++14,gnu++11,c++11',
        'cuda_std=c++14',
        'warning_level=3',
        'b_ndebug=if-release',
        'buildtype=release',
        'cudart=shared',
    ],
)

add_project_arguments     ('-Wno-deprecated-gpu-targets',   language: 'cuda')
add_project_link_arguments('-Wno-deprecated-gpu-targets',   language: 'cuda')



fs            = import('fs')
py_mod        = import('python')
snippets      = import('snippets')
cuda_mod      = import('unstable-cuda')



py            = py_mod.find_installation()
c             = meson.get_compiler('c')
cpp           = meson.get_compiler('cpp')
nvcc          = meson.get_compiler('cuda')
ld            = cpp.get_linker_id()
nvcc_version  = nvcc.version()
cuda_target   = get_option('gpu_arch')
cuda_flags    = cuda_mod.nvcc_arch_flags   (nvcc_version, cuda_target)
cuda_readable = cuda_mod.nvcc_arch_readable(nvcc_version, cuda_target)

message('Building with NVCC '+nvcc_version)
message('Building for architectures ' + ' '.join(cuda_readable))



cudart  = get_option('cudart') == 'shared' ? 'cudart' : 'cudart_static'
cudart  = get_option('cudart') == 'none' ? [] : [cudart]
curand  = dependency('cuda',    modules: ['curand'] + cudart, disabler: true)
cufft   = dependency('cuda',    modules: ['cufft']  + cudart, required: true)
nppc    = dependency('cuda',    modules: ['nppc']   + cudart, required: true)
nppist  = dependency('cuda',    modules: ['nppist'] + cudart, required: true)
cfitsio = dependency('cfitsio',                               required: false,
                     fallback: ['cfitsio', 'cfitsio_dep'],
                     default_options: ['warning_level=0'])
popt    = dependency('popt',                                  required: true,
                     fallback: ['popt',    'popt_dep'],
                     default_options: ['c_std=gnu99,c99',
                                       'warning_level=0'])


#
# CASACore >= 3.7.0 has a requirement on C++17, and we must promote the C++
# standard version to at least that version if we find such a new CASACore
# and the Meson-selected C++ standard is too old.
#
# If our compiler or system does not support it, fail-fast at configure time
# rather than allowing ourselves to proceed to a build-time failure.
#
# To do this, test-compile and -link any snippet of code featuring CASACore
# things liable to break under C++11. Thus far, the includes alone were
# discriminative-enough.
#
# There is a further complication. On some distributions, notably Ubuntu
# 22.04, no pkgconfig file is distributed. We are forced to synthesize
# a dependency ourselves, provided we can find it.
#
CASACORE_CPP17_TEST_SNIPPET = '''
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableRecord.h>
int main(void){return 0;}
'''

casacore = dependency('casacore', version: '>=3', required: false)
if not casacore.found()
  warning('Could not find CASACore through system dependencies, possibly '+
          'because the pkgconfig file casacore.pc is not installed. '+
          'Searching for CASACore manually.')
  cpp.check_header('casacore/casa/version.h')
  casacore_version_major = cpp.compute_int('CASACORE_MAJOR_VERSION',
      prefix: '#include <casacore/casa/version.h>',
      low: 0, guess: 10, high: 1000)
  casacore_version_minor = cpp.compute_int('CASACORE_MINOR_VERSION',
      prefix: '#include <casacore/casa/version.h>',
      low: 0, guess: 10, high: 1000)
  casacore_version_patch = cpp.compute_int('CASACORE_PATCH_VERSION',
      prefix: '#include <casacore/casa/version.h>',
      low: 0, guess: 10, high: 1000)

  casacore_libs = []
  foreach casacore_libname : [
      'casa', 'tables', 'scimath_f', 'scimath', 'measures', 'meas', 'fits',
      'ms', 'msfits', 'derivedmscal', 'lattices', 'mirlib', 'coordinates',
      'images', 'python3'
  ]
    casacore_lib   = cpp.find_library(f'casa_@casacore_libname@',
        required: casacore_libname in ['casa', 'tables'],
    )
    casacore_libs += casacore_lib.found() ? [casacore_lib] : []
  endforeach

  casacore          = declare_dependency(
      dependencies: casacore_libs,
      version:      f'@casacore_version_major@.'+
                    f'@casacore_version_minor@.'+
                    f'@casacore_version_patch@',
  )
endif

casacore_override = {'cpp_std': get_option('cpp_std')}
casacore_version  = casacore.version()
if casacore_version.version_compare('>=3.7.0')
  if   casacore_override['cpp_std'] in ['none',  'c++98', 'c++0x', 'c++03',
                                        'c++1y', 'c++11', 'c++14']
    casacore_override = {'cpp_std'  :   'c++17'}
  elif casacore_override['cpp_std'] in ['gnu++98', 'gnu++0x', 'gnu++03',
                                        'gnu++1y', 'gnu++11', 'gnu++14']
    casacore_override = {'cpp_std'  :   'gnu++17'}
  elif casacore_override['cpp_std'] in ['vc++11', 'vc++14']
    casacore_override = {'cpp_std'  :   'vc++17'}
  endif

  # Forced test for "latest", since we don't know what standard it refers to.
  # Otherwise, test only if a C++ standard promotion was found to be needed.
  if (casacore_override['cpp_std'].endswith('latest') or
      casacore_override['cpp_std'] != get_option('cpp_std'))
    casacore_override_std_flag = casacore_override['cpp_std'].substring(
        casacore_override['cpp_std'].startswith('vc++') ? 1 : 0
    )
    if cpp.get_argument_syntax() == 'msvc'
      casacore_override_std_flag = '/std:'+casacore_override_std_flag
    else
      casacore_override_std_flag = '-std='+casacore_override_std_flag
    endif

    assert(
        cpp.compiles(CASACORE_CPP17_TEST_SNIPPET,
            args:         [casacore_override_std_flag],
            dependencies: [casacore],
            name:         'CASACore C++17 Test Snippet'),
        f'We found CASACore @casacore_version@ >= 3.7.0, which '+
        f'requires C++17 support. However, Meson was given (or determined) '+
        f'-Dcpp_std='+get_option('cpp_std')+' and we are unable to compile '+
        f'under the higher C++17 standard (attempted with flag: '+
        f'"@casacore_override_std_flag@"). '+
        f'Aborting the build, because compiler is too old!'
    )
    assert(
        cpp.links   (CASACORE_CPP17_TEST_SNIPPET,
            args:         [casacore_override_std_flag],
            dependencies: [casacore],
            name:         'CASACore C++17 Test Snippet'),
        f'We found CASACore @casacore_version@ >= 3.7.0, which '+
        f'requires C++17 support. However, Meson was given (or determined) '+
        f'-Dcpp_std='+get_option('cpp_std')+' and we are unable to link '+
        f'under the higher C++17 standard (attempted with flag: '+
        f'"@casacore_override_std_flag@"). '+
        f'Aborting the build, likely because linker-found standard C++ '+
        f'libraries are too old!'
    )
    message(f'Found CASACore @casacore_version@ >= 3.7.0, requiring C++17!')
    message(f'Forcing promotion of C++ standard to C++17 with flag '+
            f'"@casacore_override_std_flag@"!')
  else
    message(f'Found CASACore @casacore_version@ >= 3.7.0, requiring C++17!')
    message(f'Compiling with C++ standard -Dcpp_std='+get_option('cpp_std')+'.')
  endif
else
  message(f'Found CASACore @casacore_version@ < 3.7.0, requiring C++11!')
  message(f'Compiling with C++ standard -Dcpp_std='+get_option('cpp_std')+'.')
endif


#
# With critical dependencies found, collect FIP headers and source files,
# and compile them.
#
subdir('include')
subdir('src')
